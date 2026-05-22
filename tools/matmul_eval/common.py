from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from op_eval.common import *

ADV_DB_SIZE = 2
ADV_DATA_SIZE_FP32 = 4
ADV_BASIC_BLOCK_16 = 16
ADV_BASIC_BLOCK_64 = 64
ADV_BASIC_BLOCK_128 = 128
ADV_BASIC_BLOCK_256 = 256
ADV_BASIC_BLOCK_K_128_BYTE = 128
ADV_BASIC_BLOCK_K_256_BYTE = 256
ADV_BLOCK_BYTE_SIZE = 32
ADV_CACHELINE = 512
ADV_BASIC_L1_BUFFER_NUM = 4
ADV_MIN_TAIL_BLOCK_SIZE = 1024
ADV_CUBE_BOUND_RATIO = 0.85
ADV_STREAM_K_MIN_K_THRESHOLD = 8192


@dataclass(frozen=True)
class MatmulSpec:
    m: int
    n: int
    k: int
    batch: int
    trans_a: bool
    trans_b: bool
    a_format: str = "ND"
    b_format: str = "ND"
    output_format: str = "ND"
    a_storage_elements: int | None = None
    b_storage_elements: int | None = None
    output_storage_elements: int | None = None


@dataclass(frozen=True)
class TileEstimate:
    base_m: int
    base_n: int
    base_k: int
    db_l0c: int
    tile_m: int
    tile_n: int
    tile_k: int
    mn_tile_count: int
    tile_count: int
    used_core_num: int
    core_efficiency: float
    tail_efficiency: float
    aligned_flops: int
    gm_bytes_min: int
    gm_bytes_tiled_raw: int
    gm_bytes_tiled: int
    compute_us: float | None
    hbm_us: float
    lower_bound_us: float
    source: str = "analytic_search"
    runtime_kb_id: str = ""
    runtime_kb_file: str = ""
    tiling_enable: int | None = None
    depth_a1: int | None = None
    depth_b1: int | None = None
    step_m: int | None = None
    step_n: int | None = None
    step_ka: int | None = None
    step_kb: int | None = None
    l2_m_tile: int | None = None
    l2_n_tile: int | None = None
    tiling_strategy: str = ""
    full_load: str = ""
    l0c2out: str = ""
    asw_window_len: int | None = None
    l1_buffer_num: int | None = None
    ub_db: int | None = None
    tiling_split_core: int | None = None
    tiling_full_load: int | None = None
    tiling_fix_opti: int | None = None
    tiling_special_opti: int | None = None


@dataclass(frozen=True)
class QuantSpec:
    is_quant: bool
    mode: str = "none"
    granularity: str = "none"
    compute_path: str = "non_quant"
    aux_elements: int = 0
    aux_bytes: int = 0
    notes: str = ""


@dataclass(frozen=True)
class RuntimeKbEntry:
    source_file: str
    entry_id: str
    key: tuple[Any, ...]
    info: dict[str, Any]
    knowledge: dict[str, Any]


def batch_dims_for_format(shape: list[int], tensor_format: str) -> list[int]:
    if tensor_format == "FRACTAL_NZ":
        return shape[:-4] if len(shape) >= 4 else []
    return shape[:-2] if len(shape) >= 2 else []


def effective_matrix_dims(shape: list[int], tensor_format: str) -> tuple[int, int] | None:
    if tensor_format == "FRACTAL_NZ":
        if len(shape) < 4:
            return None
        # Matches MatMulV3 GetInputDims for NZ storage:
        # dim0 = storage[-3] * storage[-2], dim1 = storage[-4] * storage[-1].
        return shape[-3] * shape[-2], shape[-4] * shape[-1]
    if len(shape) < 2:
        return None
    return shape[-2], shape[-1]


def broadcast_batch(lhs: list[int], rhs: list[int]) -> int | None:
    rank = max(len(lhs), len(rhs))
    lhs = [1] * (rank - len(lhs)) + lhs
    rhs = [1] * (rank - len(rhs)) + rhs

    batch = 1
    for left, right in zip(lhs, rhs):
        if left == 1:
            batch *= right
        elif right == 1 or left == right:
            batch *= left
        else:
            return None
    return batch


def reconcile_k_dim(left_k: int, right_k: int, left_format: str, right_format: str) -> tuple[int, float] | None:
    if left_k == right_k:
        return left_k, 2.0
    if right_format == "FRACTAL_NZ" and right_k >= left_k and ceil_align(left_k, 16) == right_k:
        return left_k, 1.0
    if left_format == "FRACTAL_NZ" and left_k >= right_k and ceil_align(right_k, 16) == left_k:
        return right_k, 1.0
    return None


def output_dim_score(candidate: int, actual: int | None, tensor_format: str) -> tuple[int, float] | None:
    if actual is None:
        return candidate, 0.0
    if candidate == actual:
        return actual, 2.0
    if tensor_format == "FRACTAL_NZ" and candidate >= actual and ceil_align(actual, 16) == candidate:
        return actual, 1.0
    return None


def infer_matmul_spec(
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    input_formats: list[str] | None = None,
    output_formats: list[str] | None = None,
) -> MatmulSpec | None:
    if len(input_shapes) < 2:
        return None
    input_formats = input_formats or []
    output_formats = output_formats or []
    lhs, rhs = input_shapes[0], input_shapes[1]
    lhs_format = format_at(input_formats, 0)
    rhs_format = format_at(input_formats, 1)
    output_format = format_at(output_formats, 0)

    lhs_dims = effective_matrix_dims(lhs, lhs_format)
    rhs_dims = effective_matrix_dims(rhs, rhs_format)
    if lhs_dims is None or rhs_dims is None:
        return None

    out_dims: tuple[int, int] | None = None
    if output_shapes:
        out_dims = effective_matrix_dims(output_shapes[0], output_format)
    out_m = out_dims[0] if out_dims is not None else None
    out_n = out_dims[1] if out_dims is not None else None

    batch = broadcast_batch(batch_dims_for_format(lhs, lhs_format), batch_dims_for_format(rhs, rhs_format))
    if batch is None:
        return None

    candidates: list[tuple[float, MatmulSpec]] = []
    for trans_a in (False, True):
        lhs_m_raw, lhs_k_raw = lhs_dims if not trans_a else (lhs_dims[1], lhs_dims[0])
        for trans_b in (False, True):
            rhs_k_raw, rhs_n_raw = rhs_dims if not trans_b else (rhs_dims[1], rhs_dims[0])
            k_match = reconcile_k_dim(lhs_k_raw, rhs_k_raw, lhs_format, rhs_format)
            if k_match is None:
                continue
            logical_k, k_score = k_match

            m_match = output_dim_score(lhs_m_raw, out_m, lhs_format)
            n_match = output_dim_score(rhs_n_raw, out_n, rhs_format)
            if m_match is None or n_match is None:
                continue
            logical_m, m_score = m_match
            logical_n, n_score = n_match

            score = k_score + m_score + n_score
            if not trans_a:
                score += 0.1
            if not trans_b:
                score += 0.1
            if lhs_format == "FRACTAL_NZ" or rhs_format == "FRACTAL_NZ":
                score += 0.5

            candidates.append(
                (
                    score,
                    MatmulSpec(
                        logical_m,
                        logical_n,
                        logical_k,
                        batch,
                        trans_a,
                        trans_b,
                        a_format=lhs_format,
                        b_format=rhs_format,
                        output_format=output_format,
                        a_storage_elements=num_elements(lhs),
                        b_storage_elements=num_elements(rhs),
                        output_storage_elements=num_elements(output_shapes[0]) if output_shapes else None,
                    ),
                )
            )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def infer_standard_matmul(
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
) -> MatmulSpec | None:
    if len(input_shapes) < 2:
        return None
    lhs, rhs = input_shapes[0], input_shapes[1]
    if len(lhs) < 2 or len(rhs) < 2:
        return None

    out = output_shapes[0] if output_shapes else []
    candidates: list[tuple[float, MatmulSpec]] = []
    for trans_a in (False, True):
        lhs_m, lhs_k = (lhs[-2], lhs[-1]) if not trans_a else (lhs[-1], lhs[-2])
        for trans_b in (False, True):
            rhs_k, rhs_n = (rhs[-2], rhs[-1]) if not trans_b else (rhs[-1], rhs[-2])
            batch = broadcast_batch(lhs[:-2], rhs[:-2])
            if batch is None or lhs_k != rhs_k:
                continue

            score = 0.0
            if len(out) >= 2:
                score += 2.0 if out[-2] == lhs_m else -10.0
                score += 2.0 if out[-1] == rhs_n else -10.0
            if not trans_a:
                score += 0.1
            if not trans_b:
                score += 0.1
            candidates.append((score, MatmulSpec(lhs_m, rhs_n, lhs_k, batch, trans_a, trans_b)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def infer_transpose_batch_matmul_spec(
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    input_formats: list[str] | None = None,
    output_formats: list[str] | None = None,
) -> MatmulSpec | None:
    if len(input_shapes) < 2:
        return None
    input_formats = input_formats or []
    lhs_format = format_at(input_formats, 0)
    rhs_format = format_at(input_formats, 1)
    if lhs_format != "ND" or rhs_format != "ND":
        return None
    lhs, rhs = input_shapes[0], input_shapes[1]
    if len(lhs) < 3 or len(rhs) < 3:
        return None
    if lhs[0] != rhs[0] or lhs[-1] != rhs[-2]:
        return None
    batch, m, k = lhs[0], lhs[-2], lhs[-1]
    n = rhs[-1]
    if output_shapes:
        output_elems = num_elements(output_shapes[0])
        if output_elems and output_elems != batch * m * n:
            return None
    return MatmulSpec(
        m,
        n,
        k,
        batch,
        trans_a=False,
        trans_b=False,
        a_format=lhs_format,
        b_format=rhs_format,
        output_format=format_at(output_formats or [], 0),
        a_storage_elements=num_elements(lhs),
        b_storage_elements=num_elements(rhs),
        output_storage_elements=num_elements(output_shapes[0]) if output_shapes else None,
    )


def infer_grouped_matmul_spec(
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    input_formats: list[str] | None = None,
    output_formats: list[str] | None = None,
) -> MatmulSpec | None:
    """Infer logical work for GroupedMatmul.

    CANN GroupedMatmul stores one weight matrix per expert, but each token is
    routed to a subset of experts. The first NZ dimension is therefore expert
    count, not a batch dimension to multiply into the logical GEMM work.
    """

    if len(input_shapes) < 2:
        return None
    input_formats = input_formats or []
    output_formats = output_formats or []
    lhs, rhs = input_shapes[0], input_shapes[1]
    lhs_format = format_at(input_formats, 0)
    rhs_format = format_at(input_formats, 1)
    output_format = format_at(output_formats, 0)
    if lhs_format != "ND" or rhs_format != "FRACTAL_NZ" or len(lhs) < 2 or len(rhs) < 4:
        return None

    lhs_dims = effective_matrix_dims(lhs, lhs_format)
    rhs_dims = effective_matrix_dims(rhs, rhs_format)
    if lhs_dims is None or rhs_dims is None:
        return None
    m, lhs_k = lhs_dims
    rhs_k, rhs_n = rhs_dims
    k_match = reconcile_k_dim(lhs_k, rhs_k, lhs_format, rhs_format)
    if k_match is None:
        return None
    k, _ = k_match

    n = rhs_n
    if output_shapes:
        out_dims = effective_matrix_dims(output_shapes[0], output_format)
        if out_dims is None:
            return None
        out_m, out_n = out_dims
        if out_m != m:
            return None
        n_match = output_dim_score(n, out_n, rhs_format)
        if n_match is None:
            return None
        n, _ = n_match

    return MatmulSpec(
        m,
        n,
        k,
        1,
        trans_a=False,
        trans_b=False,
        a_format=lhs_format,
        b_format=rhs_format,
        output_format=output_format,
        a_storage_elements=num_elements(lhs),
        b_storage_elements=num_elements(rhs),
        output_storage_elements=num_elements(output_shapes[0]) if output_shapes else None,
    )


def is_matmul_row(row: dict[str, str]) -> bool:
    text = row.get("Type", "").lower()
    return any(token in text for token in ("matmul", "mat_mul", "batchmatmul", "bmm"))


def dtype_from_row(row: dict[str, str]) -> str:
    parts = [part for part in row.get("Input Data Types", "").split(";") if part]
    if not parts:
        return "UNKNOWN"
    if is_quant_kernel_type(row.get("Type", "")) and parts:
        return parts[0]
    if len(parts) >= 2 and parts[0] != parts[1]:
        return "MIXED"
    return parts[0]


def output_dtype_from_row(row: dict[str, str]) -> str:
    parts = [part for part in row.get("Output Data Types", "").split(";") if part]
    return parts[0] if parts else dtype_from_row(row)


def input_dtypes_from_row(row: dict[str, str]) -> list[str]:
    return split_semicolon_values(row.get("Input Data Types"))


def is_ops_nn_v3_kernel_type(kernel_type: str) -> bool:
    text = kernel_type.lower()
    if "quant" in text:
        return False
    return text in {"matmulv3", "batchmatmulv3", "batch_mat_mul_v3", "mat_mul_v3"}


def is_batch_matmul_kernel_type(kernel_type: str) -> bool:
    text = kernel_type.lower()
    return "batch" in text and "matmul" in text and "quant" not in text
