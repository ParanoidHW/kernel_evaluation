#!/usr/bin/env python3
"""Evaluate Ascend matmul profiling rows with a kernel-aware analytic model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DTYPE_BYTES = {
    "FLOAT16": 2,
    "DT_FLOAT16": 2,
    "BFLOAT16": 2,
    "DT_BF16": 2,
    "FLOAT": 4,
    "FLOAT32": 4,
    "DT_FLOAT": 4,
    "INT8": 1,
    "DT_INT8": 1,
    "UINT8": 1,
    "INT32": 4,
    "DT_INT32": 4,
    "INT64": 8,
    "UINT64": 8,
    "INT4": 1,
    "UINT4": 1,
    "FLOAT4": 1,
    "FLOAT4_E2M1": 1,
    "FLOAT8": 1,
    "FLOAT8_E4M3FN": 1,
    "FLOAT8_E5M2": 1,
    "HIFLOAT8": 1,
    "HIF8": 1,
    "FP8": 1,
    "MXFP8": 1,
}

DTYPE_BITS = {
    "INT4": 4,
    "UINT4": 4,
    "INT8": 8,
    "DT_INT8": 8,
    "UINT8": 8,
    "FLOAT4": 4,
    "FLOAT4_E2M1": 4,
    "FLOAT8": 8,
    "FLOAT8_E4M3FN": 8,
    "FLOAT8_E5M2": 8,
    "HIFLOAT8": 8,
    "HIF8": 8,
    "FP8": 8,
    "MXFP8": 8,
}

GE_DTYPE_RUNTIME_KB = {
    "FLOAT": 0,
    "FLOAT32": 0,
    "DT_FLOAT": 0,
    "DT_FLOAT32": 0,
    "FLOAT16": 1,
    "DT_FLOAT16": 1,
    "BFLOAT16": 1,
    "DT_BF16": 1,
    "BF16": 1,
}

GE_FORMAT_RUNTIME_KB = {
    "ND": 2,
    "FRACTAL_NZ": 29,
}

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


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def ceil_align(value: int, align: int) -> int:
    if align <= 0:
        return value
    return ceil_div(value, align) * align


def floor_align(value: int, align: int) -> int:
    if align <= 0:
        return value
    return value // align * align


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text == "N/A":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    return int(parse_float(value, float(default)))


def parse_shapes(value: str | None) -> list[list[int]]:
    if not value or value == "N/A":
        return []
    text = value.strip().replace('"', "")
    shapes: list[list[int]] = []
    for part in text.split(";"):
        if not part:
            continue
        dims = [int(num) for num in re.findall(r"-?\d+", part)]
        if dims:
            shapes.append(dims)
    return shapes


def parse_formats(value: str | None) -> list[str]:
    if not value or value == "N/A":
        return []
    text = value.strip().replace('"', "")
    return [normalize_format(part.strip()) for part in text.split(";") if part.strip()]


def normalize_format(value: str | None) -> str:
    if not value or value == "N/A":
        return "ND"
    text = value.strip().upper().replace("FORMAT_", "")
    if text in {"NZ", "FRACTAL_NZ"}:
        return "FRACTAL_NZ"
    # MatMulV3 tiling maps every non-FRACTAL_NZ storage format to ND.
    return "ND"


def format_at(formats: list[str], index: int) -> str:
    return formats[index] if index < len(formats) else "ND"


def num_elements(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        if dim <= 0:
            return 0
        total *= dim
    return total


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


def is_matmul_row(row: dict[str, str]) -> bool:
    text = row.get("Type", "").lower()
    return any(token in text for token in ("matmul", "mat_mul", "batchmatmul", "bmm"))


def is_excluded_by_default(row: dict[str, str], include_gmm: bool, include_allgather: bool) -> bool:
    text = f"{row.get('Name', '')} {row.get('Type', '')}".lower()
    if not include_gmm and "groupedmatmul" in text:
        return True
    if not include_allgather and "allgathermatmul" in text:
        return True
    return False


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


def dtype_size(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype, DTYPE_BYTES.get(dtype.replace("DT_", ""), 4))


def dtype_bitwidth(dtype: str) -> int:
    normalized = dtype.upper().replace("DT_", "")
    if dtype.upper() in DTYPE_BITS:
        return DTYPE_BITS[dtype.upper()]
    return DTYPE_BITS.get(normalized, dtype_size(dtype) * 8)


def is_quantized_data_dtype(dtype: str) -> bool:
    normalized = dtype.upper().replace("DT_", "")
    return normalized in {
        "INT4",
        "UINT4",
        "INT8",
        "UINT8",
        "FLOAT4",
        "FLOAT4_E2M1",
        "FLOAT8",
        "FLOAT8_E4M3FN",
        "FLOAT8_E5M2",
        "HIFLOAT8",
        "HIF8",
        "FP8",
        "MXFP8",
    }


def is_scale_dtype(dtype: str) -> bool:
    normalized = dtype.upper().replace("DT_", "")
    return normalized in {"FLOAT", "FLOAT32", "FLOAT16", "BFLOAT16", "BF16"}


def is_quant_kernel_type(kernel_type: str) -> bool:
    return "quant" in kernel_type.lower()


def is_ops_nn_v3_kernel_type(kernel_type: str) -> bool:
    text = kernel_type.lower()
    if "quant" in text:
        return False
    return text in {"matmulv3", "batchmatmulv3", "batch_mat_mul_v3", "mat_mul_v3"}


def is_batch_matmul_kernel_type(kernel_type: str) -> bool:
    text = kernel_type.lower()
    return "batch" in text and "matmul" in text and "quant" not in text


def peak_for_dtype(config: dict[str, Any], dtype: str) -> float | None:
    peaks = config.get("peak_tflops", {})
    candidates = [dtype, dtype.replace("DT_", "")]
    if dtype == "DT_BF16":
        candidates.append("BFLOAT16")
    for candidate in candidates:
        value = peaks.get(candidate)
        if value is not None:
            return float(value)
    return None


def calibration_value(config: dict[str, Any], section: str, key: str, default: float) -> float:
    calibration = config.get("calibration", {})
    mapping = calibration.get(section, {})
    if key in mapping:
        return float(mapping[key])
    return float(mapping.get("default", default))


def decode_tiling_enable(value: int | None) -> dict[str, int | None]:
    if value is None or value < 0:
        return {
            "split_core": None,
            "full_load": None,
            "fix_opti": None,
            "special_opti": None,
        }
    return {
        "split_core": value % 10,
        "full_load": (value // 10) % 10,
        "fix_opti": (value // 1000) % 10,
        "special_opti": (value // 10000) % 10,
    }


def tiling_full_load_name(value: int | None) -> str:
    return {0: "NONE_FULL_LOAD", 1: "A_FULL_LOAD", 2: "B_FULL_LOAD"}.get(value, "")


def runtime_kb_dtype_code(dtype: str) -> int | None:
    normalized = dtype.upper().replace("DT_", "")
    candidates = [dtype.upper(), normalized]
    if normalized == "BF16":
        candidates.extend(["BFLOAT16", "DT_BF16"])
    for candidate in candidates:
        if candidate in GE_DTYPE_RUNTIME_KB:
            return GE_DTYPE_RUNTIME_KB[candidate]
    return None


def runtime_kb_format_code(tensor_format: str) -> int | None:
    return GE_FORMAT_RUNTIME_KB.get(tensor_format)


def runtime_kb_aligned_spec(spec: MatmulSpec, dtype: str) -> tuple[int, int, int, bool, bool, bool]:
    reduce_align = 8 if is_fp32_dtype(dtype) else 16
    aligned_m = ceil_align(spec.m, 16)
    aligned_n = ceil_align(spec.n, 16)
    aligned_k = ceil_align(spec.k, reduce_align)
    return aligned_m, aligned_n, aligned_k, spec.m == aligned_m, spec.n == aligned_n, spec.k == aligned_k


def runtime_kb_key_from_parts(
    a_dtype_code: int,
    b_dtype_code: int,
    out_dtype_code: int,
    a_format_code: int,
    b_format_code: int,
    out_format_code: int,
    m: int,
    n: int,
    k: int,
    m_align: bool,
    n_align: bool,
    k_align: bool,
    trans_a: bool,
    trans_b: bool,
    bias: bool,
) -> tuple[Any, ...]:
    return (
        a_dtype_code,
        b_dtype_code,
        out_dtype_code,
        a_format_code,
        b_format_code,
        out_format_code,
        m,
        n,
        k,
        bool(m_align),
        bool(n_align),
        bool(k_align),
        bool(trans_a),
        bool(trans_b),
        bool(bias),
    )


def runtime_kb_key_from_entry(info: dict[str, Any]) -> tuple[Any, ...]:
    return runtime_kb_key_from_parts(
        int(info.get("a_dtype", -1)),
        int(info.get("b_dtype", -1)),
        int(info.get("out_dtype", -1)),
        int(info.get("a_format", -1)),
        int(info.get("b_format", -1)),
        int(info.get("out_format", -1)),
        int(info.get("m", -1)),
        int(info.get("n", -1)),
        int(info.get("k", -1)),
        bool(info.get("m_align_flag", False)),
        bool(info.get("n_align_flag", False)),
        bool(info.get("k_align_flag", False)),
        bool(info.get("trans_a_flag", False)),
        bool(info.get("trans_b_flag", False)),
        bool(info.get("bias_flag", False)),
    )


def runtime_kb_key_from_row(
    spec: MatmulSpec,
    input_dtypes: list[str],
    output_dtype: str,
) -> tuple[Any, ...] | None:
    if spec.batch != 1:
        return None
    a_dtype = input_dtypes[0] if input_dtypes else "UNKNOWN"
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else a_dtype
    a_dtype_code = runtime_kb_dtype_code(a_dtype)
    b_dtype_code = runtime_kb_dtype_code(b_dtype)
    out_dtype_code = runtime_kb_dtype_code(output_dtype)
    a_format_code = runtime_kb_format_code(spec.a_format)
    b_format_code = runtime_kb_format_code(spec.b_format)
    out_format_code = runtime_kb_format_code(spec.output_format)
    if None in {a_dtype_code, b_dtype_code, out_dtype_code, a_format_code, b_format_code, out_format_code}:
        return None
    aligned_m, aligned_n, aligned_k, m_align, n_align, k_align = runtime_kb_aligned_spec(spec, a_dtype)
    bias = len(input_dtypes) >= 3 and is_scale_dtype(input_dtypes[2])
    return runtime_kb_key_from_parts(
        a_dtype_code,
        b_dtype_code,
        out_dtype_code,
        a_format_code,
        b_format_code,
        out_format_code,
        aligned_m,
        aligned_n,
        aligned_k,
        m_align,
        n_align,
        k_align,
        spec.trans_a,
        spec.trans_b,
        bias,
    )


def expand_config_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if any(char in pattern for char in "*?[]"):
            paths.extend(sorted(Path.cwd().glob(pattern)))
        elif pattern_path.exists():
            paths.append(pattern_path)
    return sorted(set(paths))


def load_runtime_kb(config: dict[str, Any]) -> dict[tuple[Any, ...], list[RuntimeKbEntry]]:
    kernel_model = config.get("kernel_model", {})
    runtime_cfg = kernel_model.get("runtime_kb", {})
    if not runtime_cfg.get("enabled", False):
        return {}

    index: dict[tuple[Any, ...], list[RuntimeKbEntry]] = {}
    for path in expand_config_paths(list(runtime_cfg.get("matmul_v3_paths", []))):
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                info = record.get("info_dict", {})
                knowledge = record.get("knowledge", {})
                key = runtime_kb_key_from_entry(info)
                entry = RuntimeKbEntry(
                    source_file=display_path(path),
                    entry_id=str(record.get("id", f"{path.name}:{line_no}")),
                    key=key,
                    info=info,
                    knowledge=knowledge,
                )
                index.setdefault(key, []).append(entry)
    return index


def candidate_base_values(limit: int) -> list[int]:
    max_value = max(16, min(512, ceil_align(limit, 16)))
    values = list(range(16, max_value + 1, 16))
    preferred = [64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 336]
    return sorted(set(values + [value for value in preferred if value <= max_value]))


def source_base_values(limit: int, max_base: int = ADV_BASIC_BLOCK_256, align: int = ADV_BASIC_BLOCK_16) -> list[int]:
    align = max(ADV_BASIC_BLOCK_16, align)
    max_value = max(ADV_BASIC_BLOCK_16, min(max_base, ceil_align(limit, ADV_BASIC_BLOCK_16)))
    values = [value for value in range(align, max_value + 1, align) if value % ADV_BASIC_BLOCK_16 == 0]
    if ADV_BASIC_BLOCK_16 not in values:
        values.append(ADV_BASIC_BLOCK_16)
    return sorted(set(values))


def storage_element_counts(spec: MatmulSpec) -> tuple[int, int, int]:
    logical_a = spec.batch * spec.m * spec.k
    logical_b = spec.batch * spec.k * spec.n
    logical_c = spec.batch * spec.m * spec.n
    return (
        spec.a_storage_elements or logical_a,
        spec.b_storage_elements or logical_b,
        spec.output_storage_elements or logical_c,
    )


def l2_aware_gm_bytes(gm_bytes_min: int, gm_bytes_tiled_raw: int, config: dict[str, Any]) -> int:
    l2_bytes = int(config.get("l2_bytes", 0))
    if l2_bytes > 0 and gm_bytes_min <= l2_bytes:
        return gm_bytes_min
    if l2_bytes > 0:
        redundant = max(0, gm_bytes_tiled_raw - gm_bytes_min)
        l2_pressure = max(0.0, 1.0 - l2_bytes / max(gm_bytes_min, 1))
        return int(gm_bytes_min + redundant * l2_pressure)
    return gm_bytes_tiled_raw


def asw_window_len(aic_num: int) -> int:
    sqrt_num = int(math.sqrt(max(aic_num, 1)))
    for factor in range(sqrt_num, 0, -1):
        if aic_num % factor == 0:
            return factor
    return 1


def advanced_base_k(
    spec: MatmulSpec,
    elem_size: int,
    l0a: int,
    base_m: int,
    base_n: int,
) -> int:
    k_value_align = ceil_align(spec.k, ADV_BASIC_BLOCK_16)
    max_base_k = l0a // ADV_DB_SIZE // max(elem_size, 1) // max(base_m, base_n, 1)
    if k_value_align <= max_base_k:
        return max(ADV_BASIC_BLOCK_16, k_value_align)
    if spec.trans_a and not spec.trans_b:
        return max(ADV_BASIC_BLOCK_16, floor_align(max_base_k, ADV_BASIC_BLOCK_16))
    align_256b = max(ADV_BASIC_BLOCK_16, ADV_BASIC_BLOCK_K_256_BYTE // max(elem_size, 1))
    if max_base_k * elem_size >= ADV_BASIC_BLOCK_K_256_BYTE:
        return max(ADV_BASIC_BLOCK_16, floor_align(max_base_k, align_256b))
    for candidate in (128, 64, 32, 16):
        if max_base_k >= candidate:
            return candidate
    return ADV_BASIC_BLOCK_16


def advanced_cal_l1_tiling(
    spec: MatmulSpec,
    elem_size: int,
    l1: int,
    base_m: int,
    base_n: int,
    base_k: int,
    single_core_k: int,
) -> tuple[int, int, int, int, int, int]:
    depth_a1 = max(1, l1 // ADV_DB_SIZE // max(base_m, 1) // max(base_k, 1) // max(elem_size, 1))
    depth_b1 = max(1, l1 // ADV_DB_SIZE // max(base_n, 1) // max(base_k, 1) // max(elem_size, 1))
    depth_a_size = depth_a1 * base_m * base_k * elem_size
    depth_b_size = depth_b1 * base_n * base_k * elem_size
    if depth_a_size + depth_b_size > l1:
        if base_m <= base_n:
            depth_a1 = max(depth_a1 // ADV_DB_SIZE, 1)
        else:
            depth_b1 = max(depth_b1 // ADV_DB_SIZE, 1)

    step_ka = max(depth_a1 // ADV_DB_SIZE, 1)
    step_kb = max(depth_b1 // ADV_DB_SIZE, 1)
    if (
        base_m == ADV_BASIC_BLOCK_256
        and base_n == ADV_BASIC_BLOCK_256
        and spec.m % ADV_BASIC_BLOCK_16 == 0
        and spec.n % ADV_BASIC_BLOCK_16 == 0
        and spec.k % ADV_BASIC_BLOCK_16 == 0
        and single_core_k <= ADV_BASIC_BLOCK_256
    ):
        step_ka = min(step_ka, 2)
        step_kb = min(step_kb, 2)

    if step_ka >= step_kb:
        step_ka = max((step_ka // max(step_kb, 1)) * step_kb, 1)
    else:
        step_kb = max((step_kb // max(step_ka, 1)) * step_ka, 1)
    return step_ka * ADV_DB_SIZE, step_kb * ADV_DB_SIZE, 1, 1, step_ka, step_kb


def advanced_step_small_k(
    spec: MatmulSpec,
    dtype: str,
    elem_size: int,
    base_k: int,
    step_ka: int,
    step_kb: int,
    is_bl1_full_load: bool,
) -> int:
    step_big_k = step_kb if is_bl1_full_load else step_ka
    step_small_k = step_ka if is_bl1_full_load else step_kb
    is_trans = spec.trans_a if is_bl1_full_load else spec.trans_b
    step_small_k = max(step_small_k, 1)
    is_small_tail = (step_big_k % step_small_k) / step_small_k <= 0.25
    is_small_tail = (is_small_tail and not is_trans) or base_k * elem_size >= ADV_BASIC_BLOCK_K_256_BYTE
    if is_fp32_dtype(dtype):
        return 1
    if is_small_tail:
        return 2
    return step_small_k


def advanced_l0c2out(
    spec: MatmulSpec,
    output_dtype: str,
    aic_num: int,
    aiv_num: int,
    single_core_m: int,
    single_core_n: int,
    dtype: str,
) -> str:
    is_valid_mkn = spec.k <= ADV_BASIC_BLOCK_256 and spec.m >= ADV_BASIC_BLOCK_256
    m_cnt = ceil_div(spec.m, max(single_core_m, 1))
    n_cnt = ceil_div(spec.n, max(single_core_n, 1))
    is_multi_round = m_cnt * n_cnt >= 2 * aic_num
    c_size = dtype_size(output_dtype)
    is_unaligned_n = spec.n * c_size % 128 != 0 and spec.n * c_size > ADV_BASIC_BLOCK_256
    fixpipe_bound = is_valid_mkn and is_multi_round and is_unaligned_n
    if not fixpipe_bound or aiv_num != aic_num * 2:
        return "ON_THE_FLY"
    if dtype in {"FLOAT16", "DT_FLOAT16", "DT_BF16", "BFLOAT16"}:
        return "ND_FIXPIPE_1_1"
    return "ND_FIXPIPE_1_2"


def advanced_stream_k_kind(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
    kernel_type: str,
) -> str | None:
    aic_num = int(config["aic_num"])
    aiv_num = int(config.get("aiv_num", aic_num * 2))
    elem_size = dtype_size(dtype)
    if aiv_num != aic_num * 2 or spec.a_format != "ND":
        return None

    align_value = ADV_BLOCK_BYTE_SIZE if is_fp32_dtype(dtype) else ADV_BASIC_BLOCK_256
    k_threshold_sk = max(ADV_STREAM_K_MIN_K_THRESHOLD, aic_num * ADV_BASIC_BLOCK_K_256_BYTE) // max(elem_size, 1)
    k_large_enough = ceil_align(spec.k, ADV_BASIC_BLOCK_256) >= k_threshold_sk

    if is_batch_matmul_kernel_type(kernel_type):
        if is_fp32_dtype(dtype) and spec.k > 2_000_000:
            return None
        if not k_large_enough:
            return None
        m_cnt = ceil_div(spec.m, align_value)
        n_cnt = ceil_div(spec.n, align_value)
        if spec.batch * m_cnt * n_cnt <= max(1, aic_num // 2):
            return "batch_stream_k_sk"
        return None

    if k_large_enough:
        m_cnt = ceil_div(spec.m, align_value)
        n_cnt = ceil_div(spec.n, align_value)
        if m_cnt * n_cnt <= max(1, aic_num // 2):
            return "stream_k_sk"

    k_threshold_dpsk = max(ADV_STREAM_K_MIN_K_THRESHOLD, aic_num * ADV_BASIC_BLOCK_K_128_BYTE) // max(elem_size, 1)
    if spec.m % ADV_BASIC_BLOCK_256 == 0 and spec.n % ADV_BASIC_BLOCK_256 == 0 and spec.k >= k_threshold_dpsk:
        m_cnt = ceil_div(spec.m, ADV_BASIC_BLOCK_256)
        n_cnt = ceil_div(spec.n, ADV_BASIC_BLOCK_256)
        total_mn = m_cnt * n_cnt
        remainder = total_mn % aic_num
        if total_mn >= aic_num and remainder != 0 and remainder <= aic_num // 2:
            return "stream_k_dpsk"
    return None


def advanced_stream_k_tile(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    kernel_type: str,
    kind: str,
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    base_m = ADV_BASIC_BLOCK_256
    base_n = ADV_BASIC_BLOCK_256
    m_cnt = ceil_div(spec.m, base_m)
    n_cnt = ceil_div(spec.n, base_n)

    if kind == "batch_stream_k_sk":
        blocks_per_batch = max(1, aic_num // max(spec.batch, 1))
        if m_cnt > blocks_per_batch // 3 and m_cnt < blocks_per_batch // 2:
            m_cnt = max(1, blocks_per_batch // 2)
        if n_cnt > blocks_per_batch // 3 and n_cnt < blocks_per_batch // 2:
            n_cnt = max(1, blocks_per_batch // 2)
        mn_per_batch = max(1, m_cnt * n_cnt)
        k_cnt = max(1, blocks_per_batch // mn_per_batch)
    else:
        total_mn = max(1, m_cnt * n_cnt)
        if total_mn <= max(1, aic_num // 2):
            if m_cnt > aic_num // 3 and m_cnt < aic_num // 2:
                m_cnt = max(1, aic_num // 2)
            if n_cnt > aic_num // 3 and n_cnt < aic_num // 2:
                n_cnt = max(1, aic_num // 2)
            total_mn = max(1, m_cnt * n_cnt)
            k_cnt = max(1, aic_num // total_mn)
        else:
            remainder = max(1, total_mn % aic_num)
            k_cnt = max(1, aic_num // remainder)

    base_m = ceil_align(ceil_div(spec.m, max(m_cnt, 1)), ADV_BASIC_BLOCK_16)
    base_n = ceil_align(ceil_div(spec.n, max(n_cnt, 1)), ADV_BASIC_BLOCK_16)
    single_core_m = base_m
    single_core_n = base_n
    single_core_k = ceil_div(spec.k, max(k_cnt, 1))
    if spec.b_format != "ND":
        k_align = ADV_BASIC_BLOCK_16 if elem_size == 2 or (elem_size == 4 and not spec.trans_b) else ADV_BASIC_BLOCK_16 // 2
        single_core_k = ceil_align(single_core_k, max(k_align, 1))

    base_k_align = ADV_BASIC_BLOCK_128 // max(elem_size, 1) if (not spec.trans_a or spec.trans_b) else ADV_BASIC_BLOCK_16
    k_value_max = floor_align(
        int(config["l0a_bytes"]) // ADV_DB_SIZE // max(elem_size, 1) // max(base_m, base_n, 1),
        max(base_k_align, 1),
    )
    base_k = max(ADV_BASIC_BLOCK_16, min(single_core_k, max(k_value_max, ADV_BASIC_BLOCK_16)))
    depth_a1, depth_b1, step_m, step_n, step_ka, step_kb = advanced_cal_l1_tiling(
        spec, elem_size, int(config["l1_bytes"]), base_m, base_n, base_k, single_core_k
    )
    if base_m == base_n and depth_b1 == depth_a1 * 2:
        depth_a1 = depth_a1 * 2
        depth_b1 = max(1, depth_b1 // 2)
        step_kb = max(1, depth_b1 // ADV_DB_SIZE)
        step_ka = max(1, depth_a1 // ADV_DB_SIZE)

    return make_tile_estimate_from_source_tiling(
        spec=spec,
        dtype=dtype,
        output_dtype=output_dtype,
        config=config,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        single_core_m=single_core_m,
        single_core_n=single_core_n,
        single_core_k=single_core_k,
        used_core_num=min(aic_num, max(1, spec.batch * m_cnt * n_cnt * k_cnt)),
        core_work_tiles=max(1, spec.batch * m_cnt * n_cnt * k_cnt),
        source="advanced_tiling_heuristic",
        tiling_strategy=kind,
        full_load="NONE_FULL_LOAD",
        l0c2out=advanced_l0c2out(
            spec, output_dtype, aic_num, int(config.get("aiv_num", aic_num * 2)), base_m, base_n, dtype
        ),
        asw_window=asw_window_len(aic_num),
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        step_m=step_m,
        step_n=step_n,
        step_ka=step_ka,
        step_kb=step_kb,
        l1_buffer_num=ADV_DB_SIZE,
        ub_db=ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 <= int(config.get("ub_bytes", 0)) else 1,
        tiling_split_core=0,
        tiling_full_load=0,
    )


def advanced_balance_rate_with_tail(
    spec: MatmulSpec,
    used_core_num: int,
    base_m: int,
    base_n: int,
) -> float:
    total_round = spec.batch * ceil_div(spec.m, base_m) * ceil_div(spec.n, base_n)
    if total_round <= 0 or used_core_num <= 0:
        return 0.0
    main_round = ceil_div(total_round, used_core_num) - 1
    tail_blocks = total_round - used_core_num * main_round
    if tail_blocks <= 0:
        return 1.0
    if main_round == 0 or (base_m * base_n) // tail_blocks < ADV_MIN_TAIL_BLOCK_SIZE or spec.batch != 1:
        return (spec.batch * spec.m * spec.n / used_core_num) / ((main_round + 1) * base_m * base_n)
    tail_split_sqrt = max(1, int(math.sqrt(tail_blocks)))
    offset = (tail_blocks - tail_split_sqrt * tail_split_sqrt) // tail_split_sqrt + 1
    tail_round = 1.0 / (tail_split_sqrt * (tail_split_sqrt + offset - 1))
    return (spec.m * spec.n / used_core_num) / ((main_round + tail_round) * base_m * base_n)


def advanced_max_base_with_limit(
    spec: MatmulSpec,
    elem_size: int,
    config: dict[str, Any],
    base_mn_buffer_limit: int,
    base_align_unit: int,
    is_right_matrix: bool,
    is_memory_bound: bool,
) -> int:
    shape_value = spec.n if is_right_matrix else spec.m
    k_align_value = ceil_align(spec.k, ADV_BASIC_BLOCK_16)
    k_limit_value = ADV_BASIC_BLOCK_16 if is_memory_bound else ADV_BASIC_BLOCK_K_128_BYTE // max(elem_size, 1)
    min_k_l0_bytes = min(k_limit_value, k_align_value) * elem_size
    l0_size = int(config["l0b_bytes"] if is_right_matrix else config["l0a_bytes"])
    max_base_mn_with_buffer = base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // ADV_BASIC_BLOCK_16
    max_base_block = min(
        l0_size // ADV_DB_SIZE // max(min_k_l0_bytes, 1),
        max_base_mn_with_buffer,
    )
    k_align_unit = (
        (ADV_BASIC_BLOCK_K_256_BYTE if is_memory_bound and spec.batch == 1 else ADV_BASIC_BLOCK_K_256_BYTE * 2)
        // max(elem_size, 1)
        if (not spec.trans_a or spec.trans_b)
        else ADV_BASIC_BLOCK_16
    )
    max_base_mn_with_k_inner = int(config["l1_bytes"]) // (
        2 * ADV_DB_SIZE * max(elem_size, 1) * max(1, min(k_align_unit, k_align_value))
    )
    max_base_block = min(max_base_block, max_base_mn_with_k_inner)
    max_base_block = min(ceil_align(shape_value, base_align_unit), floor_align(max_base_block, base_align_unit))
    return max(ADV_BASIC_BLOCK_16, max_base_block)


def advanced_batch_rebalance_base(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
) -> tuple[int, int, int, int, float]:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    base_mn_buffer_limit = int(config["l0c_bytes"])
    core_freq_ghz = float(config.get("kernel_model", {}).get("advanced_tiling", {}).get("core_freq_ghz", 1.65))
    l2_rate = float(config.get("kernel_model", {}).get("advanced_tiling", {}).get("l2_rate", 100.0))
    hbm_bw = float(config["hbm_bandwidth_tbps"])
    l2_bw = core_freq_ghz * aic_num * l2_rate / 1024.0
    compute_power = core_freq_ghz * 8.0 * aic_num
    cmr = (spec.m + spec.n) / max(spec.m * spec.n, 1)
    l2_cache_usage = max(spec.batch * (spec.m + spec.n) * spec.k * elem_size / max(int(config.get("l2_bytes", 1)), 1), 1.0)
    cube_bound_edge = (
        (l2_bw / max(compute_power, 1e-9))
        + l2_cache_usage * (1 - l2_bw / max(hbm_bw, 1e-9)) * cmr
        - (1 + l2_bw / max(hbm_bw, 1e-9)) / max(spec.k, 1)
    )

    base_m_best = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
    base_n_best = max(
        ADV_BASIC_BLOCK_16,
        min(
            ceil_align(spec.n, ADV_BASIC_BLOCK_16),
            floor_align(base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // max(base_m_best, 1), ADV_BASIC_BLOCK_16),
        ),
    )
    is_memory_bound = (1.0 / base_m_best + 1.0 / base_n_best) > cube_bound_edge
    inner_align_unit = ADV_BASIC_BLOCK_128 if is_memory_bound else ADV_BASIC_BLOCK_64
    fixp_bound_edge = (spec.m * spec.n * hbm_bw) / max((spec.m + spec.n) * l2_bw, 1e-9)
    base_m_align_unit = inner_align_unit // max(elem_size, 1) if spec.trans_a else ADV_BASIC_BLOCK_16
    base_n_align_unit = (
        ADV_BASIC_BLOCK_K_256_BYTE // max(elem_size, 1)
        if spec.k < fixp_bound_edge
        else (ADV_BASIC_BLOCK_16 if spec.trans_b else inner_align_unit // max(elem_size, 1))
    )
    base_m_align_unit = max(ADV_BASIC_BLOCK_16, base_m_align_unit)
    base_n_align_unit = max(ADV_BASIC_BLOCK_16, base_n_align_unit)
    max_base_m = advanced_max_base_with_limit(
        spec, elem_size, config, base_mn_buffer_limit, base_m_align_unit, False, is_memory_bound
    )
    max_base_n = advanced_max_base_with_limit(
        spec, elem_size, config, base_mn_buffer_limit, base_n_align_unit, True, is_memory_bound
    )

    best_base_m = max(ADV_BASIC_BLOCK_16, min(max_base_m, ADV_BASIC_BLOCK_256))
    best_base_n = max(
        ADV_BASIC_BLOCK_16,
        min(max_base_n, floor_align(base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // best_base_m, base_n_align_unit)),
    )
    best_cube_param = 1.0 / best_base_m + 1.0 / best_base_n
    cube_bound_edge *= ADV_CUBE_BOUND_RATIO
    best_balance = advanced_balance_rate_with_tail(spec, aic_num, best_base_m, best_base_n)

    for cur_base_m in range(max_base_m, 0, -base_m_align_unit):
        cur_base_m = floor_align(cur_base_m, base_m_align_unit)
        if cur_base_m < ADV_BASIC_BLOCK_16:
            continue
        cur_max_base_n = min(max_base_n, floor_align(base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // cur_base_m, base_n_align_unit))
        for cur_base_n in range(cur_max_base_n, 0, -base_n_align_unit):
            cur_base_n = floor_align(cur_base_n, base_n_align_unit)
            if cur_base_n < ADV_BASIC_BLOCK_16:
                continue
            cur_cube_param = 1.0 / cur_base_m + 1.0 / cur_base_n
            cur_balance = advanced_balance_rate_with_tail(spec, aic_num, cur_base_m, cur_base_n)
            if best_balance >= 0.9 and cur_cube_param > best_cube_param and cur_cube_param > cube_bound_edge:
                continue
            cube_bound_cond = cur_cube_param <= cube_bound_edge and cur_balance > best_balance
            current_score = cur_cube_param / max(cur_balance, 1e-9)
            best_score = best_cube_param / max(best_balance, 1e-9)
            balance_cond = current_score < best_score or (abs(current_score - best_score) < 1e-9 and cur_balance > best_balance)
            if cube_bound_cond or balance_cond:
                best_base_m = cur_base_m
                best_base_n = cur_base_n
                best_cube_param = cur_cube_param
                best_balance = cur_balance

    best_base_m = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), best_base_m)
    best_base_n = min(ceil_align(spec.n, ADV_BASIC_BLOCK_16), best_base_n)
    base_k = advanced_base_k(spec, elem_size, int(config["l0a_bytes"]), best_base_m, best_base_n)
    m_core = ceil_div(spec.m, best_base_m)
    n_core = ceil_div(spec.n, best_base_n)
    used_core = min(spec.batch * m_core * n_core, aic_num)
    return best_base_m, best_base_n, base_k, used_core, best_balance


def advanced_matmul_base(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    if ceil_div(spec.m, ADV_BASIC_BLOCK_256) * ceil_div(spec.n, ADV_BASIC_BLOCK_256) >= aic_num:
        base_m = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
        base_n = min(ceil_align(spec.n, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
    else:
        best: tuple[float, int, int] | None = None
        for base_m_candidate in source_base_values(spec.m):
            for base_n_candidate in source_base_values(spec.n):
                if base_m_candidate * base_n_candidate * ADV_DATA_SIZE_FP32 > int(config["l0c_bytes"]):
                    continue
                m_tiles = ceil_div(spec.m, base_m_candidate)
                n_tiles = ceil_div(spec.n, base_n_candidate)
                mn_tiles = m_tiles * n_tiles
                rounds = ceil_div(mn_tiles, aic_num)
                core_eff = mn_tiles / max(rounds * aic_num, 1)
                redundant = 1.0 / base_m_candidate + 1.0 / base_n_candidate
                tail_waste = (m_tiles * base_m_candidate * n_tiles * base_n_candidate) / max(spec.m * spec.n, 1)
                score = (1.0 / max(core_eff, 1e-9)) + 0.35 * redundant * ADV_BASIC_BLOCK_256 + 0.15 * tail_waste
                if best is None or score < best[0]:
                    best = (score, base_m_candidate, base_n_candidate)
        if best is None:
            base_m = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
            base_n = min(ceil_align(spec.n, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
        else:
            _, base_m, base_n = best
    base_k = advanced_base_k(spec, elem_size, int(config["l0a_bytes"]), base_m, base_n)
    used_core = min(aic_num, ceil_div(spec.m, base_m) * ceil_div(spec.n, base_n))
    return base_m, base_n, base_k, used_core


def apply_advanced_full_load(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    base_m: int,
    base_n: int,
    base_k: int,
    depth_a1: int,
    depth_b1: int,
    step_ka: int,
    step_kb: int,
) -> tuple[int, int, int, int, int, int, int, int, str]:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    l1 = int(config["l1_bytes"])
    l0a = int(config["l0a_bytes"])
    l0b = int(config["l0b_bytes"])
    l0c = int(config["l0c_bytes"])
    asw_window = asw_window_len(aic_num)
    l0c2out = advanced_l0c2out(
        spec, output_dtype, aic_num, int(config.get("aiv_num", aic_num * 2)), base_m, base_n, dtype
    )
    m_cnt = ceil_div(spec.m, base_m)
    n_cnt = ceil_div(spec.n, base_n)
    is_single_round = m_cnt * n_cnt <= aic_num

    full_load = "NONE_FULL_LOAD"
    single_core_m = base_m
    single_core_n = base_n
    step_m = 1
    step_n = 1

    m_aligned = ceil_align(spec.m, ADV_BASIC_BLOCK_16)
    n_aligned = ceil_align(spec.n, ADV_BASIC_BLOCK_16)
    k_aligned_a = ceil_align(spec.k, ADV_BASIC_BLOCK_16 if spec.trans_a else max(1, ADV_BLOCK_BYTE_SIZE // elem_size))
    k_aligned_b = ceil_align(spec.k, max(1, ADV_BLOCK_BYTE_SIZE // elem_size) if spec.trans_b else ADV_BASIC_BLOCK_16)
    max_step = max(1, asw_window - 1)

    al1_size = k_aligned_a * m_aligned * elem_size
    a_l1_full_mte2 = spec.m * aic_num + spec.n * m_cnt
    base_mte2_for_a = spec.m * n_cnt + spec.n * m_cnt
    al1_ok = (
        l0c2out == "ON_THE_FLY"
        and spec.n >= ADV_CACHELINE
        and not is_single_round
        and spec.m < max_step * base_m
        and al1_size <= l1 * 3 // 4
        and not (spec.m > ADV_BASIC_BLOCK_256 and base_mte2_for_a < 1.2 * a_l1_full_mte2)
    )

    bl1_size = k_aligned_b * n_aligned * elem_size
    b_l1_full_mte2 = spec.n * aic_num + spec.m * n_cnt
    base_mte2_for_b = spec.m * n_cnt + spec.n * m_cnt
    bl1_ok = (
        spec.m >= ADV_CACHELINE
        and not is_single_round
        and spec.n < max_step * base_n
        and bl1_size <= l1 * 3 // 4
        and not (spec.n > ADV_BASIC_BLOCK_256 and base_mte2_for_b < 1.2 * b_l1_full_mte2)
    )

    if al1_ok:
        if m_aligned * base_k * elem_size * ADV_DB_SIZE <= l0a:
            base_m = m_aligned
        else:
            base_m = min(m_aligned, base_m)
        step_m = ceil_div(spec.m, base_m)
        step_ka = ceil_div(spec.k, base_k)
        step_kb = advanced_step_small_k(spec, dtype, elem_size, base_k, step_ka, step_kb, False)
        if ceil_div(spec.n, base_n) < aic_num:
            base_n = max(ADV_BASIC_BLOCK_16, ceil_align(ceil_div(spec.n, aic_num), ADV_BASIC_BLOCK_16))
        depth_b1 = ADV_DB_SIZE * step_kb
        depth_a1 = step_m * step_ka
        a_l1_size = ceil_align(spec.k, ADV_BASIC_BLOCK_16) * m_aligned * elem_size
        b_l1_load_size = base_k * depth_b1 * base_n * elem_size
        while base_n > ADV_BASIC_BLOCK_16 and b_l1_load_size > l1 - a_l1_size:
            if step_kb == min(step_ka, 2):
                base_n = ceil_align(max(ADV_BASIC_BLOCK_16, base_n >> 1), ADV_BASIC_BLOCK_16)
            step_kb = min(step_ka, 2)
            depth_b1 = ADV_DB_SIZE * step_kb
            b_l1_load_size = depth_b1 * base_n * base_k * elem_size
        single_core_m = spec.m
        single_core_n = base_n
        db_l0c = ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 * ADV_DB_SIZE <= l0c else 1
        if (
            (spec.trans_b and (base_n * elem_size) % ADV_BASIC_BLOCK_K_256_BYTE == 0)
            or (base_n * elem_size) % (ADV_BASIC_BLOCK_K_256_BYTE * 2) == 0
        ) and db_l0c <= 1:
            base_n = ceil_align(max(ADV_BASIC_BLOCK_16, base_n >> 1), ADV_BASIC_BLOCK_16)
            single_core_n = base_n
        full_load = "A_FULL_LOAD"
    elif bl1_ok:
        if n_aligned * base_k * elem_size * ADV_DB_SIZE <= l0b:
            base_n = n_aligned
        else:
            base_n = min(n_aligned, base_n)
        step_n = ceil_div(spec.n, base_n)
        step_kb = ceil_div(spec.k, base_k)
        step_ka = advanced_step_small_k(spec, dtype, elem_size, base_k, step_ka, step_kb, True)
        if ceil_div(spec.m, base_m) < aic_num:
            base_m = max(ADV_BASIC_BLOCK_16, ceil_align(ceil_div(spec.m, aic_num), ADV_BASIC_BLOCK_16))
        depth_a1 = ADV_DB_SIZE * step_ka
        depth_b1 = step_n * step_kb
        b_l1_size = ceil_align(spec.k, ADV_BASIC_BLOCK_16) * n_aligned * elem_size
        a_l1_load_size = base_k * depth_a1 * base_m * elem_size
        while base_m > ADV_BASIC_BLOCK_16 and a_l1_load_size > l1 - b_l1_size:
            if step_ka == min(step_kb, 2):
                base_m = ceil_align(max(ADV_BASIC_BLOCK_16, base_m >> 1), ADV_BASIC_BLOCK_16)
            step_ka = min(step_kb, 2)
            depth_a1 = ADV_DB_SIZE * step_ka
            a_l1_load_size = depth_a1 * base_m * base_k * elem_size
        single_core_n = spec.n
        single_core_m = base_m
        if (not spec.trans_a or (base_m * elem_size) % (ADV_BASIC_BLOCK_K_256_BYTE * 2) == 0) and (
            base_m * base_n * ADV_DATA_SIZE_FP32 * ADV_DB_SIZE > l0c
        ):
            base_m = ceil_align(max(ADV_BASIC_BLOCK_16, base_m >> 1), ADV_BASIC_BLOCK_16)
            single_core_m = base_m
        full_load = "B_FULL_LOAD"

    return base_m, base_n, single_core_m, single_core_n, depth_a1, depth_b1, step_m, step_n, full_load


def make_tile_estimate_from_source_tiling(
    *,
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    base_m: int,
    base_n: int,
    base_k: int,
    single_core_m: int,
    single_core_n: int,
    single_core_k: int,
    used_core_num: int,
    core_work_tiles: int,
    source: str,
    tiling_strategy: str,
    full_load: str,
    l0c2out: str,
    asw_window: int,
    depth_a1: int,
    depth_b1: int,
    step_m: int,
    step_n: int,
    step_ka: int,
    step_kb: int,
    l1_buffer_num: int,
    ub_db: int,
    tiling_split_core: int,
    tiling_full_load: int,
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    peak_tflops = peak_for_dtype(config, dtype)
    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    a_storage, b_storage, c_storage = storage_element_counts(spec)
    gm_bytes_min = a_storage * elem_size + b_storage * elem_size + c_storage * out_size

    tile_m = max(1, ceil_div(spec.m, max(single_core_m, 1)))
    tile_n = max(1, ceil_div(spec.n, max(single_core_n, 1)))
    tile_k = max(1, ceil_div(spec.k, max(single_core_k, 1)))
    mn_tile_count = tile_m * tile_n * spec.batch
    tile_count = mn_tile_count * tile_k
    rounds = max(1, ceil_div(max(core_work_tiles, mn_tile_count), aic_num))
    core_eff = min(1.0, max(core_work_tiles, mn_tile_count) / max(rounds * aic_num, 1))
    aligned_flops = (
        2
        * tile_m
        * single_core_m
        * tile_n
        * single_core_n
        * tile_k
        * single_core_k
        * spec.batch
    )
    tail_eff = true_flops / aligned_flops if aligned_flops else 0.0
    gm_bytes_tiled_raw = tile_n * a_storage * elem_size + tile_m * b_storage * elem_size + c_storage * out_size
    if "stream_k" in tiling_strategy and tile_k > 1:
        stream_k_factor = float(
            config.get("kernel_model", {}).get("advanced_tiling", {}).get("stream_k_reduction_traffic_factor", 0.25)
        )
        gm_bytes_tiled_raw += int((tile_k - 1) * c_storage * ADV_DATA_SIZE_FP32 * stream_k_factor)
    gm_bytes_tiled = l2_aware_gm_bytes(gm_bytes_min, gm_bytes_tiled_raw, config)
    hbm_us = gm_bytes_tiled / (float(config["hbm_bandwidth_tbps"]) * 1_000_000.0)
    compute_us = None
    if peak_tflops is not None:
        compute_us = aligned_flops / (peak_tflops * 1_000_000.0 * max(core_eff, 1e-9))
    lower_bound_us = max(value for value in (compute_us, hbm_us) if value is not None)
    return TileEstimate(
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        db_l0c=ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 * ADV_DB_SIZE <= int(config["l0c_bytes"]) else 1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        mn_tile_count=mn_tile_count,
        tile_count=tile_count,
        used_core_num=min(aic_num, max(1, used_core_num)),
        core_efficiency=core_eff,
        tail_efficiency=tail_eff,
        aligned_flops=aligned_flops,
        gm_bytes_min=gm_bytes_min,
        gm_bytes_tiled_raw=gm_bytes_tiled_raw,
        gm_bytes_tiled=gm_bytes_tiled,
        compute_us=compute_us,
        hbm_us=hbm_us,
        lower_bound_us=lower_bound_us,
        source=source,
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        step_m=step_m,
        step_n=step_n,
        step_ka=step_ka,
        step_kb=step_kb,
        tiling_strategy=tiling_strategy,
        full_load=full_load,
        l0c2out=l0c2out,
        asw_window_len=asw_window,
        l1_buffer_num=l1_buffer_num,
        ub_db=ub_db,
        tiling_split_core=tiling_split_core,
        tiling_full_load=tiling_full_load,
        tiling_fix_opti=0,
        tiling_special_opti=0,
    )


def estimate_tile_from_advanced_tiling(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    kernel_type: str,
) -> TileEstimate:
    stream_k = advanced_stream_k_kind(spec, dtype, config, kernel_type)
    if stream_k is not None:
        return advanced_stream_k_tile(spec, dtype, output_dtype, config, kernel_type, stream_k)

    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    if is_batch_matmul_kernel_type(kernel_type):
        base_m, base_n, base_k, used_core, _ = advanced_batch_rebalance_base(spec, dtype, config)
        tiling_strategy = "batch_asw_basic_rebalance"
    else:
        base_m, base_n, base_k, used_core = advanced_matmul_base(spec, dtype, config)
        tiling_strategy = "basic_aswt"

    single_core_k = spec.k
    depth_a1, depth_b1, step_m, step_n, step_ka, step_kb = advanced_cal_l1_tiling(
        spec, elem_size, int(config["l1_bytes"]), base_m, base_n, base_k, single_core_k
    )
    full_load = "NONE_FULL_LOAD"
    if not is_batch_matmul_kernel_type(kernel_type):
        base_m, base_n, single_core_m, single_core_n, depth_a1, depth_b1, step_m, step_n, full_load = apply_advanced_full_load(
            spec, dtype, output_dtype, config, base_m, base_n, base_k, depth_a1, depth_b1, step_ka, step_kb
        )
        if full_load == "A_FULL_LOAD":
            tiling_strategy = "basic_aswt_al1_full_load"
            step_ka = max(1, depth_a1 // max(step_m, 1))
            step_kb = max(1, depth_b1 // ADV_DB_SIZE)
        elif full_load == "B_FULL_LOAD":
            tiling_strategy = "basic_aswt_bl1_full_load"
            step_ka = max(1, depth_a1 // ADV_DB_SIZE)
            step_kb = max(1, depth_b1 // max(step_n, 1))
        else:
            single_core_m = base_m
            single_core_n = base_n
    else:
        single_core_m = base_m
        single_core_n = base_n

    l0c2out = advanced_l0c2out(
        spec, output_dtype, aic_num, int(config.get("aiv_num", aic_num * 2)), single_core_m, single_core_n, dtype
    )
    core_work_tiles = max(1, spec.batch * ceil_div(spec.m, max(single_core_m, 1)) * ceil_div(spec.n, max(single_core_n, 1)))
    tiling_full_load = {"NONE_FULL_LOAD": 0, "A_FULL_LOAD": 1, "B_FULL_LOAD": 2}.get(full_load, 0)
    l1_tensor_size = base_k * max(step_ka, step_kb) * (base_m + base_n) * elem_size
    l1_buffer_num = ADV_BASIC_L1_BUFFER_NUM if l1_tensor_size * ADV_BASIC_L1_BUFFER_NUM <= int(config["l1_bytes"]) else ADV_DB_SIZE
    return make_tile_estimate_from_source_tiling(
        spec=spec,
        dtype=dtype,
        output_dtype=output_dtype,
        config=config,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        single_core_m=single_core_m,
        single_core_n=single_core_n,
        single_core_k=single_core_k,
        used_core_num=used_core,
        core_work_tiles=core_work_tiles,
        source="advanced_tiling_heuristic",
        tiling_strategy=tiling_strategy,
        full_load=full_load,
        l0c2out=l0c2out,
        asw_window=asw_window_len(aic_num),
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        step_m=step_m,
        step_n=step_n,
        step_ka=step_ka,
        step_kb=step_kb,
        l1_buffer_num=l1_buffer_num,
        ub_db=ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 <= int(config.get("ub_bytes", 0)) else 1,
        tiling_split_core=0,
        tiling_full_load=tiling_full_load,
    )


def estimate_tile(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    l0a = int(config["l0a_bytes"])
    l0b = int(config["l0b_bytes"])
    l0c = int(config["l0c_bytes"])
    hbm_bandwidth_tbps = float(config["hbm_bandwidth_tbps"])
    peak_tflops = peak_for_dtype(config, dtype)

    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    logical_a_elements = spec.batch * spec.m * spec.k
    logical_b_elements = spec.batch * spec.k * spec.n
    logical_c_elements = spec.batch * spec.m * spec.n
    a_storage_elements = spec.a_storage_elements or logical_a_elements
    b_storage_elements = spec.b_storage_elements or logical_b_elements
    output_storage_elements = spec.output_storage_elements or logical_c_elements
    gm_bytes_min = (
        a_storage_elements * elem_size
        + b_storage_elements * elem_size
        + output_storage_elements * out_size
    )

    best: TileEstimate | None = None
    for base_m in candidate_base_values(spec.m):
        for base_n in candidate_base_values(spec.n):
            # L0C accumulates FP32 partial sums.
            if base_m * base_n * 4 > l0c:
                continue
            db_l0c = 2 if base_m * base_n * 4 * 2 <= l0c else 1

            max_k_a = floor_align(l0a // max(1, 2 * elem_size * base_m), 16)
            max_k_b = floor_align(l0b // max(1, 2 * elem_size * base_n), 16)
            max_base_k = min(max_k_a, max_k_b)
            if max_base_k < 16:
                continue

            base_k = min(ceil_align(spec.k, 16), max_base_k)
            base_k = max(16, floor_align(base_k, 16))

            tile_m = ceil_div(spec.m, base_m)
            tile_n = ceil_div(spec.n, base_n)
            tile_k = ceil_div(spec.k, base_k)
            mn_tile_count = tile_m * tile_n * spec.batch
            tile_count = mn_tile_count * tile_k
            used_core_num = min(aic_num, max(1, mn_tile_count))
            rounds = max(1, ceil_div(mn_tile_count, aic_num))
            core_eff = mn_tile_count / (rounds * aic_num)

            aligned_m = tile_m * base_m
            aligned_n = tile_n * base_n
            aligned_k = tile_k * base_k
            aligned_flops = 2 * aligned_m * aligned_n * aligned_k * spec.batch
            tail_eff = true_flops / aligned_flops if aligned_flops else 0.0

            # Raw repeated traffic if every tile reread hits GM. The effective
            # HBM estimate below is L2-aware, matching the kernel's L2 cache
            # decision path rather than pessimistically charging every repeat.
            gm_bytes_tiled_raw = (
                tile_n * a_storage_elements * elem_size
                + tile_m * b_storage_elements * elem_size
                + output_storage_elements * out_size
            )
            l2_bytes = int(config.get("l2_bytes", 0))
            if l2_bytes > 0 and gm_bytes_min <= l2_bytes:
                gm_bytes_tiled = gm_bytes_min
            elif l2_bytes > 0:
                redundant = max(0, gm_bytes_tiled_raw - gm_bytes_min)
                l2_pressure = max(0.0, 1.0 - l2_bytes / max(gm_bytes_min, 1))
                gm_bytes_tiled = int(gm_bytes_min + redundant * l2_pressure)
            else:
                gm_bytes_tiled = gm_bytes_tiled_raw
            hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1_000_000.0)
            if peak_tflops is None:
                compute_us = None
                lower_bound_us = hbm_us
            else:
                compute_us = aligned_flops / (peak_tflops * 1_000_000.0 * max(core_eff, 1e-9))
                lower_bound_us = max(compute_us, hbm_us)

            estimate = TileEstimate(
                base_m=base_m,
                base_n=base_n,
                base_k=base_k,
                db_l0c=db_l0c,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                mn_tile_count=mn_tile_count,
                tile_count=tile_count,
                used_core_num=used_core_num,
                core_efficiency=core_eff,
                tail_efficiency=tail_eff,
                aligned_flops=aligned_flops,
                gm_bytes_min=gm_bytes_min,
                gm_bytes_tiled_raw=gm_bytes_tiled_raw,
                gm_bytes_tiled=gm_bytes_tiled,
                compute_us=compute_us,
                hbm_us=hbm_us,
                lower_bound_us=lower_bound_us,
            )
            if best is None or estimate.lower_bound_us < best.lower_bound_us:
                best = estimate

    if best is None:
        # Fallback should be rare; it keeps the evaluator usable for odd shapes.
        aligned_m = ceil_align(spec.m, 16)
        aligned_n = ceil_align(spec.n, 16)
        aligned_k = ceil_align(spec.k, 16)
        aligned_flops = 2 * aligned_m * aligned_n * aligned_k * spec.batch
        hbm_us = gm_bytes_min / (hbm_bandwidth_tbps * 1_000_000.0)
        compute_us = None
        if peak_tflops is not None:
            compute_us = aligned_flops / (peak_tflops * 1_000_000.0)
        return TileEstimate(
            base_m=16,
            base_n=16,
            base_k=16,
            db_l0c=1,
            tile_m=ceil_div(spec.m, 16),
            tile_n=ceil_div(spec.n, 16),
            tile_k=ceil_div(spec.k, 16),
            mn_tile_count=ceil_div(spec.m, 16) * ceil_div(spec.n, 16) * spec.batch,
            tile_count=ceil_div(spec.m, 16) * ceil_div(spec.n, 16) * ceil_div(spec.k, 16) * spec.batch,
            used_core_num=min(aic_num, ceil_div(spec.m, 16) * ceil_div(spec.n, 16) * spec.batch),
            core_efficiency=1.0,
            tail_efficiency=(2 * spec.m * spec.n * spec.k * spec.batch) / aligned_flops,
            aligned_flops=aligned_flops,
            gm_bytes_min=gm_bytes_min,
            gm_bytes_tiled_raw=gm_bytes_min,
            gm_bytes_tiled=gm_bytes_min,
            compute_us=compute_us,
            hbm_us=hbm_us,
            lower_bound_us=max(value for value in (compute_us, hbm_us) if value is not None),
        )
    return best


def estimate_tile_from_runtime_kb(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    entry: RuntimeKbEntry,
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    hbm_bandwidth_tbps = float(config["hbm_bandwidth_tbps"])
    peak_tflops = peak_for_dtype(config, dtype)
    info = entry.info
    knowledge = entry.knowledge

    aligned_m = int(info["m"])
    aligned_n = int(info["n"])
    aligned_k = int(info["k"])
    base_m = max(1, int(knowledge.get("baseM", aligned_m or 1)))
    base_n = max(1, int(knowledge.get("baseN", aligned_n or 1)))
    base_k = max(1, int(knowledge.get("baseK", aligned_k or 1)))
    single_core_m = max(1, int(knowledge.get("singleCoreM", base_m)))
    single_core_n = max(1, int(knowledge.get("singleCoreN", base_n)))
    single_core_k = max(1, int(knowledge.get("singleCoreK", aligned_k or base_k)))

    tile_m = max(1, ceil_div(aligned_m, single_core_m))
    tile_n = max(1, ceil_div(aligned_n, single_core_n))
    tile_k = max(1, ceil_div(aligned_k, single_core_k))
    mn_tile_count = tile_m * tile_n * spec.batch
    tile_count = mn_tile_count * tile_k
    used_core_num = max(1, min(aic_num, int(knowledge.get("usedCoreNum", min(aic_num, mn_tile_count)))))
    core_eff = min(1.0, used_core_num / max(aic_num, 1))

    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    aligned_flops = 2 * tile_m * single_core_m * tile_n * single_core_n * tile_k * single_core_k * spec.batch
    tail_eff = true_flops / aligned_flops if aligned_flops else 0.0

    logical_a_elements = spec.batch * spec.m * spec.k
    logical_b_elements = spec.batch * spec.k * spec.n
    logical_c_elements = spec.batch * spec.m * spec.n
    a_storage_elements = spec.a_storage_elements or logical_a_elements
    b_storage_elements = spec.b_storage_elements or logical_b_elements
    output_storage_elements = spec.output_storage_elements or logical_c_elements
    gm_bytes_min = (
        a_storage_elements * elem_size
        + b_storage_elements * elem_size
        + output_storage_elements * out_size
    )
    gm_bytes_tiled_raw = (
        tile_n * a_storage_elements * elem_size
        + tile_m * b_storage_elements * elem_size
        + output_storage_elements * out_size
    )
    l2_bytes = int(config.get("l2_bytes", 0))
    if l2_bytes > 0 and gm_bytes_min <= l2_bytes:
        gm_bytes_tiled = gm_bytes_min
    elif l2_bytes > 0:
        redundant = max(0, gm_bytes_tiled_raw - gm_bytes_min)
        l2_pressure = max(0.0, 1.0 - l2_bytes / max(gm_bytes_min, 1))
        gm_bytes_tiled = int(gm_bytes_min + redundant * l2_pressure)
    else:
        gm_bytes_tiled = gm_bytes_tiled_raw

    hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1_000_000.0)
    compute_us = None
    if peak_tflops is not None:
        compute_us = aligned_flops / (peak_tflops * 1_000_000.0 * max(core_eff, 1e-9))
    lower_bound_us = max(value for value in (compute_us, hbm_us) if value is not None)
    tiling_enable = int(knowledge.get("tilingEnable", -1))
    decoded_tiling = decode_tiling_enable(tiling_enable)
    return TileEstimate(
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        db_l0c=int(knowledge.get("dbL0C", 1)),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        mn_tile_count=mn_tile_count,
        tile_count=tile_count,
        used_core_num=used_core_num,
        core_efficiency=core_eff,
        tail_efficiency=tail_eff,
        aligned_flops=aligned_flops,
        gm_bytes_min=gm_bytes_min,
        gm_bytes_tiled_raw=gm_bytes_tiled_raw,
        gm_bytes_tiled=gm_bytes_tiled,
        compute_us=compute_us,
        hbm_us=hbm_us,
        lower_bound_us=lower_bound_us,
        source="runtime_kb_exact",
        runtime_kb_id=entry.entry_id,
        runtime_kb_file=entry.source_file,
        tiling_enable=tiling_enable,
        depth_a1=int(knowledge.get("depthA1", 0)),
        depth_b1=int(knowledge.get("depthB1", 0)),
        step_m=int(knowledge.get("stepM", 0)),
        step_n=int(knowledge.get("stepN", 0)),
        step_ka=int(knowledge.get("stepKa", 0)),
        step_kb=int(knowledge.get("stepKb", 0)),
        l2_m_tile=int(knowledge.get("l2MTileCnt", 0)),
        l2_n_tile=int(knowledge.get("l2NTileCnt", 0)),
        tiling_strategy="runtime_kb",
        full_load=tiling_full_load_name(decoded_tiling["full_load"]),
        tiling_split_core=decoded_tiling["split_core"],
        tiling_full_load=decoded_tiling["full_load"],
        tiling_fix_opti=decoded_tiling["fix_opti"],
        tiling_special_opti=decoded_tiling["special_opti"],
    )


def advanced_tiling_notes(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
    kernel_type: str,
    tile: TileEstimate | None = None,
) -> str:
    kernel_model = config.get("kernel_model", {})
    advanced_cfg = kernel_model.get("advanced_tiling", {})
    if not advanced_cfg.get("enabled", False):
        return "disabled"
    if not is_ops_nn_v3_kernel_type(kernel_type):
        return "not_ops_nn_v3"

    notes: list[str] = ["advanced_soc"]
    if tile is not None:
        if tile.tiling_strategy:
            notes.append(tile.tiling_strategy)
        if tile.full_load and tile.full_load != "NONE_FULL_LOAD":
            notes.append(tile.full_load.lower())
        if tile.l0c2out and tile.l0c2out != "ON_THE_FLY":
            notes.append(tile.l0c2out.lower())
    aic_num = int(config.get("aic_num", 1))
    m_cnt_128 = ceil_div(spec.m, 128)
    n_cnt_128 = ceil_div(spec.n, 128)
    mn_cnt = m_cnt_128 * n_cnt_128 * spec.batch
    if advanced_stream_k_kind(spec, dtype, config, kernel_type) is not None:
        notes.append("stream_k_capable_by_shape")
    elif spec.k >= aic_num * 384 and mn_cnt < max(1, aic_num // 2) and not (not spec.trans_a and spec.trans_b):
        notes.append("multi_core_splitk_candidate")
    if aic_num == 20:
        notes.append("aic20_l2_conflict_factor")
    elif aic_num == 24:
        notes.append("aic24_factor_table")
    if spec.a_format == "FRACTAL_NZ" or spec.b_format == "FRACTAL_NZ":
        notes.append("nz_layout")
    return "|".join(notes)


def select_tile_estimate(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    runtime_kb: dict[tuple[Any, ...], list[RuntimeKbEntry]],
    input_dtypes: list[str],
    kernel_type: str,
) -> TileEstimate:
    use_ops_nn_v3_model = is_ops_nn_v3_kernel_type(kernel_type)
    key = runtime_kb_key_from_row(spec, input_dtypes, output_dtype) if use_ops_nn_v3_model else None
    if key is not None and not is_batch_matmul_kernel_type(kernel_type) and key in runtime_kb:
        entries = runtime_kb[key]
        estimates = [estimate_tile_from_runtime_kb(spec, dtype, output_dtype, config, entry) for entry in entries]
        return min(estimates, key=lambda estimate: estimate.lower_bound_us)

    if use_ops_nn_v3_model and config.get("kernel_model", {}).get("advanced_tiling", {}).get("enabled", False):
        return estimate_tile_from_advanced_tiling(spec, dtype, output_dtype, config, kernel_type)

    estimate = estimate_tile(spec, dtype, output_dtype, config)
    return estimate


def ideal_kernel_bounds(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
) -> tuple[float | None, float, float, int]:
    peak_tflops = peak_for_dtype(config, dtype)
    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    compute_us = None
    if peak_tflops is not None:
        compute_us = true_flops / (peak_tflops * 1_000_000.0)
    logical_a_elements = spec.batch * spec.m * spec.k
    logical_b_elements = spec.batch * spec.k * spec.n
    logical_c_elements = spec.batch * spec.m * spec.n
    a_storage_elements = spec.a_storage_elements or logical_a_elements
    b_storage_elements = spec.b_storage_elements or logical_b_elements
    output_storage_elements = spec.output_storage_elements or logical_c_elements
    gm_bytes_min = (
        a_storage_elements * dtype_size(dtype)
        + b_storage_elements * dtype_size(dtype)
        + output_storage_elements * dtype_size(output_dtype)
    )
    hbm_us = gm_bytes_min / (float(config["hbm_bandwidth_tbps"]) * 1_000_000.0)
    lower = max(value for value in (compute_us, hbm_us) if value is not None)
    return compute_us, hbm_us, lower, gm_bytes_min


def dominant_bottleneck(launch_us: float, compute_us: float | None, hbm_us: float, format_us: float) -> str:
    components = {"launch": launch_us, "hbm": hbm_us, "format": format_us}
    if compute_us is not None:
        components["compute"] = compute_us
    return max(components.items(), key=lambda item: item[1])[0]


def is_fp32_dtype(dtype: str) -> bool:
    return dtype in {"FLOAT", "FLOAT32", "DT_FLOAT"}


def is_256b_aligned(inner_size: int, dtype_size_value: int) -> bool:
    return (inner_size * dtype_size_value) % 256 == 0


def is_nd2nz_on_the_way_supported(tensor_format: str, inner_size: int, dtype_size_value: int) -> bool:
    # From MatMulV3 SUPPORT_ND2NZ_GM2L0. These are byte lengths, not element counts.
    supported_bytes = {32, 64, 96, 128, 160, 192, 224, 256, 384}
    return tensor_format == "ND" and inner_size * dtype_size_value in supported_bytes


def need_nd2nz_for_operand(
    tensor_format: str,
    inner_size: int,
    outer_size: int,
    dtype: str,
    dtype_size_value: int,
) -> bool:
    if tensor_format != "ND" or dtype_size_value <= 0:
        return False

    support_on_the_way = is_nd2nz_on_the_way_supported(tensor_format, inner_size, dtype_size_value)
    inner_aligned = is_256b_aligned(inner_size, dtype_size_value)
    normal_nd2nz = (
        (not inner_aligned or inner_size > 65535)
        and not support_on_the_way
        and not (is_fp32_dtype(dtype) and inner_size < 65535)
    )

    will_fit_vnchw = (
        outer_size > 8192
        and inner_size > 1
        and (
            inner_size * dtype_size_value <= 192
            or (inner_size * dtype_size_value <= 384 and inner_size % 2 == 0)
            or (inner_size * dtype_size_value <= 512 and inner_size % 4 == 0)
        )
    )
    inner_equals_c0 = inner_size == 32 // dtype_size_value
    vnchw_nd2nz = will_fit_vnchw and not inner_aligned and not support_on_the_way and not inner_equals_c0
    return normal_nd2nz or vnchw_nd2nz


def infer_nd2nz_operands(spec: MatmulSpec, dtype: str) -> tuple[bool, bool]:
    elem_size = dtype_size(dtype)
    inner_a = spec.m if spec.trans_a else spec.k
    outer_a = spec.k if spec.trans_a else spec.m
    inner_b = spec.k if spec.trans_b else spec.n
    outer_b = spec.n if spec.trans_b else spec.k
    nd2nz_a = need_nd2nz_for_operand(spec.a_format, inner_a, outer_a, dtype, elem_size)
    nd2nz_b = need_nd2nz_for_operand(spec.b_format, inner_b, outer_b, dtype, elem_size)
    return nd2nz_a, nd2nz_b


def split_semicolon_values(value: str | None) -> list[str]:
    if not value or value == "N/A":
        return []
    return [part.strip() for part in value.strip().replace('"', "").split(";") if part.strip()]


def quant_storage_bytes(elements: int, dtype: str) -> int:
    bitwidth = dtype_bitwidth(dtype)
    if bitwidth < 8:
        return ceil_div(elements * bitwidth, 8)
    return elements * dtype_size(dtype)


def infer_quant_spec(
    row: dict[str, str],
    spec: MatmulSpec,
    input_shapes: list[list[int]],
) -> QuantSpec:
    kernel_type = row.get("Type", "")
    input_dtypes = split_semicolon_values(row.get("Input Data Types"))
    output_dtype = output_dtype_from_row(row)
    if not is_quant_kernel_type(kernel_type) and not any(is_quantized_data_dtype(dtype) for dtype in input_dtypes[:2]):
        return QuantSpec(is_quant=False)

    a_dtype = input_dtypes[0] if input_dtypes else "UNKNOWN"
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else "UNKNOWN"
    aux_shapes = input_shapes[2:] if len(input_shapes) > 2 else []
    aux_dtypes = input_dtypes[2:] if len(input_dtypes) > 2 else []
    aux_elements = sum(num_elements(shape) for shape in aux_shapes)
    aux_bytes = 0
    for shape, dtype in zip(aux_shapes, aux_dtypes):
        aux_bytes += num_elements(shape) * dtype_size(dtype)

    quant_data_inputs = sum(1 for dtype in (a_dtype, b_dtype) if is_quantized_data_dtype(dtype))
    has_scale = any(is_scale_dtype(dtype) for dtype in aux_dtypes)
    input_type_names = [dtype.upper().replace("DT_", "") for dtype in input_dtypes]
    min_data_bits = min(dtype_bitwidth(a_dtype), dtype_bitwidth(b_dtype))
    if any(dtype in {"MXFP8", "HIFLOAT8", "HIF8"} for dtype in input_type_names):
        mode = "mxfp8"
    elif any(dtype in {"FLOAT8", "FLOAT8_E4M3FN", "FLOAT8_E5M2", "FP8"} for dtype in input_type_names):
        mode = "fp8"
    elif any(dtype in {"FLOAT4", "FLOAT4_E2M1"} for dtype in input_type_names):
        mode = "float4"
    else:
        mode = f"int{min_data_bits}"

    kernel_type_lower = kernel_type.lower()
    if "weightquant" in kernel_type_lower and quant_data_inputs >= 1 and has_scale:
        compute_path = "weight_only_quant_with_dequant"
    elif "weightquant" in kernel_type_lower and quant_data_inputs >= 1:
        compute_path = "weight_only_quant"
    elif quant_data_inputs == 2 and has_scale and output_dtype in {"FLOAT16", "DT_FLOAT16", "DT_BF16", "BFLOAT16", "FLOAT", "DT_FLOAT"}:
        compute_path = "full_quant_with_dequant"
    elif quant_data_inputs == 2:
        compute_path = "full_quant"
    else:
        compute_path = "fake_quant_or_mixed"

    granularity = "unknown"
    notes: list[str] = []
    is_qbm_v3 = "quantbatchmatmulv3" in kernel_type_lower or "quantmatmulv3" in kernel_type_lower
    if is_qbm_v3:
        scale_shape = aux_shapes[0] if aux_shapes else []
        pertoken_shape = aux_shapes[3] if len(aux_shapes) >= 4 else []
        scale_is_channel = len(scale_shape) == 1 and scale_shape[0] in {1, spec.n}
        pertoken_is_m = len(pertoken_shape) == 1 and pertoken_shape[0] in {spec.m, spec.batch * spec.m}
        if pertoken_is_m and scale_is_channel:
            granularity = "per_token_per_channel"
        elif pertoken_is_m:
            granularity = "per_token_m"
        elif scale_is_channel:
            granularity = "per_channel_n" if scale_shape[0] == spec.n else "per_tensor"
        if granularity != "unknown":
            notes.append("qbm_v3_scale_offset_order")

    for shape in aux_shapes:
        if is_qbm_v3 and granularity != "unknown":
            break
        if len(shape) == 1:
            dim = shape[0]
            if dim == spec.n and dim != spec.m:
                granularity = "per_channel_n"
            elif dim == spec.m and dim != spec.n:
                granularity = "per_token_m"
            elif dim == spec.n and dim == spec.m:
                granularity = "per_channel_n_or_per_token_m"
                notes.append("scale_shape_equals_m_and_n")
            elif dim == 1:
                granularity = "per_tensor"
            elif spec.n % dim == 0 or spec.m % dim == 0:
                granularity = "per_group_or_block"
            else:
                granularity = "vector_scale_unknown_axis"
        elif len(shape) >= 2:
            if shape[-1] == spec.n and shape[-2] in {1, spec.m}:
                granularity = "per_token_per_channel"
            elif shape[-1] != spec.n and spec.n % shape[-1] == 0:
                granularity = "per_group"
            else:
                granularity = "tensor_scale"

    if not aux_shapes:
        granularity = "none"
    if has_scale and granularity == "unknown":
        notes.append("scale_dtype_present_but_granularity_unknown")
    if compute_path.startswith("full_quant") and has_scale:
        notes.append("int_accumulate_then_dequant")
    if compute_path.startswith("weight_only_quant"):
        notes.append("weight_dequant_on_matmul_path")

    deduped_notes = list(dict.fromkeys(notes))
    return QuantSpec(
        is_quant=True,
        mode=mode,
        granularity=granularity,
        compute_path=compute_path,
        aux_elements=aux_elements,
        aux_bytes=aux_bytes,
        notes="|".join(deduped_notes),
    )


def estimate_quant_cost(
    spec: MatmulSpec,
    tile: TileEstimate,
    quant_spec: QuantSpec,
    input_shapes: list[list[int]],
    input_dtypes: list[str],
    output_dtype: str,
    config: dict[str, Any],
) -> tuple[float | None, float, float, int, int]:
    quant_cfg = config.get("quant_matmul", {})
    output_elements = spec.output_storage_elements or spec.batch * spec.m * spec.n
    a_elements = spec.a_storage_elements or spec.batch * spec.m * spec.k
    b_elements = spec.b_storage_elements or spec.batch * spec.k * spec.n
    a_dtype = input_dtypes[0] if input_dtypes else "UNKNOWN"
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else "UNKNOWN"
    out_size = dtype_size(output_dtype)

    aux_bytes = quant_spec.aux_bytes
    quant_gm_bytes_min = (
        quant_storage_bytes(a_elements, a_dtype)
        + quant_storage_bytes(b_elements, b_dtype)
        + aux_bytes
        + output_elements * out_size
    )

    scale_replay = float(quant_cfg.get("scale_replay_factor", 1.0))
    quant_gm_bytes_tiled = int(
        quant_storage_bytes(a_elements, a_dtype) * tile.tile_n
        + quant_storage_bytes(b_elements, b_dtype) * tile.tile_m
        + aux_bytes * scale_replay
        + output_elements * out_size
    )
    l2_bytes = int(config.get("l2_bytes", 0))
    if l2_bytes > 0 and quant_gm_bytes_min <= l2_bytes:
        quant_gm_bytes_tiled = quant_gm_bytes_min
    elif l2_bytes > 0:
        redundant = max(0, quant_gm_bytes_tiled - quant_gm_bytes_min)
        l2_pressure = max(0.0, 1.0 - l2_bytes / max(quant_gm_bytes_min, 1))
        quant_gm_bytes_tiled = int(quant_gm_bytes_min + redundant * l2_pressure)

    hbm_us = quant_gm_bytes_tiled / (float(config["hbm_bandwidth_tbps"]) * 1_000_000.0)

    peak_tops = peak_for_dtype(config, a_dtype)
    if peak_tops is None:
        peak_tops = quant_cfg.get("peak_tops", {}).get(a_dtype)
    if peak_tops is None:
        peak_tops = quant_cfg.get("peak_tops", {}).get(a_dtype.replace("DT_", ""))

    compute_us: float | None = None
    if peak_tops is not None:
        efficiency = float(quant_cfg.get("pipeline_efficiency", {}).get(a_dtype, quant_cfg.get("pipeline_efficiency", {}).get("default", 1.0)))
        efficiency = max(efficiency, 1e-9)
        op_factor = float(quant_cfg.get("operation_factor", {}).get(quant_spec.compute_path, 1.0))
        compute_us = tile.aligned_flops * op_factor / (float(peak_tops) * 1_000_000.0 * tile.core_efficiency * efficiency)

    dequant_per_elem_us = float(quant_cfg.get("dequant_us_per_output_element", 0.0))
    dequant_us = (
        output_elements * dequant_per_elem_us
        if quant_spec.compute_path in {"full_quant_with_dequant", "weight_only_quant_with_dequant"}
        else 0.0
    )
    if compute_us is not None:
        compute_us += dequant_us

    return compute_us, hbm_us, dequant_us, quant_gm_bytes_min, quant_gm_bytes_tiled


def classify(row: dict[str, Any]) -> tuple[str, str]:
    tags: list[str] = []
    if row.get("kernel_tiling_source") == "runtime_kb_exact":
        tags.append("runtime_kb_exact")
    elif row.get("kernel_tiling_source") == "advanced_tiling_heuristic":
        tags.append("advanced_tiling_heuristic")
    if "stream_k" in str(row.get("tiling_strategy", "")):
        tags.append("stream_k")
    if row.get("full_load") == "A_FULL_LOAD":
        tags.append("al1_full_load")
    elif row.get("full_load") == "B_FULL_LOAD":
        tags.append("bl1_full_load")
    if row.get("l0c2out") and row.get("l0c2out") != "ON_THE_FLY":
        tags.append("fixpipe_output")
    if row.get("quant_mode", "none") != "none":
        tags.append("quant_matmul")
    if row.get("quant_compute_path") == "full_quant_with_dequant":
        tags.append("full_quant_dequant")
    elif row.get("quant_compute_path") == "weight_only_quant_with_dequant":
        tags.append("weight_only_quant_dequant")
    elif row.get("quant_compute_path") == "weight_only_quant":
        tags.append("weight_only_quant")
    elif row.get("quant_compute_path") == "fake_quant_or_mixed":
        tags.append("fake_or_mixed_quant")
    if row["b_format"] == "FRACTAL_NZ":
        tags.append("weight_nz")
    elif row["a_format"] == "FRACTAL_NZ" or row["output_format"] == "FRACTAL_NZ":
        tags.append("fractal_nz")
    if row["nd2nz_a"] or row["nd2nz_b"]:
        tags.append("runtime_nd2nz")
    if row["storage_padding_ratio"] > 1.05:
        tags.append("layout_padding")
    if row["m"] <= 4:
        tags.append("small_m_overhead")
    if row["mn_tile_count"] < row["aic_num"]:
        tags.append("low_tile_count")
    if row["cube_utilization_pct"] and row["cube_utilization_pct"] < 80:
        tags.append("low_cube_utilization")
    if row["compute_us"] is None:
        tags.append("unknown_compute_peak")
    elif row["compute_us"] > row["hbm_us"] * 1.2:
        tags.append("compute_bound")
    elif row["hbm_us"] > row["compute_us"] * 1.2:
        tags.append("memory_bound")
    else:
        tags.append("balanced_bound")
    if row.get("bottleneck") == "launch":
        tags.append("launch_bound")
    if row["duration_us"] > 0 and row["estimated_us"] > 0:
        ratio = row["duration_us"] / row["estimated_us"]
        if ratio > 5:
            tags.append("large_residual")
    confidence = "high"
    if row["m"] <= 4 or row["mn_tile_count"] < row["aic_num"]:
        confidence = "low"
    elif "unknown_compute_peak" in tags:
        confidence = "medium"
    return "|".join(tags), confidence


def iter_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"profiling path not found: {item}")
    return sorted(set(files))


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def evaluate_file(
    path: Path,
    config: dict[str, Any],
    runtime_kb: dict[tuple[Any, ...], list[RuntimeKbEntry]],
    include_gmm: bool,
    include_allgather: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    source_file = display_path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            if not is_matmul_row(row):
                continue
            if is_excluded_by_default(row, include_gmm, include_allgather):
                continue

            input_shapes = parse_shapes(row.get("Input Shapes"))
            output_shapes = parse_shapes(row.get("Output Shapes"))
            input_formats = parse_formats(row.get("Input Formats"))
            output_formats = parse_formats(row.get("Output Formats"))
            spec = infer_matmul_spec(input_shapes, output_shapes, input_formats, output_formats)
            if spec is None:
                unresolved.append(
                    {
                        "file": source_file,
                        "line": line_no,
                        "type": row.get("Type", ""),
                        "name": row.get("Name", ""),
                        "input_shapes": row.get("Input Shapes", ""),
                        "output_shapes": row.get("Output Shapes", ""),
                        "input_formats": row.get("Input Formats", ""),
                        "output_formats": row.get("Output Formats", ""),
                    }
                )
                continue

            dtype = dtype_from_row(row)
            output_dtype = output_dtype_from_row(row)
            input_dtypes = input_dtypes_from_row(row)
            kernel_type = row.get("Type", "")
            tile = select_tile_estimate(spec, dtype, output_dtype, config, runtime_kb, input_dtypes, kernel_type)
            true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
            duration_us = parse_float(row.get("Duration(us)"))
            achieved_tflops = true_flops / duration_us / 1_000_000.0 if duration_us > 0 else 0.0
            launch_us = calibration_value(config, "launch_overhead_us_by_type", kernel_type, 0.0)
            pipeline_eff = calibration_value(config, "pipeline_efficiency_by_dtype", dtype, 1.0)
            pipeline_eff = max(pipeline_eff, 1e-9)
            nd2nz_a, nd2nz_b = infer_nd2nz_operands(spec, dtype)
            format_overhead = 0.0
            if nd2nz_a or nd2nz_b:
                format_overhead += (
                    int(nd2nz_a) + int(nd2nz_b)
                ) * calibration_value(config, "format_overhead_us", "ND2NZ", 0.0)

            quant_spec = infer_quant_spec(row, spec, input_shapes)
            quant_compute_us: float | None = None
            quant_hbm_us: float | None = None
            quant_dequant_us = 0.0
            quant_gm_bytes_min: int | None = None
            quant_gm_bytes_tiled: int | None = None
            if quant_spec.is_quant:
                (
                    quant_compute_us,
                    quant_hbm_us,
                    quant_dequant_us,
                    quant_gm_bytes_min,
                    quant_gm_bytes_tiled,
                ) = estimate_quant_cost(spec, tile, quant_spec, input_shapes, input_dtypes, output_dtype, config)

            compute_for_est = None if tile.compute_us is None else tile.compute_us / pipeline_eff
            if quant_spec.is_quant:
                launch_us = calibration_value(config, "launch_overhead_us_by_type", kernel_type, launch_us)
                compute_for_est = quant_compute_us
                if compute_for_est is None:
                    estimated_us = launch_us + (quant_hbm_us or tile.hbm_us) + format_overhead
                else:
                    estimated_us = launch_us + max(compute_for_est, quant_hbm_us or tile.hbm_us) + format_overhead
            elif compute_for_est is None:
                estimated_us = launch_us + tile.hbm_us + format_overhead
            else:
                estimated_us = launch_us + max(compute_for_est, tile.hbm_us) + format_overhead
            ideal_compute_us, ideal_hbm_us, ideal_lower_bound_us, ideal_gm_bytes_min = ideal_kernel_bounds(
                spec, dtype, output_dtype, config
            )
            best_kernel_us = launch_us + max(
                ideal_hbm_us,
                (ideal_compute_us / pipeline_eff) if ideal_compute_us is not None else ideal_hbm_us,
            )
            current_kernel_bound_us = tile.lower_bound_us
            current_theoretical_tflops = (
                true_flops / current_kernel_bound_us / 1_000_000.0 if current_kernel_bound_us > 0 else 0.0
            )
            best_kernel_tflops = true_flops / best_kernel_us / 1_000_000.0 if best_kernel_us > 0 else 0.0
            ideal_tflops = true_flops / ideal_lower_bound_us / 1_000_000.0 if ideal_lower_bound_us > 0 else 0.0
            bottleneck = dominant_bottleneck(launch_us, compute_for_est, tile.hbm_us, format_overhead)
            logical_storage_elements = (
                spec.batch * spec.m * spec.k
                + spec.batch * spec.k * spec.n
                + spec.batch * spec.m * spec.n
            )
            physical_storage_elements = (
                (spec.a_storage_elements or spec.batch * spec.m * spec.k)
                + (spec.b_storage_elements or spec.batch * spec.k * spec.n)
                + (spec.output_storage_elements or spec.batch * spec.m * spec.n)
            )
            storage_padding_ratio = (
                physical_storage_elements / logical_storage_elements if logical_storage_elements > 0 else 1.0
            )

            result: dict[str, Any] = {
                "file": source_file,
                "line": line_no,
                "name": row.get("Name", ""),
                "type": kernel_type,
                "accelerator_core": row.get("Accelerator Core", ""),
                "dtype": dtype,
                "output_dtype": output_dtype,
                "input_formats": row.get("Input Formats", ""),
                "output_formats": row.get("Output Formats", ""),
                "a_format": spec.a_format,
                "b_format": spec.b_format,
                "output_format": spec.output_format,
                "m": spec.m,
                "n": spec.n,
                "k": spec.k,
                "batch": spec.batch,
                "trans_a": int(spec.trans_a),
                "trans_b": int(spec.trans_b),
                "a_storage_elements": spec.a_storage_elements,
                "b_storage_elements": spec.b_storage_elements,
                "output_storage_elements": spec.output_storage_elements,
                "storage_padding_ratio": storage_padding_ratio,
                "nd2nz_a": int(nd2nz_a),
                "nd2nz_b": int(nd2nz_b),
                "quant_mode": quant_spec.mode,
                "quant_granularity": quant_spec.granularity,
                "quant_compute_path": quant_spec.compute_path,
                "quant_aux_elements": quant_spec.aux_elements,
                "quant_aux_bytes": quant_spec.aux_bytes,
                "quant_notes": quant_spec.notes,
                "kernel_tiling_source": tile.source,
                "runtime_kb_id": tile.runtime_kb_id,
                "runtime_kb_file": tile.runtime_kb_file,
                "tiling_enable": tile.tiling_enable,
                "depth_a1": tile.depth_a1,
                "depth_b1": tile.depth_b1,
                "step_m": tile.step_m,
                "step_n": tile.step_n,
                "step_ka": tile.step_ka,
                "step_kb": tile.step_kb,
                "l2_m_tile": tile.l2_m_tile,
                "l2_n_tile": tile.l2_n_tile,
                "tiling_strategy": tile.tiling_strategy,
                "full_load": tile.full_load,
                "l0c2out": tile.l0c2out,
                "asw_window_len": tile.asw_window_len,
                "l1_buffer_num": tile.l1_buffer_num,
                "ub_db": tile.ub_db,
                "tiling_split_core": tile.tiling_split_core,
                "tiling_full_load": tile.tiling_full_load,
                "tiling_fix_opti": tile.tiling_fix_opti,
                "tiling_special_opti": tile.tiling_special_opti,
                "advanced_tiling_notes": advanced_tiling_notes(spec, dtype, config, kernel_type, tile),
                "block_dim": parse_int(row.get("Block Dim")),
                "mix_block_dim": parse_int(row.get("Mix Block Dim")),
                "aic_num": int(config["aic_num"]),
                "duration_us": duration_us,
                "aicore_time_us": parse_float(row.get("aicore_time(us)")),
                "aic_mac_time_us": parse_float(row.get("aic_mac_time(us)")),
                "aic_mac_ratio": parse_float(row.get("aic_mac_ratio")),
                "aic_mte1_ratio": parse_float(row.get("aic_mte1_ratio")),
                "aic_mte2_ratio": parse_float(row.get("aic_mte2_ratio")),
                "aic_fixpipe_ratio": parse_float(row.get("aic_fixpipe_ratio")),
                "cube_utilization_pct": parse_float(row.get("cube_utilization(%)")),
                "flops": true_flops,
                "achieved_tflops": achieved_tflops,
                "peak_tflops": peak_for_dtype(config, dtype),
                "base_m": tile.base_m,
                "base_n": tile.base_n,
                "base_k": tile.base_k,
                "db_l0c": tile.db_l0c,
                "tile_m": tile.tile_m,
                "tile_n": tile.tile_n,
                "tile_k": tile.tile_k,
                "mn_tile_count": tile.mn_tile_count,
                "tile_count": tile.tile_count,
                "used_core_num_est": tile.used_core_num,
                "core_efficiency": tile.core_efficiency,
                "tail_efficiency": tile.tail_efficiency,
                "aligned_flops": tile.aligned_flops,
                "gm_bytes_min": tile.gm_bytes_min,
                "gm_bytes_tiled_raw": tile.gm_bytes_tiled_raw,
                "gm_bytes_tiled": tile.gm_bytes_tiled,
                "compute_us": quant_compute_us if quant_spec.is_quant else tile.compute_us,
                "hbm_us": quant_hbm_us if quant_spec.is_quant and quant_hbm_us is not None else tile.hbm_us,
                "lower_bound_us": (
                    max(value for value in (quant_compute_us, quant_hbm_us) if value is not None)
                    if quant_spec.is_quant and (quant_compute_us is not None or quant_hbm_us is not None)
                    else tile.lower_bound_us
                ),
                "quant_dequant_us": quant_dequant_us,
                "quant_gm_bytes_min": quant_gm_bytes_min,
                "quant_gm_bytes_tiled": quant_gm_bytes_tiled,
                "launch_overhead_us": launch_us,
                "pipeline_efficiency": pipeline_eff,
                "format_overhead_us": format_overhead,
                "estimated_us": estimated_us,
                "current_kernel_bound_us": current_kernel_bound_us,
                "current_theoretical_tflops": current_theoretical_tflops,
                "ideal_compute_us": ideal_compute_us,
                "ideal_hbm_us": ideal_hbm_us,
                "ideal_lower_bound_us": ideal_lower_bound_us,
                "ideal_gm_bytes_min": ideal_gm_bytes_min,
                "best_kernel_us": best_kernel_us,
                "best_kernel_tflops": best_kernel_tflops,
                "ideal_tflops": ideal_tflops,
                "kernel_gap_to_best": estimated_us / best_kernel_us if best_kernel_us > 0 else None,
                "current_gap_to_ideal": current_kernel_bound_us / ideal_lower_bound_us if ideal_lower_bound_us > 0 else None,
                "bottleneck": bottleneck,
                "residual_us": duration_us - estimated_us if duration_us > 0 else None,
                "duration_over_estimate": duration_us / estimated_us if duration_us > 0 and estimated_us > 0 else None,
            }
            tags, confidence = classify(result)
            result["diagnosis"] = tags
            result["confidence"] = confidence
            records.append(result)
    return records, unresolved


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], unresolved: list[dict[str, Any]]) -> None:
    print(f"resolved_matmul_rows={len(rows)} unresolved_rows={len(unresolved)}")
    if not rows:
        return

    by_file: dict[str, list[dict[str, Any]]] = {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    by_source: dict[str, list[dict[str, Any]]] = {}
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)
        by_type.setdefault(row["type"], []).append(row)
        by_source.setdefault(row.get("kernel_tiling_source", "unknown"), []).append(row)
        by_strategy.setdefault(row.get("tiling_strategy") or "none", []).append(row)

    def median(values: Iterable[float]) -> float:
        values = list(values)
        return statistics.median(values) if values else 0.0

    print("\nBy file:")
    for file_name, file_rows in sorted(by_file.items()):
        print(
            f"  {file_name}: n={len(file_rows)} "
            f"median_tflops={median(row['achieved_tflops'] for row in file_rows):.3f} "
            f"max_tflops={max(row['achieved_tflops'] for row in file_rows):.3f} "
            f"median_cube={median(row['cube_utilization_pct'] for row in file_rows):.2f}%"
        )

    print("\nBy type:")
    for kernel_type, type_rows in sorted(by_type.items()):
        print(
            f"  {kernel_type}: n={len(type_rows)} "
            f"median_tflops={median(row['achieved_tflops'] for row in type_rows):.3f} "
            f"max_tflops={max(row['achieved_tflops'] for row in type_rows):.3f} "
            f"median_duration_over_estimate={median(row['duration_over_estimate'] or 0.0 for row in type_rows):.2f}"
        )

    print("\nBy tiling source:")
    for source, source_rows in sorted(by_source.items()):
        print(
            f"  {source}: n={len(source_rows)} "
            f"median_current_gap_to_ideal={median(row['current_gap_to_ideal'] or 0.0 for row in source_rows):.2f} "
            f"median_kernel_gap_to_best={median(row['kernel_gap_to_best'] or 0.0 for row in source_rows):.2f}"
        )

    print("\nBy kernel strategy:")
    for strategy, strategy_rows in sorted(by_strategy.items()):
        print(
            f"  {strategy}: n={len(strategy_rows)} "
            f"median_current_tflops={median(row['current_theoretical_tflops'] for row in strategy_rows):.3f} "
            f"median_bottleneck={statistics.mode(row['bottleneck'] for row in strategy_rows)}"
        )

    print("\nTop residual examples:")
    residual_rows = [row for row in rows if row["residual_us"] is not None]
    residual_rows.sort(key=lambda row: row["duration_over_estimate"] or 0.0, reverse=True)
    for row in residual_rows[:10]:
        print(
            f"  {row['file']}:{row['line']} {row['type']} "
            f"dtype={row['dtype']} M,N,K={row['m']},{row['n']},{row['k']} "
            f"duration={row['duration_us']:.3f}us estimate={row['estimated_us']:.3f}us "
            f"ratio={row['duration_over_estimate']:.2f} diagnosis={row['diagnosis']}"
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * pct
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return values[int(position)]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def calibration_suggestions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Suggest global calibration constants without fitting per-shape curves."""
    launch_by_type: dict[str, float] = {}
    pipeline_by_dtype: dict[str, float] = {}
    quant_pipeline: dict[str, float] = {}

    kernel_types = sorted({row["type"] for row in rows})
    for kernel_type in kernel_types:
        candidates = [
            row["duration_us"] - row["lower_bound_us"]
            for row in rows
            if row["type"] == kernel_type
            and (row["m"] <= 4 or row["mn_tile_count"] < row["aic_num"])
            and row["duration_us"] > row["lower_bound_us"]
        ]
        if candidates:
            # Low percentile avoids baking memory/cache misses into launch cost.
            launch_by_type[kernel_type] = round(max(0.0, percentile(candidates, 0.2)), 6)

    dtypes = sorted({row["dtype"] for row in rows})
    for dtype in dtypes:
        efficiencies = [
            min(1.0, row["achieved_tflops"] / row["peak_tflops"])
            for row in rows
            if row["dtype"] == dtype
            and row["peak_tflops"]
            and row.get("quant_mode", "none") == "none"
            and row["m"] >= 128
            and row["n"] >= 128
            and row["k"] >= 128
            and row["cube_utilization_pct"] >= 95
            and row["mn_tile_count"] >= row["aic_num"]
        ]
        if efficiencies:
            # High percentile approximates sustained compute efficiency from
            # the best compute-dominant examples, not an interpolation curve.
            pipeline_by_dtype[dtype] = round(max(0.1, percentile(efficiencies, 0.9)), 6)

    quant_modes = sorted({row.get("quant_mode", "none") for row in rows if row.get("quant_mode", "none") != "none"})
    for mode in quant_modes:
        efficiencies = [
            min(
                1.0,
                row["aligned_flops"] / (row["duration_us"] * row["peak_tflops"] * 1_000_000.0 * max(row["core_efficiency"], 1e-9)),
            )
            for row in rows
            if row.get("quant_mode") == mode
            and row["duration_us"] > 0
            and row["peak_tflops"]
            and row.get("quant_compute_path") in {"full_quant", "full_quant_with_dequant"}
            and row["cube_utilization_pct"] >= 90
        ]
        if efficiencies:
            quant_pipeline[mode.upper()] = round(max(0.1, percentile(efficiencies, 0.5)), 6)

    suggestions: dict[str, Any] = {
        "calibration": {
            "launch_overhead_us_by_type": launch_by_type,
            "pipeline_efficiency_by_dtype": pipeline_by_dtype,
        }
    }
    if quant_pipeline:
        suggestions["quant_matmul"] = {"pipeline_efficiency": quant_pipeline}
    return suggestions


def print_calibration_suggestions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    suggestions = calibration_suggestions(rows)
    print("\nCalibration suggestions:")
    print(json.dumps(suggestions, indent=2, sort_keys=True))
    return suggestions


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiling",
        nargs="+",
        default=["example_profilings/910B4"],
        help="Profiling CSV file(s) or directories. Default: example_profilings/910B4",
    )
    parser.add_argument(
        "--config",
        default="configs/ascend_910b4.json",
        help="Hardware/config JSON. Default: configs/ascend_910b4.json",
    )
    parser.add_argument("--output", help="Write detailed resolved report CSV.")
    parser.add_argument("--unresolved-output", help="Write unresolved matmul rows CSV.")
    parser.add_argument(
        "--suggest-calibration",
        action="store_true",
        help="Print global launch/pipeline calibration suggestions from residuals.",
    )
    parser.add_argument("--calibration-output", help="Write calibration suggestions JSON.")
    parser.add_argument("--include-gmm", action="store_true", help="Include GroupedMatmul rows.")
    parser.add_argument("--include-allgather", action="store_true", help="Include AllGatherMatmul rows.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with Path(args.config).open() as handle:
        config = json.load(handle)
    runtime_kb = load_runtime_kb(config)

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for profiling_file in iter_input_files(args.profiling):
        file_rows, file_unresolved = evaluate_file(
            profiling_file,
            config=config,
            runtime_kb=runtime_kb,
            include_gmm=args.include_gmm,
            include_allgather=args.include_allgather,
        )
        rows.extend(file_rows)
        unresolved.extend(file_unresolved)

    print_summary(rows, unresolved)

    suggestions: dict[str, Any] | None = None
    if args.suggest_calibration or args.calibration_output:
        suggestions = print_calibration_suggestions(rows)

    if args.output:
        write_csv(Path(args.output), rows)
        print(f"\nwrote_report={args.output}")
    if args.unresolved_output:
        write_csv(Path(args.unresolved_output), unresolved)
        print(f"wrote_unresolved={args.unresolved_output}")
    if args.calibration_output:
        assert suggestions is not None
        Path(args.calibration_output).write_text(json.dumps(suggestions, indent=2, sort_keys=True) + "\n")
        print(f"wrote_calibration={args.calibration_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
