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
    "INT4": 1,
    "UINT4": 1,
    "FLOAT8": 1,
    "FP8": 1,
    "MXFP8": 1,
}

DTYPE_BITS = {
    "INT4": 4,
    "UINT4": 4,
    "INT8": 8,
    "DT_INT8": 8,
    "UINT8": 8,
    "FLOAT8": 8,
    "FP8": 8,
    "MXFP8": 8,
}


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


@dataclass(frozen=True)
class QuantSpec:
    is_quant: bool
    mode: str = "none"
    granularity: str = "none"
    compute_path: str = "non_quant"
    aux_elements: int = 0
    aux_bytes: int = 0
    notes: str = ""


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


def dtype_size(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype, DTYPE_BYTES.get(dtype.replace("DT_", ""), 4))


def dtype_bitwidth(dtype: str) -> int:
    normalized = dtype.upper().replace("DT_", "")
    if dtype.upper() in DTYPE_BITS:
        return DTYPE_BITS[dtype.upper()]
    return DTYPE_BITS.get(normalized, dtype_size(dtype) * 8)


def is_quantized_data_dtype(dtype: str) -> bool:
    normalized = dtype.upper().replace("DT_", "")
    return normalized in {"INT4", "UINT4", "INT8", "UINT8", "FLOAT8", "FP8", "MXFP8"}


def is_scale_dtype(dtype: str) -> bool:
    normalized = dtype.upper().replace("DT_", "")
    return normalized in {"FLOAT", "FLOAT32", "FLOAT16", "BFLOAT16", "BF16"}


def is_quant_kernel_type(kernel_type: str) -> bool:
    return "quant" in kernel_type.lower()


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


def candidate_base_values(limit: int) -> list[int]:
    max_value = max(16, min(512, ceil_align(limit, 16)))
    values = list(range(16, max_value + 1, 16))
    preferred = [64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 336]
    return sorted(set(values + [value for value in preferred if value <= max_value]))


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
    has_fp8 = any(dtype.upper().replace("DT_", "") in {"FLOAT8", "FP8", "MXFP8"} for dtype in input_dtypes)
    mode = "mxfp8" if has_fp8 else "int"
    mode += f"{min(dtype_bitwidth(a_dtype), dtype_bitwidth(b_dtype))}"

    if quant_data_inputs == 2 and has_scale and output_dtype in {"FLOAT16", "DT_FLOAT16", "DT_BF16", "BFLOAT16", "FLOAT", "DT_FLOAT"}:
        compute_path = "full_quant_with_dequant"
    elif quant_data_inputs == 2:
        compute_path = "full_quant"
    else:
        compute_path = "fake_quant_or_mixed"

    granularity = "unknown"
    notes: list[str] = []
    for shape in aux_shapes:
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
    dequant_us = output_elements * dequant_per_elem_us if quant_spec.compute_path == "full_quant_with_dequant" else 0.0
    if compute_us is not None:
        compute_us += dequant_us

    return compute_us, hbm_us, dequant_us, quant_gm_bytes_min, quant_gm_bytes_tiled


def classify(row: dict[str, Any]) -> tuple[str, str]:
    tags: list[str] = []
    if row.get("quant_mode", "none") != "none":
        tags.append("quant_matmul")
    if row.get("quant_compute_path") == "full_quant_with_dequant":
        tags.append("full_quant_dequant")
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
            tile = estimate_tile(spec, dtype, output_dtype, config)
            true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
            duration_us = parse_float(row.get("Duration(us)"))
            achieved_tflops = true_flops / duration_us / 1_000_000.0 if duration_us > 0 else 0.0
            kernel_type = row.get("Type", "")
            launch_us = calibration_value(config, "launch_overhead_us_by_type", kernel_type, 0.0)
            pipeline_eff = calibration_value(config, "pipeline_efficiency_by_dtype", dtype, 1.0)
            pipeline_eff = max(pipeline_eff, 1e-9)
            nd2nz_a, nd2nz_b = infer_nd2nz_operands(spec, dtype)
            format_overhead = 0.0
            if nd2nz_a or nd2nz_b:
                format_overhead += (
                    int(nd2nz_a) + int(nd2nz_b)
                ) * calibration_value(config, "format_overhead_us", "ND2NZ", 0.0)

            input_dtypes = split_semicolon_values(row.get("Input Data Types"))
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
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)
        by_type.setdefault(row["type"], []).append(row)

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

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for profiling_file in iter_input_files(args.profiling):
        file_rows, file_unresolved = evaluate_file(
            profiling_file,
            config=config,
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
