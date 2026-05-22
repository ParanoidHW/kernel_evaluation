from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as handle:
        return json.load(handle)


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


def first_int(row: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if value not in {"", None, "N/A"}:
            return parse_int(value, default)
    return default


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


def is_fp32_dtype(dtype: str) -> bool:
    return dtype in {"FLOAT", "FLOAT32", "DT_FLOAT"}


def split_semicolon_values(value: str | None) -> list[str]:
    if not value or value == "N/A":
        return []
    return [part.strip() for part in value.strip().replace('"', "").split(";") if part.strip()]


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "DTYPE_BITS",
    "DTYPE_BYTES",
    "GE_DTYPE_RUNTIME_KB",
    "GE_FORMAT_RUNTIME_KB",
    "calibration_value",
    "ceil_align",
    "ceil_div",
    "display_path",
    "dtype_bitwidth",
    "dtype_size",
    "floor_align",
    "format_at",
    "is_fp32_dtype",
    "is_quant_kernel_type",
    "is_quantized_data_dtype",
    "is_scale_dtype",
    "load_config",
    "normalize_format",
    "num_elements",
    "parse_float",
    "parse_formats",
    "parse_int",
    "parse_shapes",
    "peak_for_dtype",
    "split_semicolon_values",
]
