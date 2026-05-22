#!/usr/bin/env python3
"""Append kernel evaluation values to an existing profiling CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from attention_eval.api import estimate_attention_cost
from attention_eval.common import (
    infer_attention_spec,
    input_dtypes_from_row as attention_input_dtypes_from_row,
    is_attention_row,
    output_dtype_from_row as attention_output_dtype_from_row,
)
from matmul_eval.api import estimate_matmul_cost
from matmul_eval.common import (
    dtype_from_row,
    infer_grouped_matmul_spec,
    infer_matmul_spec,
    infer_transpose_batch_matmul_spec,
    input_dtypes_from_row as matmul_input_dtypes_from_row,
    is_matmul_row,
    output_dtype_from_row as matmul_output_dtype_from_row,
)
from matmul_eval.gmm_model import estimate_grouped_matmul_bounds
from matmul_eval.quant_model import infer_quant_spec
from matmul_eval.runtime_kb import load_runtime_kb
from op_eval.common import load_config, parse_formats, parse_shapes


DEFAULT_COLUMN = "kernel_eval_value"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one kernel-evaluation column to a profiling CSV. "
        "Unsupported non-MatMul/GMM/Attention rows are left empty."
    )
    parser.add_argument("--profiling", required=True, help="Input profiling CSV file.")
    parser.add_argument("--config", required=True, help="Hardware/config JSON file.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help=f"Name of the appended column. Default: {DEFAULT_COLUMN}",
    )
    parser.add_argument(
        "--gmm-value",
        choices=["midpoint", "bounds"],
        default="midpoint",
        help="How to encode GroupedMatmul rows. Default: midpoint.",
    )
    return parser.parse_args(argv)


def _first_attention_compute_dtype(dtypes: list[str]) -> str:
    for dtype in dtypes:
        normalized = dtype.upper()
        if normalized not in {"DT_UNDEFINED", "UNDEFINED", "BOOL", "INT32", "INT64"}:
            return normalized
    return dtypes[0].upper() if dtypes else "UNKNOWN"


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _estimate_attention_row(row: dict[str, str], config: dict[str, Any]) -> str:
    input_shapes = parse_shapes(row.get("Input Shapes"))
    output_shapes = parse_shapes(row.get("Output Shapes"))
    kernel_type = row.get("Type", "")
    spec = infer_attention_spec(input_shapes, output_shapes, kernel_type=kernel_type)
    if spec is None:
        return ""
    input_dtypes = attention_input_dtypes_from_row(row)
    dtype = _first_attention_compute_dtype(input_dtypes)
    output_dtype = attention_output_dtype_from_row(row)
    cost = estimate_attention_cost(
        spec,
        dtype,
        config=config,
        output_dtype=output_dtype,
        kernel_type=kernel_type,
    )
    return _format_float(cost.total_us)


def _estimate_matmul_row(
    row: dict[str, str],
    config: dict[str, Any],
    runtime_kb: dict[tuple[Any, ...], list[Any]],
    *,
    gmm_value: str,
) -> str:
    input_shapes = parse_shapes(row.get("Input Shapes"))
    output_shapes = parse_shapes(row.get("Output Shapes"))
    input_formats = parse_formats(row.get("Input Formats"))
    output_formats = parse_formats(row.get("Output Formats"))
    kernel_type = row.get("Type", "")
    kernel_type_lower = kernel_type.lower()

    if kernel_type_lower == "groupedmatmul":
        spec = infer_grouped_matmul_spec(input_shapes, output_shapes, input_formats, output_formats)
    else:
        spec = infer_matmul_spec(input_shapes, output_shapes, input_formats, output_formats)
    if spec is None and kernel_type_lower == "transposebatchmatmul":
        spec = infer_transpose_batch_matmul_spec(input_shapes, output_shapes, input_formats, output_formats)
    if spec is None:
        return ""

    dtype = dtype_from_row(row)
    output_dtype = matmul_output_dtype_from_row(row)
    input_dtypes = matmul_input_dtypes_from_row(row)
    quant_spec = infer_quant_spec(row, spec, input_shapes)

    if kernel_type_lower == "groupedmatmul":
        bounds = estimate_grouped_matmul_bounds(
            spec,
            input_shapes,
            input_dtypes,
            dtype,
            output_dtype,
            quant_spec,
            config,
        )
        if bounds is None:
            return ""
        low = min(bounds.balanced.total_us, bounds.extreme.total_us)
        high = max(bounds.balanced.total_us, bounds.extreme.total_us)
        if gmm_value == "midpoint":
            return _format_float((low + high) / 2.0)
        return f"[{_format_float(low)},{_format_float(high)}]"

    cost = estimate_matmul_cost(
        spec,
        dtype,
        config=config,
        output_dtype=output_dtype,
        runtime_kb=runtime_kb,
        input_dtypes=input_dtypes,
        kernel_type=kernel_type,
        quant_spec=quant_spec,
        input_shapes=input_shapes,
    )
    return _format_float(cost.total_us)


def estimate_row(
    row: dict[str, str],
    config: dict[str, Any],
    runtime_kb: dict[tuple[Any, ...], list[Any]],
    *,
    gmm_value: str,
) -> str:
    if is_attention_row(row):
        return _estimate_attention_row(row, config)
    if is_matmul_row(row):
        text = f"{row.get('Name', '')} {row.get('Type', '')}".lower()
        if "allgathermatmul" in text:
            return ""
        return _estimate_matmul_row(row, config, runtime_kb, gmm_value=gmm_value)
    return ""


def annotate_csv(input_path: Path, output_path: Path, config: dict[str, Any], column: str, gmm_value: str) -> tuple[int, int]:
    runtime_kb = load_runtime_kb(config)
    total_rows = 0
    annotated_rows = 0
    with input_path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"input CSV has no header: {input_path}")
        fieldnames = list(reader.fieldnames)
        if column not in fieldnames:
            fieldnames.append(column)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                total_rows += 1
                value = estimate_row(row, config, runtime_kb, gmm_value=gmm_value)
                if value:
                    annotated_rows += 1
                row[column] = value
                writer.writerow(row)
    return total_rows, annotated_rows


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_path = Path(args.profiling)
    if not input_path.is_file():
        raise FileNotFoundError(f"profiling CSV not found: {input_path}")
    config = load_config(args.config)
    total_rows, annotated_rows = annotate_csv(
        input_path,
        Path(args.output),
        config,
        args.column,
        args.gmm_value,
    )
    print(f"rows={total_rows} annotated_rows={annotated_rows} skipped_rows={total_rows - annotated_rows}")
    print(f"wrote_output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
