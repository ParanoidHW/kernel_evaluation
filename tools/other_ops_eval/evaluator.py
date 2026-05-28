from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from op_eval.common import (
    display_path,
    parse_float,
    parse_formats,
    parse_shapes,
    split_semicolon_values,
)

from .api import estimate_other_op
from .common import build_spec, is_other_ops_row


def _diagnosis(spec: Any, cost: Any, duration_us: float) -> tuple[str, str]:
    tags = [spec.op_family, cost.tiling_source, f"{cost.dominant_component}_bound"]
    confidence = "medium"
    if spec.missing_attrs:
        tags.append("missing_runtime_attrs")
        confidence = "low"
    if "missing" in cost.tiling_source:
        confidence = "low"
    if spec.op_family in {"index_scatter_routing", "cv_regular"}:
        confidence = "low"
    if duration_us > 0 and cost.total_us > 0:
        ratio = duration_us / cost.total_us
        if ratio > 5:
            tags.append("large_residual")
    return "|".join(tags), confidence


def evaluate_file(path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    source_file = display_path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            if not is_other_ops_row(row):
                continue
            op_type = row.get("Type", "")
            input_shapes = parse_shapes(row.get("Input Shapes"))
            output_shapes = parse_shapes(row.get("Output Shapes"))
            input_dtypes = split_semicolon_values(row.get("Input Data Types"))
            output_dtypes = split_semicolon_values(row.get("Output Data Types"))
            input_formats = parse_formats(row.get("Input Formats"))
            output_formats = parse_formats(row.get("Output Formats"))
            spec = build_spec(op_type, input_shapes, output_shapes, input_dtypes, output_dtypes, input_formats, output_formats)
            if spec is None:
                unresolved.append(
                    {
                        "file": source_file,
                        "line": line_no,
                        "type": op_type,
                        "name": row.get("Name", ""),
                        "reason": "unsupported_other_op_or_unparseable_shapes",
                        "input_shapes": row.get("Input Shapes", ""),
                        "output_shapes": row.get("Output Shapes", ""),
                        "input_dtypes": row.get("Input Data Types", ""),
                        "output_dtypes": row.get("Output Data Types", ""),
                        "input_formats": row.get("Input Formats", ""),
                        "output_formats": row.get("Output Formats", ""),
                    }
                )
                continue
            cost = estimate_other_op(spec, config)
            duration_us = parse_float(row.get("Duration(us)"))
            residual_us = duration_us - cost.total_us
            duration_over_estimate = duration_us / cost.total_us if cost.total_us > 0 else 0.0
            diagnosis, confidence = _diagnosis(spec, cost, duration_us)
            records.append(
                {
                    "file": source_file,
                    "line": line_no,
                    "name": row.get("Name", ""),
                    "type": op_type,
                    "accelerator_core": row.get("Accelerator Core", ""),
                    "op_family": spec.op_family,
                    "source_repo": spec.source_repo,
                    "source_path": spec.source_path,
                    "source_strategy": spec.source_strategy,
                    "layout_pattern": spec.layout_pattern,
                    "tiling_source": cost.tiling_source,
                    "missing_attrs": spec.missing_attrs,
                    "input_shapes": row.get("Input Shapes", ""),
                    "output_shapes": row.get("Output Shapes", ""),
                    "input_dtypes": row.get("Input Data Types", ""),
                    "output_dtypes": row.get("Output Data Types", ""),
                    "input_formats": row.get("Input Formats", ""),
                    "output_formats": row.get("Output Formats", ""),
                    "input_elements": ";".join(str(value) for value in spec.input_elements),
                    "output_elements": ";".join(str(value) for value in spec.output_elements),
                    "input_bytes": sum(spec.input_bytes),
                    "output_bytes": sum(spec.output_bytes),
                    "logical_elements": spec.logical_elements,
                    "block_dim": row.get("Block Dim", ""),
                    "mix_block_dim": row.get("Mix Block Dim", ""),
                    "aicore_time_us": row.get("aicore_time(us)", ""),
                    "aiv_time_us": row.get("aiv_time(us)", ""),
                    "vector_compute_us": cost.vector_compute_us,
                    "cube_compute_us": cost.cube_compute_us,
                    "hbm_us": cost.hbm_us,
                    "layout_overhead_us": cost.layout_overhead_us,
                    "workspace_us": cost.workspace_us,
                    "sync_overhead_us": cost.sync_overhead_us,
                    "launch_overhead_us": cost.launch_overhead_us,
                    "current_kernel_bound_us": cost.current_kernel_bound_us,
                    "ideal_lower_bound_us": cost.ideal_lower_bound_us,
                    "estimated_us": cost.total_us,
                    "total_us": cost.total_us,
                    "duration_us": duration_us,
                    "residual_us": residual_us,
                    "duration_over_estimate": duration_over_estimate,
                    "bottleneck": cost.dominant_component,
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                }
            )
    return records, unresolved


def print_summary(rows: list[dict[str, Any]], unresolved: list[dict[str, Any]]) -> None:
    print(f"resolved_other_ops_rows={len(rows)} unresolved_rows={len(unresolved)}")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row.get("op_family", "")), []).append(row)
    print("\nBy family:")
    for family, family_rows in sorted(by_family.items(), key=lambda item: len(item[1]), reverse=True):
        ratios = [float(row["duration_over_estimate"]) for row in family_rows if float(row.get("estimated_us", 0) or 0) > 0]
        median_ratio = statistics.median(ratios) if ratios else 0.0
        total_duration = sum(float(row.get("duration_us", 0) or 0) for row in family_rows)
        print(f"  {family}: n={len(family_rows)} total_duration_us={total_duration:.3f} median_duration_over_estimate={median_ratio:.2f}")
    if unresolved:
        print("\nTop unresolved types:")
        counts: dict[str, int] = {}
        for row in unresolved:
            counts[str(row.get("type", ""))] = counts.get(str(row.get("type", "")), 0) + 1
        for op_type, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]:
            print(f"  {op_type}: n={count}")
    tails = sorted(
        (
            (
                abs(float(row["estimated_us"]) - float(row["duration_us"])) / max(float(row["duration_us"]), 1e-9),
                row,
            )
            for row in rows
            if float(row.get("duration_us", 0) or 0) > 0
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    print("\nTop residual examples:")
    for rel, row in tails[:10]:
        print(
            f"  {row['file']}:{row['line']} {row['type']} family={row['op_family']} "
            f"duration={float(row['duration_us']):.3f}us estimate={float(row['estimated_us']):.3f}us "
            f"rel={rel:.3f} diagnosis={row['diagnosis']}"
        )
