from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Iterable

from op_eval.common import display_path, first_int, parse_float, parse_int, parse_shapes

from .api import estimate_attention_cost
from .common import infer_attention_spec, input_dtypes_from_row, is_attention_row, output_dtype_from_row


def _first_compute_dtype(dtypes: list[str]) -> str:
    for dtype in dtypes:
        normalized = dtype.upper()
        if normalized not in {"DT_UNDEFINED", "UNDEFINED", "BOOL", "INT32", "INT64"}:
            return normalized
    return dtypes[0].upper() if dtypes else "UNKNOWN"


def classify(row: dict[str, Any]) -> tuple[str, str]:
    tags: list[str] = [row["variant"]]
    if row["actual_tiling_source"].startswith("unavailable"):
        tags.append("actual_tiling_unavailable")
    elif row["actual_tiling_source"] == "ops_transformer_source_strategy_replay":
        tags.append("ops_transformer_source_strategy_replay")
    if row["causal_or_masked"]:
        tags.append("masked_or_aux_inputs")
    if row["q_seq"] <= 1:
        tags.append("decode_like")
    elif row["q_seq"] >= 1024:
        tags.append("prefill_like")
    if row["kv_heads"] < row["q_heads"]:
        tags.append("mqa_gqa")
    if row["variant"] == "kv_quant_sparse_flash_attention":
        tags.append("specialized_kv_quant_sparse_path")
        tags.append("generic_attention_cost_low_confidence")
    tags.append(row["kernel_bound_type"])
    if row["duration_us"] > 0 and row["estimated_us"] > 0 and row["duration_us"] / row["estimated_us"] > 5:
        tags.append("large_residual")
    confidence = "low" if row["actual_tiling_source"].startswith("unavailable") else "medium"
    if row["variant"] == "kv_quant_sparse_flash_attention":
        confidence = "low"
    return "|".join(tags), confidence


def evaluate_file(path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    source_file = display_path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            if not is_attention_row(row):
                continue
            input_shapes = parse_shapes(row.get("Input Shapes"))
            output_shapes = parse_shapes(row.get("Output Shapes"))
            kernel_type = row.get("Type", "")
            spec = infer_attention_spec(input_shapes, output_shapes, kernel_type=kernel_type)
            if spec is None:
                unresolved.append(
                    {
                        "file": source_file,
                        "line": line_no,
                        "type": kernel_type,
                        "name": row.get("Name", ""),
                        "input_shapes": row.get("Input Shapes", ""),
                        "output_shapes": row.get("Output Shapes", ""),
                    }
                )
                continue
            input_dtypes = input_dtypes_from_row(row)
            dtype = _first_compute_dtype(input_dtypes)
            output_dtype = output_dtype_from_row(row)
            cost = estimate_attention_cost(
                spec,
                dtype,
                config=config,
                output_dtype=output_dtype,
                kernel_type=kernel_type,
            )
            duration_us = parse_float(row.get("Duration(us)"))
            achieved_tflops = cost.flops / duration_us / 1_000_000.0 if duration_us > 0 else 0.0
            result: dict[str, Any] = {
                "file": source_file,
                "line": line_no,
                "name": row.get("Name", ""),
                "type": kernel_type,
                "variant": spec.variant,
                "accelerator_core": row.get("Accelerator Core", ""),
                "dtype": dtype,
                "output_dtype": output_dtype,
                "batch": spec.batch,
                "q_heads": spec.q_heads,
                "kv_heads": spec.kv_heads,
                "q_seq": spec.q_seq,
                "kv_seq": spec.kv_seq,
                "head_dim": spec.head_dim,
                "value_dim": spec.value_dim,
                "layout": spec.layout,
                "causal_or_masked": int(spec.causal_or_masked),
                "q_elements": spec.q_elements,
                "k_elements": spec.k_elements,
                "v_elements": spec.v_elements,
                "aux_elements": spec.aux_elements,
                "raw_aux_elements": spec.raw_aux_elements,
                "output_elements": spec.output_elements,
                "score_elements": spec.score_elements,
                "block_dim": first_int(row, "Block Dim", "Block Num"),
                "mix_block_dim": first_int(row, "Mix Block Dim", "Mix Block Num"),
                "aic_num": int(config["aic_num"]),
                "duration_us": duration_us,
                "aicore_time_us": parse_float(row.get("aicore_time(us)")),
                "aic_mac_time_us": parse_float(row.get("aic_mac_time(us)")),
                "aic_mac_ratio": parse_float(row.get("aic_mac_ratio")),
                "aic_mte1_ratio": parse_float(row.get("aic_mte1_ratio")),
                "aic_mte2_ratio": parse_float(row.get("aic_mte2_ratio")),
                "aic_fixpipe_ratio": parse_float(row.get("aic_fixpipe_ratio")),
                "aiv_time_us": parse_float(row.get("aiv_time(us)")),
                "aiv_vec_ratio": parse_float(row.get("aiv_vec_ratio")),
                "cube_utilization_pct": parse_float(row.get("cube_utilization(%)")),
                "flops": cost.flops,
                "vector_ops": cost.vector_ops,
                "achieved_tflops": achieved_tflops,
                "compute_us": cost.compute_us,
                "vector_us": cost.vector_us,
                "hbm_us": cost.hbm_us,
                "current_compute_us": cost.current_compute_us,
                "current_vector_us": cost.current_vector_us,
                "current_hbm_us": cost.current_hbm_us,
                "gm_bytes_min": cost.gm_bytes_min,
                "current_gm_bytes": cost.current_gm_bytes,
                "lower_bound_us": cost.lower_bound_us,
                "current_kernel_bound_us": cost.current_kernel_bound_us,
                "launch_overhead_us": cost.launch_overhead_us,
                "pipeline_efficiency": cost.pipeline_efficiency,
                "occupancy_efficiency": cost.occupancy_efficiency,
                "traffic_factor": cost.traffic_factor,
                "q_block_tiles": cost.q_block_tiles,
                "kv_block_tiles": cost.kv_block_tiles,
                "work_tiles": cost.work_tiles,
                "sync_overhead_us": cost.sync_overhead_us,
                "latency_floor_us": cost.latency_floor_us,
                "template_overhead_factor": cost.template_overhead_factor,
                "estimated_us": cost.total_us,
                "total_us": cost.total_us,
                "ideal_lower_bound_us": cost.lower_bound_us,
                "current_theoretical_tflops": cost.flops / cost.current_kernel_bound_us / 1_000_000.0 if cost.current_kernel_bound_us > 0 else 0.0,
                "kernel_gap_to_best": cost.total_us / cost.lower_bound_us if cost.lower_bound_us > 0 else None,
                "current_gap_to_ideal": cost.current_kernel_bound_us / cost.lower_bound_us if cost.lower_bound_us > 0 else None,
                "actual_tiling_source": cost.actual_tiling_source,
                "fallback_tiling_source": cost.fallback_tiling_source,
                "optimal_tiling_source": cost.optimal_tiling_source,
                "current_tiling_kind": cost.current_tiling_kind,
                "tiling_strategy": cost.tiling_strategy,
                "ops_transformer_source_file": cost.ops_transformer_source_file,
                "tiling_notes": cost.tiling_notes,
                "bottleneck": cost.dominant_component,
                "kernel_bound_type": "unknown_compute_bound" if cost.compute_us is None else cost.bound_type,
                "bound_type": cost.bound_type,
                "residual_us": duration_us - cost.total_us if duration_us > 0 else None,
                "duration_over_estimate": duration_us / cost.total_us if duration_us > 0 and cost.total_us > 0 else None,
            }
            tags, confidence = classify(result)
            result["diagnosis"] = tags
            result["confidence"] = confidence
            records.append(result)
    return records, unresolved


def print_summary(rows: list[dict[str, Any]], unresolved: list[dict[str, Any]]) -> None:
    print(f"resolved_attention_rows={len(rows)} unresolved_rows={len(unresolved)}")
    if not rows:
        return

    by_type: dict[str, list[dict[str, Any]]] = {}
    by_variant: dict[str, list[dict[str, Any]]] = {}
    by_tiling: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(row)
        by_variant.setdefault(row["variant"], []).append(row)
        by_tiling.setdefault(row["current_tiling_kind"], []).append(row)

    def median(values: Iterable[float]) -> float:
        values = list(values)
        return statistics.median(values) if values else 0.0

    print("\nBy type:")
    for kernel_type, type_rows in sorted(by_type.items()):
        print(
            f"  {kernel_type}: n={len(type_rows)} "
            f"median_tflops={median(row['achieved_tflops'] for row in type_rows):.3f} "
            f"median_duration_over_estimate={median(row['duration_over_estimate'] or 0.0 for row in type_rows):.2f}"
        )

    print("\nBy variant:")
    for variant, variant_rows in sorted(by_variant.items()):
        print(
            f"  {variant}: n={len(variant_rows)} "
            f"median_q_seq={median(row['q_seq'] for row in variant_rows):.0f} "
            f"median_kv_seq={median(row['kv_seq'] for row in variant_rows):.0f} "
            f"median_bottleneck={statistics.mode(row['bottleneck'] for row in variant_rows)}"
        )

    print("\nBy tiling kind:")
    for kind, kind_rows in sorted(by_tiling.items()):
        print(
            f"  {kind}: n={len(kind_rows)} "
            f"median_duration_over_estimate={median(row['duration_over_estimate'] or 0.0 for row in kind_rows):.2f}"
        )

    print("\nTop residual examples:")
    residual_rows = [row for row in rows if row["duration_over_estimate"] is not None]
    residual_rows.sort(key=lambda row: row["duration_over_estimate"] or 0.0, reverse=True)
    for row in residual_rows[:10]:
        print(
            f"  {row['file']}:{row['line']} {row['type']} "
            f"B,H,Sq,Sk,D={row['batch']},{row['q_heads']},{row['q_seq']},{row['kv_seq']},{row['head_dim']} "
            f"duration={row['duration_us']:.3f}us estimate={row['estimated_us']:.3f}us "
            f"ratio={row['duration_over_estimate']:.2f} diagnosis={row['diagnosis']}"
        )
