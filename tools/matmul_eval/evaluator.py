from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from op_eval.common import first_int
from op_eval.profiling import is_excluded_by_default

from .api import estimate_matmul_cost
from .common import *
from .gmm_model import estimate_grouped_matmul_bounds
from .kernel_model import advanced_tiling_notes, ideal_kernel_bounds
from .quant_model import infer_quant_spec


def tiling_source_split(source: str) -> tuple[str, str, str, str]:
    """Classify current tiling source into actual/fallback/optimal semantics."""

    if source in {"runtime_kb_exact", "advanced_tiling_heuristic"}:
        return source, "", "physical_lower_bound", "actual_tiling"
    if source == "analytic_search":
        return "", source, "physical_lower_bound", "fallback_tiling"
    return "", source or "unknown", "physical_lower_bound", "fallback_tiling"


def classify(row: dict[str, Any]) -> tuple[str, str]:
    tags: list[str] = []
    if row.get("kernel_tiling_source") == "runtime_kb_exact":
        tags.append("runtime_kb_exact")
    elif row.get("kernel_tiling_source") == "advanced_tiling_heuristic":
        tags.append("advanced_tiling_heuristic")
    if row.get("current_tiling_kind") == "fallback_tiling":
        tags.append("fallback_tiling")
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
    if row.get("gmm_model_kind"):
        tags.append(str(row["gmm_model_kind"]))
        if row.get("gmm_duration_position"):
            tags.append(str(row["gmm_duration_position"]))
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
    if row.get("current_tiling_kind") == "fallback_tiling":
        confidence = "medium"
    if row["m"] <= 4 or row["mn_tile_count"] < row["aic_num"]:
        confidence = "low"
    elif "unknown_compute_peak" in tags:
        confidence = "medium"
    if row.get("gmm_model_kind"):
        confidence = "low"
    return "|".join(tags), confidence


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
            if row.get("Type", "").lower() == "groupedmatmul":
                spec = infer_grouped_matmul_spec(input_shapes, output_shapes, input_formats, output_formats)
            else:
                spec = infer_matmul_spec(input_shapes, output_shapes, input_formats, output_formats)
            if spec is None and row.get("Type", "").lower() == "transposebatchmatmul":
                spec = infer_transpose_batch_matmul_spec(input_shapes, output_shapes, input_formats, output_formats)
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
            quant_spec = infer_quant_spec(row, spec, input_shapes)
            gmm_bounds = None
            if kernel_type.lower() == "groupedmatmul":
                gmm_bounds = estimate_grouped_matmul_bounds(
                    spec,
                    input_shapes,
                    input_dtypes,
                    dtype,
                    output_dtype,
                    quant_spec,
                    config,
                )
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
            tile = cost.tile
            true_flops = cost.flops
            duration_us = parse_float(row.get("Duration(us)"))
            achieved_tflops = true_flops / duration_us / 1_000_000.0 if duration_us > 0 else 0.0
            launch_us = cost.launch_overhead_us
            pipeline_eff = cost.pipeline_efficiency
            nd2nz_a, nd2nz_b = cost.nd2nz_a, cost.nd2nz_b
            format_overhead = cost.format_overhead_us
            quant_compute_us = cost.quant_compute_us
            quant_hbm_us = cost.quant_hbm_us
            quant_dequant_us = cost.quant_dequant_us
            quant_gm_bytes_min = cost.quant_gm_bytes_min
            quant_gm_bytes_tiled = cost.quant_gm_bytes_tiled
            compute_for_est = cost.flops_cost_us
            estimated_us = cost.total_us
            ideal_compute_us, ideal_hbm_us, ideal_lower_bound_us, ideal_gm_bytes_min = ideal_kernel_bounds(
                spec, dtype, output_dtype, config
            )
            best_kernel_us = launch_us + max(
                ideal_hbm_us,
                (ideal_compute_us / pipeline_eff) if ideal_compute_us is not None else ideal_hbm_us,
            )
            current_kernel_bound_us = tile.lower_bound_us
            actual_tiling_source, fallback_tiling_source, optimal_tiling_source, current_tiling_kind = tiling_source_split(
                tile.source
            )
            current_theoretical_tflops = (
                true_flops / current_kernel_bound_us / 1_000_000.0 if current_kernel_bound_us > 0 else 0.0
            )
            best_kernel_tflops = true_flops / best_kernel_us / 1_000_000.0 if best_kernel_us > 0 else 0.0
            ideal_tflops = true_flops / ideal_lower_bound_us / 1_000_000.0 if ideal_lower_bound_us > 0 else 0.0
            bottleneck = cost.dominant_component
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
            gmm_fields: dict[str, Any] = {}
            if gmm_bounds is not None:
                gmm_fields = gmm_bounds.to_report_fields()
                low = min(gmm_bounds.balanced.total_us, gmm_bounds.extreme.total_us)
                high = max(gmm_bounds.balanced.total_us, gmm_bounds.extreme.total_us)
                if duration_us <= 0:
                    position = ""
                elif duration_us < low:
                    position = "below_gmm_bounds"
                elif duration_us > high:
                    position = "above_gmm_bounds"
                else:
                    position = "within_gmm_bounds"
                gmm_fields.update(
                    {
                        "gmm_model_kind": "grouped_matmul_routing_bounds",
                        "gmm_duration_position": position,
                        "gmm_bounds_min_us": low,
                        "gmm_bounds_max_us": high,
                    }
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
                "actual_tiling_source": actual_tiling_source,
                "fallback_tiling_source": fallback_tiling_source,
                "optimal_tiling_source": optimal_tiling_source,
                "current_tiling_kind": current_tiling_kind,
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
                "effective_gm_bytes_min": cost.gm_bytes_min,
                "effective_gm_bytes_tiled": cost.gm_bytes_tiled,
                "compute_us": quant_compute_us if quant_spec.is_quant else tile.compute_us,
                "hbm_us": quant_hbm_us if quant_spec.is_quant and quant_hbm_us is not None else tile.hbm_us,
                "flops_cost_us": cost.flops_cost_us,
                "memory_access_us": cost.memory_access_us,
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
                "total_us": cost.total_us,
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
                "kernel_bound_type": cost.kernel_bound_type,
                "bound_type": cost.bound_type,
                "residual_us": duration_us - estimated_us if duration_us > 0 else None,
                "duration_over_estimate": duration_us / estimated_us if duration_us > 0 and estimated_us > 0 else None,
            }
            result.update(gmm_fields)
            tags, confidence = classify(result)
            result["diagnosis"] = tags
            result["confidence"] = confidence
            records.append(result)
    return records, unresolved


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
