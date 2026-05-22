#!/usr/bin/env python3
"""Summarize evaluation gaps for non-launch-dominated, highly occupied kernels."""

from __future__ import annotations

import argparse
import csv
import collections
import statistics
from pathlib import Path


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else default
    except ValueError:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(as_float(row, key, float(default)))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def rel_error(row: dict[str, str]) -> float | None:
    duration = as_float(row, "duration_us")
    estimate = as_float(row, "estimated_us")
    if duration <= 0 or estimate <= 0:
        return None
    return abs(estimate - duration) / duration


def gmm_bound_rel_error(row: dict[str, str]) -> float | None:
    """Return zero inside GMM routing bounds, otherwise distance to nearest bound."""

    duration = as_float(row, "duration_us")
    low = as_float(row, "gmm_bounds_min_us")
    high = as_float(row, "gmm_bounds_max_us")
    if duration <= 0 or low <= 0 or high <= 0:
        return None
    if low <= duration <= high:
        return 0.0
    nearest = low if duration < low else high
    return abs(nearest - duration) / duration


def max_block_dim(row: dict[str, str]) -> int:
    return max(as_int(row, "block_dim"), as_int(row, "mix_block_dim"))


def is_large_occupied(row: dict[str, str], duration_floor_us: float, block_ratio: float, cube_util_floor: float) -> bool:
    duration = as_float(row, "duration_us")
    if duration < duration_floor_us:
        return False
    aic_num = max(1, as_int(row, "aic_num", 1))
    block_ok = max_block_dim(row) >= aic_num * block_ratio
    cube_util = as_float(row, "cube_utilization_pct")
    cube_ok = cube_util >= cube_util_floor
    return block_ok or cube_ok


def compact_row(row: dict[str, str], rel: float) -> str:
    keys = [
        "file",
        "line",
        "type",
        "duration_us",
        "estimated_us",
        "duration_over_estimate",
        "ideal_lower_bound_us",
        "m",
        "n",
        "k",
        "batch",
        "q_seq",
        "kv_seq",
        "head_dim",
        "block_dim",
        "mix_block_dim",
        "aic_num",
        "cube_utilization_pct",
        "kernel_tiling_source",
        "current_tiling_kind",
        "diagnosis",
    ]
    parts = [f"rel={rel:.4f}"]
    parts.extend(f"{key}={row.get(key, '')}" for key in keys if key in row)
    return " ".join(parts)


def summarize(path: Path, args: argparse.Namespace) -> None:
    rows = list(csv.DictReader(path.open(newline="")))
    evaluated: list[tuple[float, dict[str, str]]] = []
    large: list[tuple[float, dict[str, str]]] = []
    lower_bound_violations = 0
    for row in rows:
        rel = rel_error(row)
        if rel is None:
            continue
        evaluated.append((rel, row))
        if as_float(row, "ideal_lower_bound_us") > as_float(row, "duration_us") > 0:
            lower_bound_violations += 1
        if is_large_occupied(row, args.duration_floor_us, args.block_ratio, args.cube_util_floor):
            large.append((rel, row))

    evaluated.sort(key=lambda item: item[0], reverse=True)
    large.sort(key=lambda item: item[0], reverse=True)

    print(f"\n{path}")
    print(f"rows={len(rows)} evaluated={len(evaluated)} large_occupied={len(large)}")
    print(f"filters=duration_us>={args.duration_floor_us:g}, block_dim>=aic_num*{args.block_ratio:g} OR cube_util>={args.cube_util_floor:g}")
    print(f"ideal_lower_bound_violations_all={lower_bound_violations}")
    if not large:
        return
    rels = [item[0] for item in large]
    ratios = [as_float(row, "duration_over_estimate") or as_float(row, "duration_us") / as_float(row, "estimated_us") for _, row in large]
    print(
        "large_occupied_relative_error "
        f"max={rels[0]:.4f} p95={percentile(rels, 0.95):.4f} "
        f"p90={percentile(rels, 0.90):.4f} median={statistics.median(rels):.4f}"
    )
    print(f"large_occupied_duration_over_estimate median={statistics.median(ratios):.4f}")
    by_type: dict[str, list[float]] = {}
    for rel, row in large:
        by_type.setdefault(row.get("type", ""), []).append(rel)
    print("by_type:")
    for op_type, values in sorted(by_type.items(), key=lambda item: (len(item[1]), item[0]), reverse=True):
        print(
            f"  {op_type or '<unknown>'}: n={len(values)} "
            f"max={max(values):.4f} p95={percentile(values, 0.95):.4f} median={statistics.median(values):.4f}"
        )
    gmm_items = [(gmm_bound_rel_error(row), row) for _, row in large if row.get("gmm_model_kind")]
    gmm_items = [(rel, row) for rel, row in gmm_items if rel is not None]
    if gmm_items:
        positions = collections.Counter(row.get("gmm_duration_position", "") for _, row in gmm_items)
        gmm_rels = [rel for rel, _ in gmm_items]
        print(
            "gmm_routing_bound_error "
            f"n={len(gmm_rels)} max={max(gmm_rels):.4f} p95={percentile(gmm_rels, 0.95):.4f} "
            f"median={statistics.median(gmm_rels):.4f} positions={dict(sorted(positions.items()))}"
        )
    print("top_large_occupied_tail:")
    for rel, row in large[: args.top]:
        print(f"  {compact_row(row, rel)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--duration-floor-us", type=float, default=10.0)
    parser.add_argument("--block-ratio", type=float, default=0.8)
    parser.add_argument("--cube-util-floor", type=float, default=70.0)
    parser.add_argument("--top", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for report in args.reports:
        summarize(report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
