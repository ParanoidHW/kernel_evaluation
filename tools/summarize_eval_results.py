#!/usr/bin/env python3
"""Write traceable CSV summaries for local evaluation reports."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from analyze_large_shape_gap import as_float, gmm_bound_rel_error, is_large_occupied, percentile, rel_error


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summarize_report(path: Path, evaluated_at: str, commit: str, args: argparse.Namespace) -> dict[str, object]:
    rows = list(csv.DictReader(path.open(newline="")))
    evaluated: list[tuple[float, dict[str, str]]] = []
    large: list[tuple[float, dict[str, str]]] = []
    lower_bound_violations = 0
    gmm_positions: Counter[str] = Counter()
    gmm_rels: list[float] = []

    for row in rows:
        rel = rel_error(row)
        if rel is None:
            continue
        evaluated.append((rel, row))
        if as_float(row, "ideal_lower_bound_us") > as_float(row, "duration_us") > 0:
            lower_bound_violations += 1
        if is_large_occupied(row, args.duration_floor_us, args.block_ratio, args.cube_util_floor):
            large.append((rel, row))
            if row.get("gmm_model_kind"):
                gmm_positions[row.get("gmm_duration_position", "")] += 1
                gmm_rel = gmm_bound_rel_error(row)
                if gmm_rel is not None:
                    gmm_rels.append(gmm_rel)

    large.sort(key=lambda item: item[0], reverse=True)
    rels = [rel for rel, _ in large]
    top = large[0][1] if large else {}
    by_type = Counter(row.get("type", "") for _, row in large)
    top_type = by_type.most_common(1)[0][0] if by_type else ""

    return {
        "evaluated_at_utc": evaluated_at,
        "commit": commit,
        "report": str(path),
        "rows": len(rows),
        "evaluated": len(evaluated),
        "large_occupied": len(large),
        "duration_floor_us": args.duration_floor_us,
        "block_ratio": args.block_ratio,
        "cube_util_floor": args.cube_util_floor,
        "large_rel_max": max(rels) if rels else "",
        "large_rel_p95": percentile(rels, 0.95) if rels else "",
        "large_rel_p90": percentile(rels, 0.90) if rels else "",
        "large_rel_median": statistics.median(rels) if rels else "",
        "lower_bound_violations_all": lower_bound_violations,
        "top_type": top_type,
        "top_line": top.get("line", ""),
        "top_op_type": top.get("type", ""),
        "top_duration_us": top.get("duration_us", ""),
        "top_estimated_us": top.get("estimated_us", ""),
        "top_gmm_bounds_min_us": top.get("gmm_bounds_min_us", ""),
        "top_gmm_bounds_max_us": top.get("gmm_bounds_max_us", ""),
        "top_diagnosis": top.get("diagnosis", ""),
        "gmm_bound_max": max(gmm_rels) if gmm_rels else "",
        "gmm_bound_p95": percentile(gmm_rels, 0.95) if gmm_rels else "",
        "gmm_bound_median": statistics.median(gmm_rels) if gmm_rels else "",
        "gmm_positions": ";".join(f"{key}:{value}" for key, value in sorted(gmm_positions.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--duration-floor-us", type=float, default=10.0)
    parser.add_argument("--block-ratio", type=float, default=0.8)
    parser.add_argument("--cube-util-floor", type=float, default=70.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluated_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit()
    rows = [summarize_report(report, evaluated_at, commit, args) for report in args.reports]
    run_dir = args.output_dir / f"{evaluated_at}_{commit}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "eval_summary.csv"
    metadata = run_dir / "metadata.txt"
    latest = args.output_dir / "LATEST"
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata.write_text(
        f"evaluated_at_utc={evaluated_at}\ncommit={commit}\nreports={len(rows)}\n",
        encoding="utf-8",
    )
    latest.write_text(f"{run_dir.name}\n", encoding="utf-8")
    print(f"wrote_summary={output}")
    print(f"wrote_latest={latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
