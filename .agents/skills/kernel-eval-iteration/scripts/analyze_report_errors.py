#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
import sys
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


def has_value(row: dict[str, str], key: str) -> bool:
    return row.get(key, "") not in {"", None}


def classify_tail(row: dict[str, str], rel: float) -> list[str]:
    tags: list[str] = []
    duration = as_float(row, "duration_us")
    estimate = as_float(row, "estimated_us")
    ideal = as_float(row, "ideal_lower_bound_us")
    ratio = as_float(row, "duration_over_estimate")
    if not ratio and duration > 0 and estimate > 0:
        ratio = duration / estimate

    if ideal > duration > 0:
        tags.append("min_bound_violation")
    if ratio and ratio < 0.5:
        tags.append("overestimate")
    elif ratio > 2.0:
        tags.append("underestimate")
    if rel >= 0.5:
        tags.append("large_tail")

    if has_value(row, "q_seq") and has_value(row, "kv_seq"):
        q_seq = as_int(row, "q_seq")
        kv_seq = as_int(row, "kv_seq")
        if q_seq <= 1:
            tags.append("decode")
        elif q_seq <= 512 and kv_seq <= 512:
            tags.append("short_prefill")
        elif q_seq >= 1024 or kv_seq >= 1024:
            tags.append("long_prefill")
        if as_int(row, "aux_elements") > 0 and ideal > duration > 0:
            tags.append("optional_input_overcount")

    if has_value(row, "M") and has_value(row, "N") and has_value(row, "K"):
        m = as_int(row, "M")
        n = as_int(row, "N")
        k = as_int(row, "K")
        if m <= 4:
            tags.append("small_m")
        if k >= max(m, n, 1) * 4:
            tags.append("large_k")
        if as_float(row, "storage_padding_ratio", 1.0) > 1.05:
            tags.append("layout_padding")

    diagnosis = row.get("diagnosis", "")
    for marker in ("runtime_nd2nz", "weight_nz", "stream_k", "quant_matmul", "low_tile_count"):
        if marker in diagnosis:
            tags.append(marker)

    return tags or ["unclassified"]


def summarize(path: Path) -> None:
    rows = list(csv.DictReader(path.open(newline="")))
    scored: list[tuple[float, dict[str, str]]] = []
    ratios: list[float] = []
    for row in rows:
        duration = as_float(row, "duration_us")
        estimate = as_float(row, "estimated_us")
        if duration <= 0 or estimate <= 0:
            continue
        rel = abs(estimate - duration) / duration
        scored.append((rel, row))
        ratio = as_float(row, "duration_over_estimate")
        ratios.append(ratio if ratio else duration / estimate)

    scored.sort(key=lambda item: item[0], reverse=True)
    rels = [item[0] for item in scored]
    print(f"\n{path}")
    print(f"rows={len(rows)} evaluated={len(scored)}")
    if not scored:
        return
    print(
        "relative_error "
        f"max={rels[0]:.4f} p95={percentile(rels, 0.95):.4f} "
        f"p90={percentile(rels, 0.90):.4f} median={statistics.median(rels):.4f}"
    )
    print(f"duration_over_estimate median={statistics.median(ratios):.4f}")
    print("top_tail:")
    keys = [
        "file",
        "line",
        "type",
        "duration_us",
        "estimated_us",
        "duration_over_estimate",
        "ideal_lower_bound_us",
        "M",
        "N",
        "K",
        "batch",
        "q_seq",
        "kv_seq",
        "head_dim",
        "q_heads",
        "kv_heads",
        "kernel_tiling_source",
        "current_tiling_kind",
        "tiling_strategy",
        "diagnosis",
    ]
    for rel, row in scored[:12]:
        compact = " ".join(f"{key}={row.get(key, '')}" for key in keys if key in row)
        print(f"  rel={rel:.4f} tags={','.join(classify_tail(row, rel))} {compact}")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: analyze_report_errors.py REPORT.csv [REPORT.csv ...]", file=sys.stderr)
        return 2
    for arg in argv:
        summarize(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
