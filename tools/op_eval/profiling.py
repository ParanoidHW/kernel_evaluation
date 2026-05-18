from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


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


def is_excluded_by_default(row: dict[str, str], include_gmm: bool, include_allgather: bool) -> bool:
    text = f"{row.get('Name', '')} {row.get('Type', '')}".lower()
    if not include_gmm and "groupedmatmul" in text:
        return True
    if not include_allgather and "allgathermatmul" in text:
        return True
    return False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
