from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import evaluate_profiling
from .profiling import write_csv


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate operator profiling CSV files.")
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
    parser.add_argument(
        "--op-kind",
        default="matmul",
        choices=["matmul", "attention"],
        help="Operator family to evaluate. Default: matmul.",
    )
    parser.add_argument("--output", help="Write detailed resolved report CSV.")
    parser.add_argument("--unresolved-output", help="Write unresolved rows CSV.")
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
    report = evaluate_profiling(
        args.profiling,
        op_kind=args.op_kind,
        config_path=args.config,
        include_gmm=args.include_gmm,
        include_allgather=args.include_allgather,
    )
    rows = report.rows
    unresolved = report.unresolved

    if args.op_kind == "attention":
        from attention_eval.evaluator import print_summary
    else:
        from matmul_eval.evaluator import print_summary

    print_summary(rows, unresolved)

    suggestions: dict[str, Any] | None = None
    if args.suggest_calibration or args.calibration_output:
        if args.op_kind != "matmul":
            raise NotImplementedError("--suggest-calibration is currently implemented only for matmul")
        from matmul_eval.evaluator import print_calibration_suggestions

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


def run_cli() -> int:
    """Console-script compatible entry point."""

    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(run_cli())
