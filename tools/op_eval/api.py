from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import load_config
from .profiling import iter_input_files
from .types import ProfilingEvaluation


def estimate_op(op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch an operator cost request to the currently implemented model."""

    text = op_type.lower()
    if any(token in text for token in ("matmul", "mat_mul", "batchmatmul", "bmm")):
        from matmul_eval import estimate_matmul

        kwargs.setdefault("kernel_type", op_type)
        return estimate_matmul(*args, **kwargs)
    if "attention" in text or "attentionscore" in text:
        from attention_eval import estimate_attention

        kwargs.setdefault("kernel_type", op_type)
        return estimate_attention(*args, **kwargs)
    raise NotImplementedError(f"unsupported operator type: {op_type}")


def evaluate_profiling(
    profiling: str | Path | list[str | Path],
    *,
    op_kind: str = "matmul",
    config: dict[str, Any] | str | Path | None = None,
    config_path: str | Path | None = None,
    include_gmm: bool = False,
    include_allgather: bool = False,
) -> ProfilingEvaluation:
    """Evaluate profiling CSV files and return library-friendly result rows.

    This is the API counterpart of the CLI. It keeps file discovery, hardware
    config loading, and operator-family dispatch out of callers such as a
    whole-network evaluator.
    """

    if isinstance(profiling, (str, Path)):
        profiling_items = [profiling]
    else:
        profiling_items = profiling
    profiling_inputs = [str(item) for item in profiling_items]

    if isinstance(config, dict):
        resolved_config = config
    else:
        path = config_path or config
        if path is None:
            raise ValueError("evaluate_profiling requires either config or config_path")
        resolved_config = load_config(path)
    normalized_kind = op_kind.lower()
    if normalized_kind not in {"matmul", "grouped_matmul", "attention"}:
        raise NotImplementedError(f"unsupported op_kind: {op_kind}")

    if normalized_kind in {"matmul", "grouped_matmul"}:
        from matmul_eval.evaluator import evaluate_file
        from matmul_eval.runtime_kb import load_runtime_kb

        runtime_kb = load_runtime_kb(resolved_config)
        report = ProfilingEvaluation(op_kind=normalized_kind)
        for profiling_file in iter_input_files(profiling_inputs):
            rows, unresolved = evaluate_file(
                profiling_file,
                config=resolved_config,
                runtime_kb=runtime_kb,
                include_gmm=True if normalized_kind == "grouped_matmul" else include_gmm,
                include_allgather=include_allgather,
            )
            if normalized_kind == "grouped_matmul":
                rows = [row for row in rows if str(row.get("type", "")).lower() == "groupedmatmul"]
                unresolved = [row for row in unresolved if str(row.get("type", "")).lower() == "groupedmatmul"]
            report.rows.extend(rows)
            report.unresolved.extend(unresolved)
        return report

    from attention_eval.evaluator import evaluate_file

    report = ProfilingEvaluation(op_kind="attention")
    for profiling_file in iter_input_files(profiling_inputs):
        rows, unresolved = evaluate_file(profiling_file, config=resolved_config)
        report.rows.extend(rows)
        report.unresolved.extend(unresolved)
    return report


__all__ = ["ProfilingEvaluation", "estimate_op", "evaluate_profiling", "load_config"]
