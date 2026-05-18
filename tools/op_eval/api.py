from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import load_config


def estimate_op(op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch an operator cost request to the currently implemented model."""

    text = op_type.lower()
    if any(token in text for token in ("matmul", "mat_mul", "batchmatmul", "bmm")):
        from matmul_eval import estimate_matmul

        kwargs.setdefault("kernel_type", op_type)
        return estimate_matmul(*args, **kwargs)
    raise NotImplementedError(f"unsupported operator type: {op_type}")


__all__ = ["estimate_op", "load_config"]
