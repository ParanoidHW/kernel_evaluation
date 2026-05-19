from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfilingEvaluation:
    """Resolved and unresolved rows from profiling CSV evaluation."""

    op_kind: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)

    @property
    def resolved_count(self) -> int:
        return len(self.rows)

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved)

    def extend(self, other: "ProfilingEvaluation") -> None:
        if other.op_kind != self.op_kind:
            raise ValueError(f"cannot merge {other.op_kind} report into {self.op_kind} report")
        self.rows.extend(other.rows)
        self.unresolved.extend(other.unresolved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_kind": self.op_kind,
            "resolved_count": self.resolved_count,
            "unresolved_count": self.unresolved_count,
            "rows": self.rows,
            "unresolved": self.unresolved,
        }


__all__ = ["ProfilingEvaluation"]
