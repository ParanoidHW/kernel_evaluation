"""Ascend attention-family cost evaluator public API and internals."""

from .api import AttentionCostEstimate, estimate_attention, estimate_attention_cost
from .common import AttentionSpec

__all__ = [
    "AttentionCostEstimate",
    "AttentionSpec",
    "estimate_attention",
    "estimate_attention_cost",
]

