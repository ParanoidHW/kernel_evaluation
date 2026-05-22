from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import AttentionSpec


@dataclass(frozen=True)
class AttentionTilingReplay:
    """Attention tiling/source classification carried into report rows.

    This is intentionally lighter than MatMul runtime KB replay. It records
    which ops-transformer source path and high-level strategy class are visible
    for a row, while keeping exact binary tiling unavailable until CANN host
    tiling contexts can be replayed.
    """

    actual_tiling_source: str
    fallback_tiling_source: str
    optimal_tiling_source: str
    current_tiling_kind: str
    tiling_strategy: str
    ops_transformer_source_file: str
    notes: str


OP_SOURCE_FILES = {
    "flash_attention": "attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp",
    "kv_quant_sparse_flash_attention": "attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp",
    "fused_infer_attention": "attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp",
    "prompt_attention": "attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp",
    "incremental_attention": "attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp",
    "paged_attention": "attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp",
}


def _ops_transformer_root(config: dict[str, Any]) -> Path | None:
    configured = config.get("attention_model", {}).get("ops_transformer_path")
    candidates = [Path(configured)] if configured else []
    candidates.extend([Path("ops-transformer-master"), Path("ops-transformer")])
    for candidate in candidates:
        if candidate and candidate.exists() and (candidate / "attention").is_dir():
            return candidate
    return None


def _variant_from_type_and_shape(kernel_type: str, spec: AttentionSpec) -> str:
    text = kernel_type.lower()
    if "kvquant" in text or "kv_quant" in text:
        return "kv_quant_sparse_flash_attention"
    if "paged" in text or "page" in text:
        return "paged_attention"
    if "incre" in text or spec.q_seq <= 1:
        return "incremental_attention"
    if "prompt" in text:
        return "prompt_attention"
    if "flashattentionscore" in text or text == "flash_attention_score":
        return "flash_attention"
    if "fusedinfer" in text:
        return "fused_infer_attention"
    return spec.variant


def replay_attention_tiling_strategy(
    spec: AttentionSpec,
    kernel_type: str,
    config: dict[str, Any],
) -> AttentionTilingReplay:
    """Replay source-visible attention strategy classes, not binary tiling data.

    The ops-transformer host tiling code is C++ and depends on CANN runtime
    context. This function deliberately exposes source-derived strategy labels
    and keeps exact tiling data separate until a full host-tiling replay exists.
    """

    root = _ops_transformer_root(config)
    variant = _variant_from_type_and_shape(kernel_type, spec)
    relative_source = OP_SOURCE_FILES.get(variant, "")
    source_file = ""
    has_source = False
    if root is not None and relative_source:
        path = root / relative_source
        if path.exists():
            has_source = True
            source_file = path.as_posix()

    strategy_tags: list[str] = [variant]
    if variant == "kv_quant_sparse_flash_attention":
        strategy_tags.extend(["kv_quant", "sparse", "mla_absorb_specialized"])
    if spec.q_seq <= 1:
        strategy_tags.append("decode")
    else:
        strategy_tags.append("prefill")
    if spec.q_seq != spec.kv_seq:
        strategy_tags.append("varlen_or_cross_attention")
    if spec.causal_or_masked:
        strategy_tags.append("mask_or_aux")
    if spec.kv_heads < spec.q_heads:
        strategy_tags.append("mqa_gqa")
    if spec.head_dim in {64, 128, 192, 256}:
        strategy_tags.append(f"d{spec.head_dim}")
    else:
        strategy_tags.append("custom_head_dim")

    if has_source:
        return AttentionTilingReplay(
            actual_tiling_source="ops_transformer_source_strategy_replay",
            fallback_tiling_source="",
            optimal_tiling_source="physical_lower_bound",
            current_tiling_kind="source_strategy_replay",
            tiling_strategy="|".join(strategy_tags),
            ops_transformer_source_file=source_file,
            notes="source_strategy_only:no_binary_tiling_data",
        )

    return AttentionTilingReplay(
        actual_tiling_source="unavailable_ops_transformer_replay",
        fallback_tiling_source="analytic_attention_bound",
        optimal_tiling_source="physical_lower_bound",
        current_tiling_kind="fallback_tiling",
        tiling_strategy="|".join(strategy_tags),
        ops_transformer_source_file=source_file,
        notes="ops_transformer_source_missing",
    )
