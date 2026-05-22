from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from op_eval.common import num_elements


ATTENTION_TYPE_TOKENS = (
    "attention",
    "flashattention",
    "flash_attention",
    "fusedinferattentionscore",
    "fusedinferattention",
    "pagedattention",
    "paged_attention",
    "promptflashattention",
    "prompt_flash_attention",
    "increflashattention",
    "incre_flash_attention",
)


@dataclass(frozen=True)
class AttentionSpec:
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    head_dim: int
    value_dim: int
    output_elements: int
    q_elements: int
    k_elements: int
    v_elements: int
    aux_elements: int
    raw_aux_elements: int
    score_elements: int
    layout: str
    variant: str
    causal_or_masked: bool


def is_attention_row(row: dict[str, str]) -> bool:
    # Use Type, not Name, to avoid pulling helper ops from attention scopes
    # such as Slice/Cast whose names contain an attention op prefix.
    text = row.get("Type", "").lower()
    return any(token in text for token in ATTENTION_TYPE_TOKENS)


def _positive_dims(shape: list[int]) -> list[int]:
    return [dim for dim in shape if dim > 0]


def _shape_elements(shape: list[int]) -> int:
    return num_elements(_positive_dims(shape))


def _choose_sequence_dim(shape: list[int], head_dim: int) -> tuple[int, int, int, str] | None:
    dims = _positive_dims(shape)
    if not dims:
        return None
    if len(dims) == 1:
        return 1, 1, dims[0], "vector"
    if len(dims) == 2:
        return 1, 1, dims[0], "s_d"
    if len(dims) == 3:
        return dims[0], 1, dims[1], "b_s_d"
    if len(dims) >= 4:
        if dims[-1] == head_dim:
            return dims[0], dims[1], dims[-2], "b_h_s_d"
        return dims[0], dims[1], dims[2], "b_h_s_d_assumed"
    return None


def _variant_from_type(kernel_type: str) -> str:
    text = kernel_type.lower()
    if "kvquant" in text or "kv_quant" in text:
        return "kv_quant_sparse_flash_attention"
    if "paged" in text:
        return "paged_attention"
    if "incre" in text or "incremental" in text:
        return "incremental_attention"
    if "prompt" in text:
        return "prompt_attention"
    if "flash" in text:
        return "flash_attention"
    if "infer" in text:
        return "fused_infer_attention"
    return "attention"


def infer_attention_spec(
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    *,
    kernel_type: str,
) -> AttentionSpec | None:
    if len(input_shapes) < 3:
        return None
    q_shape, k_shape, v_shape = input_shapes[0], input_shapes[1], input_shapes[2]
    q_dims = _positive_dims(q_shape)
    k_dims = _positive_dims(k_shape)
    v_dims = _positive_dims(v_shape)
    if not q_dims or not k_dims or not v_dims:
        return None

    head_dim = q_dims[-1]
    value_dim = v_dims[-1]
    q_info = _choose_sequence_dim(q_shape, head_dim)
    k_info = _choose_sequence_dim(k_shape, k_dims[-1])
    if q_info is None or k_info is None:
        return None
    q_batch, q_heads, q_seq, q_layout = q_info
    k_batch, kv_heads, kv_seq, k_layout = k_info

    output_elements = _shape_elements(output_shapes[0]) if output_shapes else q_batch * q_heads * q_seq * value_dim
    score_elements = q_batch * q_heads * q_seq * kv_seq
    raw_aux_elements = sum(_shape_elements(shape) for shape in input_shapes[3:])
    aux_elements = 0
    for shape in input_shapes[3:]:
        elements = _shape_elements(shape)
        if elements <= 0:
            continue
        # Large mask tensors in profiling often keep their static max shape
        # (for example 2048x2048) while the selected attention tile only
        # touches the active q_seq x kv_seq window.
        if len(_positive_dims(shape)) >= 2 and elements > score_elements:
            aux_elements += score_elements
        else:
            aux_elements += elements
    if q_batch != k_batch:
        batch = max(q_batch, k_batch)
    else:
        batch = q_batch
    causal_or_masked = aux_elements > 0
    return AttentionSpec(
        batch=max(1, batch),
        q_heads=max(1, q_heads),
        kv_heads=max(1, kv_heads),
        q_seq=max(1, q_seq),
        kv_seq=max(1, kv_seq),
        head_dim=max(1, head_dim),
        value_dim=max(1, value_dim),
        output_elements=output_elements,
        q_elements=_shape_elements(q_shape),
        k_elements=_shape_elements(k_shape),
        v_elements=_shape_elements(v_shape),
        aux_elements=aux_elements,
        raw_aux_elements=raw_aux_elements,
        score_elements=score_elements,
        layout=f"{q_layout}|{k_layout}",
        variant=_variant_from_type(kernel_type),
        causal_or_masked=causal_or_masked,
    )


def input_dtypes_from_row(row: dict[str, Any]) -> list[str]:
    value = row.get("Input Data Types", "")
    if not value or value == "N/A":
        return []
    return [part.strip() for part in str(value).replace('"', "").split(";") if part.strip()]


def output_dtype_from_row(row: dict[str, Any]) -> str:
    value = row.get("Output Data Types", "")
    if not value or value == "N/A":
        dtypes = input_dtypes_from_row(row)
        return dtypes[0] if dtypes else "UNKNOWN"
    parts = [part.strip() for part in str(value).replace('"', "").split(";") if part.strip()]
    return parts[0] if parts else "UNKNOWN"
