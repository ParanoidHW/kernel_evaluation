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
    """Logical attention problem reconstructed from profiling shapes.

    The evaluator normalizes many CANN attention variants into Q/K/V sequence,
    head and element counts so the cost model can reason about QK/PV compute,
    softmax/vector work, auxiliary metadata, output size and MQA/GQA behavior.
    """

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
    """Infer an AttentionSpec from the first Q/K/V inputs of a profiling row.

    The profiling CSV does not expose a single canonical layout for all
    attention kernels. This parser recognizes common `[B,H,S,D]`, `[B,S,D]`,
    `[S,D]` and vector-like encodings, caps oversized static mask/aux tensors
    to the active score window, and records a variant label from the kernel
    type. Rows that cannot provide Q/K/V shape semantics return `None` and are
    emitted to the unresolved report.
    """

    if len(input_shapes) < 3:
        return None
    variant = _variant_from_type(kernel_type)
    q_shape, k_shape, v_shape = input_shapes[0], input_shapes[1], input_shapes[2]
    q_dims = _positive_dims(q_shape)
    k_dims = _positive_dims(k_shape)
    v_dims = _positive_dims(v_shape)
    if not q_dims or not k_dims or not v_dims:
        return None

    if variant == "kv_quant_sparse_flash_attention":
        return _infer_kv_quant_sparse_attention_spec(input_shapes, output_shapes, q_dims, k_dims, v_dims)
    if variant == "fused_infer_attention":
        pa_spec = _infer_fused_infer_pa_attention_spec(input_shapes, output_shapes, q_dims, k_dims, v_dims)
        if pa_spec is not None:
            return pa_spec

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
        variant=variant,
        causal_or_masked=causal_or_masked,
    )


def _infer_fused_infer_pa_attention_spec(
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    q_dims: list[int],
    k_dims: list[int],
    v_dims: list[int],
) -> AttentionSpec | None:
    """Infer FIA paged-attention shapes from KV-cache storage layout.

    FusedInferAttentionScore PA uses K/V cache shapes such as
    `[blockNum, KV_N, blockSize, D]` and a block table `[B, maxBlockNum]`.
    The cache's first dimension is storage capacity, not logical batch.
    Treating it as batch makes the generic parser charge the whole cache.
    """

    if len(q_dims) < 3 or len(k_dims) not in {4, 5} or len(v_dims) not in {4, 5}:
        return None
    if len(input_shapes) <= 3:
        return None

    head_dim = q_dims[-1]
    q_info = _choose_sequence_dim(q_dims, head_dim)
    if q_info is None:
        return None
    q_batch, q_heads, q_seq, q_layout = q_info

    block_table_dims: list[int] = []
    for shape in input_shapes[3:]:
        dims = _positive_dims(shape)
        if len(dims) == 2 and dims[0] == q_batch:
            block_table_dims = dims
            break
    if len(block_table_dims) != 2:
        return None

    if len(k_dims) == 5:
        kv_heads = k_dims[1]
        block_size = k_dims[3]
        kv_head_dim = k_dims[2] * k_dims[4]
        layout = f"{q_layout}|pa_nz_cache"
    else:
        kv_heads = k_dims[1]
        block_size = k_dims[2]
        kv_head_dim = k_dims[3]
        layout = f"{q_layout}|pa_b_n_bs_d"
    if len(v_dims) == 5:
        value_dim = v_dims[2] * v_dims[4]
    else:
        value_dim = v_dims[-1]
    if kv_heads <= 0 or block_size <= 0 or kv_head_dim <= 0 or value_dim <= 0:
        return None

    table_batch, max_blocks_per_batch = block_table_dims
    batch = max(1, q_batch)
    if table_batch > 0:
        batch = table_batch
    kv_seq = max(1, max_blocks_per_batch * block_size)
    active_kv_elements = batch * kv_heads * kv_seq * kv_head_dim
    active_v_elements = batch * kv_heads * kv_seq * value_dim
    output_elements = _shape_elements(output_shapes[0]) if output_shapes else batch * q_heads * q_seq * value_dim
    score_elements = batch * q_heads * q_seq * kv_seq

    raw_aux_elements = sum(_shape_elements(shape) for shape in input_shapes[3:])
    aux_elements = 0
    for index, shape in enumerate(input_shapes[3:], start=3):
        elements = _shape_elements(shape)
        if elements <= 0:
            continue
        if index == 4:
            aux_elements += elements
            continue
        if len(_positive_dims(shape)) >= 2 and elements > score_elements:
            aux_elements += score_elements
        else:
            aux_elements += elements

    return AttentionSpec(
        batch=batch,
        q_heads=max(1, q_heads),
        kv_heads=max(1, kv_heads),
        q_seq=max(1, q_seq),
        kv_seq=kv_seq,
        head_dim=max(1, head_dim),
        value_dim=max(1, value_dim),
        output_elements=output_elements,
        q_elements=_shape_elements(q_dims),
        k_elements=active_kv_elements,
        v_elements=active_v_elements,
        aux_elements=aux_elements,
        raw_aux_elements=raw_aux_elements,
        score_elements=score_elements,
        layout=layout,
        variant="fused_infer_attention",
        causal_or_masked=aux_elements > 0,
    )


def _infer_kv_quant_sparse_attention_spec(
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    q_dims: list[int],
    k_dims: list[int],
    v_dims: list[int],
) -> AttentionSpec | None:
    """Infer QSFA MLA/PA shapes from source-visible layout conventions.

    `KvQuantSparseFlashAttention` accepts query in TND/BSND and key/value in
    BSND/TND/PA_BSND. The ds3.2 sample uses a PA-like key cache
    `[block_num, N, block_size, D]` and query/output `[T, N, D]`; treating the
    first key dimension as batch makes the generic parser count the whole cache
    as 520 independent batches. The host tiling derives `s2Size` from
    `block_table.dim1 * block_size` for PA and `gSize = n1 / n2`, so this
    parser keeps query token/head axes separate from cache storage axes.
    """

    if len(q_dims) < 3 or len(output_shapes) == 0:
        return None
    out_dims = _positive_dims(output_shapes[0])
    if len(out_dims) < 2:
        return None

    q_seq = q_dims[0]
    q_heads = q_dims[-2]
    head_dim = q_dims[-1]
    value_dim = out_dims[-1]
    output_elements = _shape_elements(output_shapes[0])

    kv_heads = 1
    block_size = 1
    kv_seq = 1
    layout = "t_n_d|pa_cache"
    if len(k_dims) >= 4:
        total_blocks = k_dims[0]
        if k_dims[1] == q_heads:
            kv_heads = k_dims[1]
            block_size = k_dims[2]
            layout = "t_n_d|pa_b_n_bs_d"
        else:
            block_size = k_dims[1]
            kv_heads = k_dims[2]
            layout = "t_n_d|pa_b_bs_n_d"
        block_table_dims = _positive_dims(input_shapes[4]) if len(input_shapes) > 4 else []
        max_blocks_per_batch = block_table_dims[1] if len(block_table_dims) >= 2 else total_blocks
        kv_seq = max(1, max_blocks_per_batch * max(1, block_size))
    else:
        k_info = _choose_sequence_dim(k_dims, k_dims[-1])
        if k_info is None:
            return None
        _, kv_heads, kv_seq, k_layout = k_info
        layout = f"t_n_d|{k_layout}"

    score_elements = max(1, q_seq * q_heads * kv_seq)
    raw_aux_elements = sum(_shape_elements(shape) for shape in input_shapes[3:])
    aux_elements = 0
    for shape in input_shapes[3:]:
        elements = _shape_elements(shape)
        if elements <= 0:
            continue
        if len(_positive_dims(shape)) >= 2 and elements > score_elements:
            aux_elements += score_elements
        else:
            aux_elements += elements

    return AttentionSpec(
        batch=1,
        q_heads=max(1, q_heads),
        kv_heads=max(1, kv_heads),
        q_seq=max(1, q_seq),
        kv_seq=max(1, kv_seq),
        head_dim=max(1, head_dim),
        value_dim=max(1, value_dim),
        output_elements=output_elements,
        q_elements=_shape_elements(q_dims),
        k_elements=_shape_elements(k_dims),
        v_elements=_shape_elements(v_dims),
        aux_elements=aux_elements,
        raw_aux_elements=raw_aux_elements,
        score_elements=score_elements,
        layout=layout,
        variant="kv_quant_sparse_flash_attention",
        causal_or_masked=aux_elements > 0,
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
