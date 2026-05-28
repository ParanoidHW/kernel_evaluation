from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from op_eval.common import ceil_div, dtype_size, num_elements

from .common import (
    OtherOpSpec,
    aligned_vector_chunks,
    count_present_output_shapes,
    dynamic_quant_scale_elements,
    infer_add_rms_norm_dims,
    infer_dynamic_quant_rows,
    last_dim,
)


@dataclass(frozen=True)
class OtherOpCostEstimate:
    vector_compute_us: float
    cube_compute_us: float
    hbm_us: float
    layout_overhead_us: float
    workspace_us: float
    sync_overhead_us: float
    launch_overhead_us: float
    bounds_min_us: float
    bounds_max_us: float
    optimal_tiling_us: float
    source_schedule_bound_us: float
    total_us: float
    ideal_lower_bound_us: float
    current_kernel_bound_us: float
    tiling_source: str
    forward_eval_mode: str
    bounds_reason: str
    dominant_component: str


def _other_ops_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("other_ops_model", {})
    return {
        "vector_bandwidth_tbps": float(model.get("vector_bandwidth_tbps", config.get("hbm_bandwidth_tbps", 1.0))),
        "vector_gops": float(model.get("vector_gops", 4000.0)),
        "launch_overhead_us": float(model.get("launch_overhead_us", 0.0)),
        "layout_strided_factor": float(model.get("layout_strided_factor", 1.0)),
        "reduction_passes": float(model.get("reduction_passes", 2.0)),
        "softmax_passes": float(model.get("softmax_passes", 4.0)),
        "norm_passes": float(model.get("norm_passes", 3.0)),
        "activation_op_factor": float(model.get("activation_op_factor", 4.0)),
        "transcendental_op_factor": float(model.get("transcendental_op_factor", 16.0)),
        "index_random_access_factor": float(model.get("index_random_access_factor", 1.5)),
        "dynamic_quant_passes": float(model.get("dynamic_quant_passes", 2.0)),
        "dynamic_quant_op_factor": float(model.get("dynamic_quant_op_factor", 9.0)),
        "dynamic_quant_min_row_us": float(model.get("dynamic_quant_min_row_us", 0.0)),
        "add_rms_norm_dynamic_quant_passes": float(model.get("add_rms_norm_dynamic_quant_passes", 4.0)),
        "add_rms_norm_dynamic_quant_op_factor": float(model.get("add_rms_norm_dynamic_quant_op_factor", 14.0)),
        "mla_prolog_norm_rope_op_factor": float(model.get("mla_prolog_norm_rope_op_factor", 18.0)),
        "mla_prolog_sync_points": float(model.get("mla_prolog_sync_points", 6.0)),
        "mla_prolog_sync_us": float(model.get("mla_prolog_sync_us", 0.0)),
        "quant_lightning_indexer_topk_op_factor": float(model.get("quant_lightning_indexer_topk_op_factor", 32.0)),
        "quant_lightning_indexer_sync_points": float(model.get("quant_lightning_indexer_sync_points", 4.0)),
        "quant_lightning_indexer_sync_us": float(model.get("quant_lightning_indexer_sync_us", 0.0)),
    }


def _bytes_to_us(num_bytes: float, bandwidth_tbps: float) -> float:
    return num_bytes / max(bandwidth_tbps * 1_000_000.0, 1e-9)


def _ops_to_us(num_ops: float, vector_gops: float) -> float:
    return num_ops / max(vector_gops * 1_000.0, 1e-9)


def _is_activation_only(op_type: str) -> bool:
    return op_type.replace("_", "").lower() in {
        "swish",
        "gelu",
        "swiglu",
        "gegluv2",
        "dequantswigluquant",
    }


def _elementwise_op_factor(spec: OtherOpSpec, cfg: dict[str, Any]) -> float:
    normalized = spec.op_type.replace("_", "").replace("-", "").lower()
    if normalized in {"zeroslike", "oneslike", "fill"}:
        return 0.5
    if normalized == "range":
        return 1.0
    if normalized in {"rotarypositionembedding", "rotarymul"}:
        return 6.0
    if normalized in {"cos", "sin"}:
        return cfg["transcendental_op_factor"]
    if normalized == "rsqrt":
        return max(cfg["transcendental_op_factor"], 8.0)
    if normalized in {"sigmoid"}:
        return cfg["activation_op_factor"]
    if normalized in {"pows", "pow"}:
        return max(cfg["transcendental_op_factor"], 8.0)
    if normalized in {"realdiv"}:
        return 4.0
    if normalized in {"floordiv", "floormod"}:
        return 5.0
    if normalized in {"greaterequal", "greater", "less", "equal", "maximum", "logicalnot", "selectv2", "clipbyvaluev2"}:
        return 2.0
    return 1.0


def _reduction_passes(spec: OtherOpSpec, cfg: dict[str, Any]) -> float:
    normalized = spec.op_type.replace("_", "").replace("-", "").lower()
    if normalized == "softmaxv2":
        return cfg["softmax_passes"]
    if normalized == "reducemean":
        return cfg["reduction_passes"] + 1.0
    if normalized == "cumsum":
        return cfg["reduction_passes"] + 1.0
    if normalized == "argmaxwithvalue":
        return cfg["reduction_passes"] + 1.0
    return cfg["reduction_passes"]


def _norm_activation_passes(spec: OtherOpSpec, cfg: dict[str, Any]) -> float:
    normalized = spec.op_type.replace("_", "").replace("-", "").lower()
    if normalized in {"swish", "gelu"}:
        return 1.0
    if normalized in {"swiglu", "gegluv2", "dequantswigluquant"}:
        return 1.5
    if normalized in {"add_rmsnorm", "addrmsnorm", "inplaceaddrmsnorm", "addrmsnormcast"}:
        return max(cfg["norm_passes"], 4.0)
    if normalized == "layernormv3":
        return max(cfg["norm_passes"], 4.0)
    if normalized == "groupnormsilu":
        return max(cfg["norm_passes"], 4.0)
    return cfg["norm_passes"]


def _estimate_dynamic_quant(spec: OtherOpSpec, cfg: dict[str, Any]) -> tuple[float, float, float]:
    rows, hidden = infer_dynamic_quant_rows(spec.input_shapes, spec.output_shapes)
    data_elements = spec.input_elements[0] if spec.input_elements else spec.logical_elements
    scale_elements = dynamic_quant_scale_elements(spec.output_shapes) or rows
    has_smooth = len(spec.input_shapes) > 1 and last_dim(spec.input_shapes[1]) == hidden
    y_bytes = spec.output_bytes[0] if spec.output_bytes else data_elements
    scale_bytes = scale_elements * dtype_size("FLOAT")
    x_bytes = spec.input_bytes[0] if spec.input_bytes else data_elements * 2
    smooth_bytes = rows * hidden * dtype_size(spec.input_dtypes[1]) if has_smooth and len(spec.input_dtypes) > 1 else 0

    passes = cfg["dynamic_quant_passes"]
    traffic_bytes = x_bytes * passes + smooth_bytes * passes + y_bytes + scale_bytes
    row_sync_us = rows * cfg["dynamic_quant_min_row_us"]
    vf_chunks = aligned_vector_chunks(max(hidden, 1)) * max(rows, 1)
    vector_ops = max(data_elements, vf_chunks * 64) * cfg["dynamic_quant_op_factor"]
    return traffic_bytes, vector_ops, row_sync_us


def _estimate_add_rms_norm_dynamic_quant(
    spec: OtherOpSpec,
    cfg: dict[str, Any],
    hbm_bandwidth: float,
) -> tuple[float, float, float]:
    rows, hidden = infer_add_rms_norm_dims(spec.input_shapes, spec.output_shapes)
    data_elements = max(rows * hidden, spec.logical_elements)
    x_dtype_size = dtype_size(spec.input_dtypes[0]) if spec.input_dtypes else 2
    y_dtype_size = dtype_size(spec.output_dtypes[0]) if spec.output_dtypes else 1
    output_count = count_present_output_shapes(spec.output_shapes)
    quant_output_count = 1
    if len(spec.output_shapes) > 1 and len(spec.output_dtypes) > 1:
        quant_output_count += int(num_elements(spec.output_shapes[1]) == data_elements)
    quant_output_count = max(1, min(2, quant_output_count))

    x1_x2_bytes = 2 * data_elements * x_dtype_size
    x_out_bytes = data_elements * x_dtype_size
    gamma_beta_bytes = hidden * x_dtype_size
    if len(spec.input_shapes) > 3 and spec.input_shapes[3]:
        gamma_beta_bytes += hidden * x_dtype_size
    if len(spec.input_shapes) > 4 and spec.input_shapes[4]:
        gamma_beta_bytes += hidden * x_dtype_size
    if len(spec.input_shapes) > 5 and spec.input_shapes[5]:
        gamma_beta_bytes += hidden * x_dtype_size

    y_bytes = quant_output_count * data_elements * y_dtype_size
    scale_bytes = quant_output_count * rows * dtype_size("FLOAT")
    traffic_bytes = (
        x1_x2_bytes
        + x_out_bytes
        + gamma_beta_bytes
        + y_bytes
        + scale_bytes
        + data_elements * x_dtype_size * max(0.0, cfg["add_rms_norm_dynamic_quant_passes"] - 2.0)
    )
    vector_ops = data_elements * cfg["add_rms_norm_dynamic_quant_op_factor"] * max(1, quant_output_count)
    workspace_us = _bytes_to_us(max(0, output_count - 3) * rows * dtype_size("FLOAT"), hbm_bandwidth)
    return traffic_bytes, vector_ops, workspace_us


def _nz_weight_dims(shape: list[int]) -> tuple[int, int] | None:
    if len(shape) < 4:
        return None
    n_dim = shape[-4] * shape[-1]
    k_dim = shape[-3] * shape[-2]
    return k_dim, n_dim


def _matmul_us(m: int, n: int, k: int, dtype: str, config: dict[str, Any]) -> float:
    if m <= 0 or n <= 0 or k <= 0:
        return 0.0
    dtype_key = "INT8" if dtype.upper() in {"INT8", "DT_INT8"} else "DT_BF16"
    peak = float(config.get("peak_tflops", {}).get(dtype_key, config.get("peak_tflops", {}).get("DT_BF16", 1.0)))
    if dtype_key == "INT8":
        peak = float(config.get("quant_matmul", {}).get("peak_tops", {}).get("INT8", peak))
    flops = 2.0 * m * n * k
    return flops / max(peak * 1_000_000.0, 1e-9)


def _op_kind_normalized(op_type: str) -> str:
    return op_type.replace("_", "").replace("-", "").lower()


def _unknown_runtime_bounds(
    spec: OtherOpSpec,
    traffic_bytes: float,
    vector_ops: float,
    cube_compute_us: float,
    cfg: dict[str, Any],
    hbm_bandwidth: float,
) -> tuple[float, float, str]:
    normalized = _op_kind_normalized(spec.op_type)
    base_hbm = _bytes_to_us(traffic_bytes, hbm_bandwidth)
    base_vector = _ops_to_us(vector_ops, cfg["vector_gops"])
    base_body = max(base_hbm, base_vector, cube_compute_us)
    reason = "source_tiling_search_bounds"

    if spec.op_family == "index_scatter_routing":
        if normalized in {"gatherv2", "gatherv3", "gatherelements", "gatherelementsv2"}:
            min_factor, max_factor = 0.65, 3.0
            reason = "missing_indices_linear_vs_random_gather"
        elif normalized in {"scatter", "scatterupdate", "scatterndupdate", "scatterelementsv2"}:
            min_factor, max_factor = 0.75, 4.0
            reason = "missing_indices_write_conflict_bounds"
        elif normalized in {"maskedselectv3", "nonzero"}:
            min_factor, max_factor = 0.35, 2.5
            reason = "missing_mask_selected_count_bounds"
        elif normalized == "topkv2":
            min_factor, max_factor = 0.7, 3.5
            reason = "missing_topk_distribution_bounds"
        elif normalized.startswith("moe"):
            min_factor, max_factor = 0.5, 3.0
            reason = "missing_moe_routing_distribution_bounds"
        else:
            min_factor, max_factor = 0.75, 3.0
            reason = "missing_index_runtime_values_bounds"
        return base_body * min_factor, base_body * max_factor, reason

    if spec.op_family == "lightning_indexer_fusion":
        return base_body * 0.6, base_body * 2.8, "missing_sparse_runtime_values_dense_vs_active_blocks"

    if spec.op_family == "mla_prolog_fusion":
        return base_body * 0.75, base_body * 1.8, "missing_mla_tiling_key_actual_seq_cache_index"

    if spec.op_family == "layout_memory" and spec.missing_attrs:
        return base_body * 0.7, base_body * 2.2, "missing_layout_attrs_candidate_strategy_bounds"

    if spec.op_family == "quant_vector_fusion" and spec.missing_attrs:
        return base_body * 0.8, base_body * 1.6, "missing_quant_attrs_candidate_bounds"

    if spec.op_family == "cv_regular":
        return base_body * 0.6, base_body * 4.0, "cv_source_path_or_attrs_pending_bounds"

    return base_body, base_body, reason


def _cacheline_aligned_bytes(num_bytes: int, cacheline_bytes: int = 128) -> int:
    if num_bytes <= 0:
        return 0
    return ceil_div(num_bytes, cacheline_bytes) * cacheline_bytes


def _index_scatter_runtime_bytes(spec: OtherOpSpec) -> tuple[float, float, str] | None:
    normalized = _op_kind_normalized(spec.op_type)
    input_bytes = sum(spec.input_bytes)
    output_bytes = sum(spec.output_bytes)
    if normalized in {"gatherv2", "gatherv3", "gatherelements", "gatherelementsv2"}:
        data_shape = spec.input_shapes[0] if spec.input_shapes else []
        output_shape = spec.output_shapes[0] if spec.output_shapes else []
        data_dtype = spec.input_dtypes[0] if spec.input_dtypes else ""
        indices_bytes = spec.input_bytes[1] if len(spec.input_bytes) > 1 else 0
        selected_bytes = output_bytes
        row_bytes = 0
        if data_shape and output_shape:
            axis_inner = output_shape[-1] if len(output_shape) >= 1 else 1
            row_bytes = axis_inner * dtype_size(data_dtype)
        random_read_bytes = _cacheline_aligned_bytes(row_bytes) * max(spec.input_elements[1] if len(spec.input_elements) > 1 else 1, 1)
        min_bytes = indices_bytes + selected_bytes + output_bytes
        max_bytes = indices_bytes + max(selected_bytes, random_read_bytes) + output_bytes
        return float(min_bytes), float(max_bytes), "missing_indices_selected_slice_cacheline_bounds"

    if normalized in {"scatter", "scatterupdate", "scatterndupdate", "scatterelementsv2"}:
        indices_bytes = spec.input_bytes[1] if len(spec.input_bytes) > 1 else 0
        updates_bytes = spec.input_bytes[2] if len(spec.input_bytes) > 2 else output_bytes
        min_bytes = indices_bytes + updates_bytes + output_bytes
        max_bytes = indices_bytes + updates_bytes * 2 + output_bytes
        return float(min_bytes), float(max_bytes), "missing_scatter_indices_update_write_bounds"

    if normalized in {"maskedselectv3", "nonzero"}:
        mask_bytes = spec.input_bytes[1] if len(spec.input_bytes) > 1 else 0
        min_bytes = input_bytes + mask_bytes
        max_bytes = input_bytes + mask_bytes + output_bytes
        return float(min_bytes), float(max_bytes), "missing_mask_selected_count_bounds"

    if normalized.startswith("moe"):
        min_bytes = input_bytes + output_bytes
        max_bytes = input_bytes + output_bytes * 2
        return float(min_bytes), float(max_bytes), "missing_moe_routing_distribution_bounds"

    return None


def _estimate_mla_prolog(spec: OtherOpSpec, cfg: dict[str, Any], config: dict[str, Any], hbm_bandwidth: float) -> tuple[float, float, float, float]:
    shapes = spec.input_shapes
    out_shapes = spec.output_shapes
    x_shape = shapes[0] if shapes else []
    bs = x_shape[0] if len(x_shape) >= 2 else max(num_elements(out_shapes[0][:-1]) if out_shapes else 1, 1)
    hidden_x = x_shape[-1] if x_shape else 0

    weight_dq = _nz_weight_dims(shapes[1]) if len(shapes) > 1 else None
    weight_uq_qr = _nz_weight_dims(shapes[2]) if len(shapes) > 2 else None
    weight_uk = (shapes[3][0], shapes[3][-1]) if len(shapes) > 3 and len(shapes[3]) >= 2 else None
    weight_dkv_kr = _nz_weight_dims(shapes[4]) if len(shapes) > 4 else None

    cq_dim = weight_dq[1] if weight_dq else 0
    qcqr_dim = weight_uq_qr[1] if weight_uq_qr else 0
    qn_k = weight_uk[0] if weight_uk else 0
    qn_dim = weight_uk[1] if weight_uk else (out_shapes[0][-1] if out_shapes else 0)
    ckvkr_dim = weight_dkv_kr[1] if weight_dkv_kr else 0
    kv_dim = last_dim(out_shapes[2]) if len(out_shapes) > 2 else 0
    rope_dim = last_dim(out_shapes[1]) if len(out_shapes) > 1 else 0

    x_dtype = spec.input_dtypes[0] if spec.input_dtypes else "DT_BF16"
    w_dq_dtype = spec.input_dtypes[1] if len(spec.input_dtypes) > 1 else x_dtype
    w_uq_dtype = spec.input_dtypes[2] if len(spec.input_dtypes) > 2 else x_dtype
    w_dkv_dtype = spec.input_dtypes[4] if len(spec.input_dtypes) > 4 else x_dtype
    compute_dtype_dq = "INT8" if w_dq_dtype.upper() in {"INT8", "DT_INT8"} else "DT_BF16"
    compute_dtype_uq = "INT8" if w_uq_dtype.upper() in {"INT8", "DT_INT8"} else "DT_BF16"
    compute_dtype_dkv = "INT8" if w_dkv_dtype.upper() in {"INT8", "DT_INT8"} else "DT_BF16"

    matmul_us = 0.0
    matmul_us += _matmul_us(bs, cq_dim, hidden_x, compute_dtype_dq, config)
    matmul_us += _matmul_us(bs, ckvkr_dim, hidden_x, compute_dtype_dkv, config)
    matmul_us += _matmul_us(bs, qcqr_dim, max(cq_dim, 1), compute_dtype_uq, config)
    matmul_us += _matmul_us(bs, qn_dim, max(qn_k, 1), "DT_BF16", config)

    active_cache_bytes = 0
    if len(out_shapes) > 2:
        active_cache_bytes += bs * max(kv_dim, 0) * dtype_size(spec.output_dtypes[2] if len(spec.output_dtypes) > 2 else "DT_BF16")
    if len(out_shapes) > 3:
        active_cache_bytes += bs * max(rope_dim, 0) * dtype_size(spec.output_dtypes[3] if len(spec.output_dtypes) > 3 else "DT_BF16")
    output_bytes = 0
    for index, out_shape in enumerate(out_shapes):
        elems = num_elements(out_shape)
        if index in {2, 3}:
            continue
        output_bytes += elems * dtype_size(spec.output_dtypes[index] if index < len(spec.output_dtypes) else "DT_BF16")
    input_bytes = spec.input_bytes[0] if spec.input_bytes else 0
    input_bytes += sum(spec.input_bytes[5:9])
    if len(spec.input_bytes) > 11:
        input_bytes += spec.input_bytes[11]
    weight_stream_bytes = sum(spec.input_bytes[index] for index in (1, 2, 3, 4) if index < len(spec.input_bytes))
    traffic_bytes = input_bytes + weight_stream_bytes + output_bytes + active_cache_bytes

    vector_elements = bs * (max(cq_dim, 0) + max(ckvkr_dim, 0) + max(qcqr_dim, 0) + max(rope_dim, 0))
    vector_ops = vector_elements * cfg["mla_prolog_norm_rope_op_factor"]
    sync_us = cfg["mla_prolog_sync_points"] * cfg["mla_prolog_sync_us"]
    return traffic_bytes, vector_ops, matmul_us, sync_us


def _estimate_quant_lightning_indexer(
    spec: OtherOpSpec,
    cfg: dict[str, Any],
    config: dict[str, Any],
    hbm_bandwidth: float,
) -> tuple[float, float, float, float, float]:
    q_shape = spec.input_shapes[0] if spec.input_shapes else []
    k_shape = spec.input_shapes[1] if len(spec.input_shapes) > 1 else []
    out_shape = spec.output_shapes[0] if spec.output_shapes else []

    if len(q_shape) == 4:
        batch = q_shape[0]
        q_seq = q_shape[1]
        q_heads = q_shape[2]
        head_dim = q_shape[3]
        q_tokens = batch * q_seq
    elif len(q_shape) == 3:
        q_tokens = q_shape[0]
        q_heads = q_shape[1]
        head_dim = q_shape[2]
        batch = spec.input_shapes[7][0] if len(spec.input_shapes) > 7 and spec.input_shapes[7] else 1
        q_seq = max(1, q_tokens // max(batch, 1))
    else:
        q_tokens = max(num_elements(q_shape[:-1]), 1)
        q_heads = 1
        head_dim = last_dim(q_shape) or 128
        batch = 1
        q_seq = q_tokens

    if len(k_shape) == 4:
        block_num, block_size, kv_heads, k_head_dim = k_shape
        kv_seq = block_num * block_size
    elif len(k_shape) >= 3:
        kv_seq = k_shape[0]
        kv_heads = k_shape[1]
        block_size = 0
        k_head_dim = last_dim(k_shape)
    else:
        kv_seq = max(num_elements(k_shape[:-1]), 1)
        kv_heads = 1
        block_size = 0
        k_head_dim = head_dim

    if len(spec.input_shapes) > 7 and len(spec.input_shapes[7]) >= 2 and block_size > 0:
        kv_seq = max(kv_seq, spec.input_shapes[7][1] * block_size)
    if len(out_shape) >= 3:
        sparse_count = out_shape[-1]
        kv_heads = max(kv_heads, out_shape[-2])
    else:
        sparse_count = 2048

    qk_ops = 2.0 * max(q_tokens, 1) * max(kv_heads, 1) * max(kv_seq, 1) * max(k_head_dim or head_dim, 1)
    peak = float(config.get("quant_matmul", {}).get("peak_tops", {}).get("INT8", config.get("peak_tflops", {}).get("INT8", 1.0)))
    cube_compute_us = qk_ops / max(peak * 1_000_000.0, 1e-9)

    s2_base = 128
    aligned_kv = ceil_div(max(kv_seq, 1), s2_base) * s2_base
    score_workspace_bytes = max(q_tokens, 1) * max(kv_heads, 1) * aligned_kv * dtype_size("FLOAT16")
    topk_workspace_bytes = max(q_tokens, 1) * max(kv_heads, 1) * max(sparse_count, 1) * 2 * dtype_size("INT32")
    traffic_bytes = sum(spec.input_bytes) + sum(spec.output_bytes) + score_workspace_bytes + topk_workspace_bytes
    vector_ops = (
        max(q_tokens, 1)
        * max(kv_heads, 1)
        * (aligned_kv + max(sparse_count, 1) * max(ceil_div(aligned_kv, s2_base), 1))
        * cfg["quant_lightning_indexer_topk_op_factor"]
    )
    workspace_us = _bytes_to_us(score_workspace_bytes + topk_workspace_bytes, hbm_bandwidth)
    sync_us = cfg["quant_lightning_indexer_sync_points"] * cfg["quant_lightning_indexer_sync_us"]
    return traffic_bytes, vector_ops, cube_compute_us, workspace_us, sync_us


def estimate_other_op(spec: OtherOpSpec, config: dict[str, Any]) -> OtherOpCostEstimate:
    cfg = _other_ops_config(config)
    hbm_bandwidth = float(config.get("hbm_bandwidth_tbps", 1.0))
    input_bytes = float(sum(spec.input_bytes))
    output_bytes = float(sum(spec.output_bytes))
    traffic_bytes = input_bytes + output_bytes
    vector_ops = float(spec.logical_elements)
    layout_overhead_us = 0.0
    workspace_us = 0.0
    cube_compute_us = 0.0
    sync_overhead_us = 0.0
    tiling_source = "analytic_fallback"

    if spec.op_family == "layout_memory":
        vector_ops = 0.0
        if spec.op_type.replace("_", "").replace("-", "").lower() == "tril":
            vector_ops = float(spec.logical_elements) * cfg["activation_op_factor"]
        if spec.source_strategy in {
            "linear_ub_cast",
            "linear_ub_copy",
            "format_transform_nz_nd_simt",
            "format_transform_5hd_simt",
            "format_transform_simt",
        }:
            tiling_source = "source_strategy_replay"
        elif spec.missing_attrs:
            tiling_source = "source_strategy_replay_missing_attrs"
        else:
            tiling_source = "source_strategy_replay"
        if spec.missing_attrs:
            layout_overhead_us = _bytes_to_us(traffic_bytes * (cfg["layout_strided_factor"] - 1.0), hbm_bandwidth)
    elif spec.op_family == "elementwise_vector":
        tiling_source = "source_strategy_replay"
        vector_ops = float(spec.logical_elements) * _elementwise_op_factor(spec, cfg)
    elif spec.op_family == "reduction":
        tiling_source = "source_strategy_replay"
        passes = _reduction_passes(spec, cfg)
        traffic_bytes = input_bytes + output_bytes * passes
        workspace_us = _bytes_to_us(output_bytes, hbm_bandwidth)
        sync_overhead_us = 0.0
        vector_ops = float(sum(spec.input_elements) or spec.logical_elements) * passes
    elif spec.op_family == "norm_activation":
        tiling_source = "source_strategy_replay"
        if _is_activation_only(spec.op_type):
            traffic_bytes = input_bytes + output_bytes
            vector_ops = float(spec.logical_elements) * cfg["activation_op_factor"] * _norm_activation_passes(spec, cfg)
        else:
            passes = _norm_activation_passes(spec, cfg)
            traffic_bytes = (input_bytes + output_bytes) * passes
            vector_ops = float(spec.logical_elements) * cfg["activation_op_factor"]
    elif spec.op_family == "quant_vector_fusion":
        normalized = spec.op_type.replace("_", "").replace("-", "").lower()
        tiling_source = "source_strategy_replay_missing_attrs" if spec.missing_attrs else "source_strategy_replay"
        if normalized.startswith("dynamicquant") and "dynamicquantupdate" not in normalized:
            traffic_bytes, vector_ops, sync_overhead_us = _estimate_dynamic_quant(spec, cfg)
        else:
            traffic_bytes, vector_ops, workspace_us = _estimate_add_rms_norm_dynamic_quant(spec, cfg, hbm_bandwidth)
    elif spec.op_family == "mla_prolog_fusion":
        tiling_source = "source_strategy_replay_missing_runtime_values"
        traffic_bytes, vector_ops, cube_compute_us, sync_overhead_us = _estimate_mla_prolog(
            spec, cfg, config, hbm_bandwidth
        )
    elif spec.op_family == "lightning_indexer_fusion":
        tiling_source = "source_strategy_replay_missing_runtime_values"
        traffic_bytes, vector_ops, cube_compute_us, workspace_us, sync_overhead_us = _estimate_quant_lightning_indexer(
            spec, cfg, config, hbm_bandwidth
        )
    elif spec.op_family == "index_scatter_routing":
        tiling_source = "analytic_fallback_missing_runtime_values"
        runtime_bounds = _index_scatter_runtime_bytes(spec)
        if runtime_bounds is not None:
            min_bytes, max_bytes, runtime_bounds_reason = runtime_bounds
            traffic_bytes = (min_bytes + max_bytes) / 2.0
        else:
            runtime_bounds_reason = "missing_index_runtime_values_bounds"
            traffic_bytes = input_bytes + output_bytes
        vector_ops = float(spec.logical_elements)
    elif spec.op_family == "cv_regular":
        tiling_source = "analytic_fallback_source_pending"
        traffic_bytes = input_bytes + output_bytes
        vector_ops = float(spec.logical_elements) * cfg["activation_op_factor"]

    hbm_us = _bytes_to_us(traffic_bytes, hbm_bandwidth)
    vector_compute_us = _ops_to_us(vector_ops, cfg["vector_gops"])
    body_us = max(vector_compute_us, hbm_us) + cube_compute_us + layout_overhead_us + workspace_us + sync_overhead_us
    launch_us = cfg["launch_overhead_us"]
    if spec.op_family == "index_scatter_routing" and "runtime_bounds" in locals() and runtime_bounds is not None:
        min_bytes, max_bytes, bounds_reason = runtime_bounds
        min_hbm = _bytes_to_us(min_bytes, hbm_bandwidth)
        max_hbm = _bytes_to_us(max_bytes, hbm_bandwidth)
        vector_bound = _ops_to_us(vector_ops, cfg["vector_gops"])
        bounds_min_us = max(min_hbm, vector_bound, cube_compute_us)
        bounds_max_us = max(max_hbm, vector_bound, cube_compute_us)
    else:
        bounds_min_us, bounds_max_us, bounds_reason = _unknown_runtime_bounds(
            spec,
            traffic_bytes,
            vector_ops,
            cube_compute_us,
            cfg,
            hbm_bandwidth,
        )
    bounds_min_us += layout_overhead_us + workspace_us + sync_overhead_us
    bounds_max_us += layout_overhead_us + workspace_us + sync_overhead_us
    if bounds_max_us < bounds_min_us:
        bounds_max_us = bounds_min_us
    if (
        spec.op_family in {
            "index_scatter_routing",
            "lightning_indexer_fusion",
            "mla_prolog_fusion",
            "cv_regular",
        }
        or spec.missing_attrs
    ):
        body_us = (bounds_min_us + bounds_max_us) / 2.0
    total_us = body_us + launch_us
    ideal_lower_bound_us = max(_bytes_to_us(input_bytes + output_bytes, hbm_bandwidth), vector_compute_us, cube_compute_us)
    optimal_tiling_us = ideal_lower_bound_us + launch_us
    source_schedule_bound_us = body_us + launch_us
    bounds_min_total_us = bounds_min_us + launch_us
    bounds_max_total_us = bounds_max_us + launch_us
    dominant_base = max(hbm_us, vector_compute_us, cube_compute_us)
    if cube_compute_us >= hbm_us and cube_compute_us >= vector_compute_us:
        dominant = "cube"
    else:
        dominant = "hbm" if hbm_us >= vector_compute_us else "vector"
    if layout_overhead_us > dominant_base:
        dominant = "layout"
    if workspace_us > max(hbm_us, vector_compute_us, cube_compute_us, layout_overhead_us):
        dominant = "workspace"
    return OtherOpCostEstimate(
        vector_compute_us=vector_compute_us,
        cube_compute_us=cube_compute_us,
        hbm_us=hbm_us,
        layout_overhead_us=layout_overhead_us,
        workspace_us=workspace_us,
        sync_overhead_us=sync_overhead_us,
        launch_overhead_us=launch_us,
        bounds_min_us=bounds_min_total_us,
        bounds_max_us=bounds_max_total_us,
        optimal_tiling_us=optimal_tiling_us,
        source_schedule_bound_us=source_schedule_bound_us,
        total_us=total_us,
        ideal_lower_bound_us=ideal_lower_bound_us,
        current_kernel_bound_us=body_us,
        tiling_source=tiling_source,
        forward_eval_mode=(
            "source_tiling_search_no_calib"
            if spec.op_family in {"mla_prolog_fusion", "lightning_indexer_fusion", "cv_regular"} or spec.missing_attrs
            else "bounds_first_no_runtime_values"
            if spec.op_family == "index_scatter_routing"
            else "source_strategy_replay"
        ),
        bounds_reason=bounds_reason,
        dominant_component=dominant,
    )
