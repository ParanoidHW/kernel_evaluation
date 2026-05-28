from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import OtherOpSpec


@dataclass(frozen=True)
class OtherOpCostEstimate:
    vector_compute_us: float
    hbm_us: float
    layout_overhead_us: float
    workspace_us: float
    sync_overhead_us: float
    launch_overhead_us: float
    total_us: float
    ideal_lower_bound_us: float
    current_kernel_bound_us: float
    tiling_source: str
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
    if normalized == "rotarypositionembedding":
        return 6.0
    if normalized in {"cos", "sin"}:
        return cfg["transcendental_op_factor"]
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


def estimate_other_op(spec: OtherOpSpec, config: dict[str, Any]) -> OtherOpCostEstimate:
    cfg = _other_ops_config(config)
    hbm_bandwidth = float(config.get("hbm_bandwidth_tbps", 1.0))
    input_bytes = float(sum(spec.input_bytes))
    output_bytes = float(sum(spec.output_bytes))
    traffic_bytes = input_bytes + output_bytes
    vector_ops = float(spec.logical_elements)
    layout_overhead_us = 0.0
    workspace_us = 0.0
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
    elif spec.op_family == "index_scatter_routing":
        tiling_source = "analytic_fallback_missing_runtime_values"
        traffic_bytes = (input_bytes + output_bytes) * cfg["index_random_access_factor"]
        vector_ops = float(spec.logical_elements)
    elif spec.op_family == "cv_regular":
        tiling_source = "analytic_fallback_source_pending"
        traffic_bytes = input_bytes + output_bytes
        vector_ops = float(spec.logical_elements) * cfg["activation_op_factor"]

    hbm_us = _bytes_to_us(traffic_bytes, hbm_bandwidth)
    vector_compute_us = _ops_to_us(vector_ops, cfg["vector_gops"])
    body_us = max(vector_compute_us, hbm_us) + layout_overhead_us + workspace_us + sync_overhead_us
    launch_us = cfg["launch_overhead_us"]
    total_us = body_us + launch_us
    ideal_lower_bound_us = max(_bytes_to_us(input_bytes + output_bytes, hbm_bandwidth), vector_compute_us)
    dominant = "hbm" if hbm_us >= vector_compute_us else "vector"
    if layout_overhead_us > max(hbm_us, vector_compute_us):
        dominant = "layout"
    if workspace_us > max(hbm_us, vector_compute_us, layout_overhead_us):
        dominant = "workspace"
    return OtherOpCostEstimate(
        vector_compute_us=vector_compute_us,
        hbm_us=hbm_us,
        layout_overhead_us=layout_overhead_us,
        workspace_us=workspace_us,
        sync_overhead_us=sync_overhead_us,
        launch_overhead_us=launch_us,
        total_us=total_us,
        ideal_lower_bound_us=ideal_lower_bound_us,
        current_kernel_bound_us=body_us,
        tiling_source=tiling_source,
        dominant_component=dominant,
    )
