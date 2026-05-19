from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from op_eval.common import calibration_value, dtype_size, load_config, peak_for_dtype

from .common import AttentionSpec
from .tiling_replay import replay_attention_tiling_strategy


@dataclass(frozen=True)
class AttentionCostEstimate:
    spec: AttentionSpec
    dtype: str
    output_dtype: str
    kernel_type: str
    flops: int
    vector_ops: int
    gm_bytes_min: int
    current_gm_bytes: int
    compute_us: float | None
    vector_us: float
    hbm_us: float
    current_compute_us: float | None
    current_vector_us: float
    current_hbm_us: float
    lower_bound_us: float
    current_kernel_bound_us: float
    total_us: float
    bound_type: str
    dominant_component: str
    launch_overhead_us: float
    pipeline_efficiency: float
    occupancy_efficiency: float
    traffic_factor: float
    q_block_tiles: int
    kv_block_tiles: int
    work_tiles: int
    sync_overhead_us: float
    latency_floor_us: float
    template_overhead_factor: float
    actual_tiling_source: str
    fallback_tiling_source: str
    optimal_tiling_source: str
    current_tiling_kind: str
    tiling_strategy: str
    ops_transformer_source_file: str
    tiling_notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_config(
    config: dict[str, Any] | str | Path | None,
    config_path: str | Path | None,
) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    path = config_path or config
    if path is None:
        raise ValueError("estimate_attention requires either config or config_path")
    return load_config(path)


def _bound_type(compute_us: float | None, vector_us: float, hbm_us: float) -> str:
    compute_like = max(value for value in (compute_us, vector_us) if value is not None)
    if compute_like > hbm_us * 1.2:
        return "compute_bound"
    if hbm_us > compute_like * 1.2:
        return "memory_access_bound"
    return "balanced_bound"


def _dominant(launch_us: float, compute_us: float | None, vector_us: float, hbm_us: float) -> str:
    values = {
        "launch": launch_us,
        "vector": vector_us,
        "hbm": hbm_us,
    }
    if compute_us is not None:
        values["compute"] = compute_us
    return max(values.items(), key=lambda item: item[1])[0]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _attention_defaults(config: dict[str, Any]) -> dict[str, float]:
    soc = str(config.get("soc", "")).lower()
    if "910b" in soc:
        return {
            "launch_overhead_us": 4.0,
            "decode_latency_floor_us": 10.0,
            "flash_decode_latency_floor_us": 10.0,
            "fused_infer_decode_latency_floor_us": 10.0,
            "short_prefill_latency_floor_us": 9.0,
            "prefill_latency_floor_us": 5.0,
            "sync_us_per_kv_tile": 0.45,
            "decode_traffic_factor": 3.5,
            "prefill_traffic_factor": 1.8,
            "mask_traffic_factor": 0.35,
            "gqa_traffic_factor": 0.20,
            "workspace_score_factor": 0.35,
            "min_occupancy_efficiency": 0.08,
            "vector_decode_multiplier": 5.0,
            "vector_prefill_multiplier": 2.5,
            "template_factor": 2.1,
            "decode_template_factor": 2.1,
            "short_prefill_template_factor": 1.15,
            "flash_short_prefill_template_factor": 1.15,
            "fused_infer_short_prefill_template_factor": 1.60,
            "flash_prefill_template_factor": 2.1,
            "fused_infer_prefill_template_factor": 2.1,
        }
    return {
        "launch_overhead_us": 3.0,
        "decode_latency_floor_us": 24.0,
        "flash_decode_latency_floor_us": 24.0,
        "fused_infer_decode_latency_floor_us": 17.0,
        "short_prefill_latency_floor_us": 32.0,
        "prefill_latency_floor_us": 4.0,
        "sync_us_per_kv_tile": 0.25,
        "decode_traffic_factor": 2.2,
        "prefill_traffic_factor": 1.6,
        "mask_traffic_factor": 0.30,
        "gqa_traffic_factor": 0.15,
        "workspace_score_factor": 0.30,
        "min_occupancy_efficiency": 0.10,
        "vector_decode_multiplier": 3.5,
        "vector_prefill_multiplier": 2.2,
        "template_factor": 1.0,
        "decode_template_factor": 0.75,
        "short_prefill_template_factor": 1.30,
        "flash_short_prefill_template_factor": 1.30,
        "fused_infer_short_prefill_template_factor": 10.80,
        "flash_prefill_template_factor": 2.55,
        "fused_infer_prefill_template_factor": 3.0,
    }


def _attention_knob(config: dict[str, Any], key: str) -> float:
    defaults = _attention_defaults(config)
    model = config.get("attention_model", {})
    return float(model.get(key, defaults[key]))


def _template_factor(config: dict[str, Any], kernel_type: str, spec: AttentionSpec, is_decode: bool) -> float:
    model = config.get("attention_model", {})
    text = kernel_type.lower()
    if is_decode:
        key = "decode_template_factor"
    elif spec.q_seq <= 512 and spec.kv_seq <= 512 and "flashattentionscore" in text:
        key = "flash_short_prefill_template_factor"
    elif spec.q_seq <= 512 and spec.kv_seq <= 512 and "fusedinfer" in text:
        key = "fused_infer_short_prefill_template_factor"
    elif spec.q_seq <= 512 and spec.kv_seq <= 512:
        key = "short_prefill_template_factor"
    elif "flashattentionscore" in text:
        key = "flash_prefill_template_factor"
    elif "fusedinfer" in text:
        key = "fused_infer_prefill_template_factor"
    else:
        key = "template_factor"
    if key in model:
        return float(model[key])
    return _attention_knob(config, key)


def _kernel_aware_components(
    spec: AttentionSpec,
    config: dict[str, Any],
    *,
    dtype: str,
    output_dtype: str,
    kernel_type: str,
    ideal_compute_us: float | None,
    ideal_vector_us: float,
    gm_bytes_min: int,
) -> dict[str, Any]:
    # ops-transformer fused-infer attention uses Q_TILE_CEIL=128 and
    # MAX_KV_STACK_LEN=512 in the split-fuse path.
    q_block = 128
    kv_block = 512
    q_tiles = _ceil_div(spec.q_seq, q_block)
    kv_tiles = _ceil_div(spec.kv_seq, kv_block)
    work_tiles = max(1, spec.batch * spec.q_heads * q_tiles * kv_tiles)
    aic_num = max(1, int(config.get("aic_num", 1)))
    occupancy = min(1.0, work_tiles / aic_num)
    occupancy = max(occupancy, _attention_knob(config, "min_occupancy_efficiency"))

    is_decode = spec.q_seq <= 1
    template_factor = _template_factor(config, kernel_type, spec, is_decode)
    traffic_factor = _attention_knob(config, "decode_traffic_factor" if is_decode else "prefill_traffic_factor")
    if spec.causal_or_masked:
        traffic_factor += _attention_knob(config, "mask_traffic_factor")
    if spec.kv_heads < spec.q_heads:
        traffic_factor += _attention_knob(config, "gqa_traffic_factor")

    # Online softmax and split-fuse paths keep score tiles in UB where possible,
    # but masks/LSE/split-KV combine paths can spill metadata/workspace traffic.
    score_workspace_factor = _attention_knob(config, "workspace_score_factor")
    workspace_bytes = int(spec.score_elements * 4 * score_workspace_factor)
    current_gm_bytes = int(gm_bytes_min * traffic_factor + workspace_bytes)
    current_hbm_us = current_gm_bytes / max(float(config["hbm_bandwidth_tbps"]) * 1_000_000.0, 1e-9)

    current_compute_us = ideal_compute_us / occupancy if ideal_compute_us is not None else None
    vector_multiplier = _attention_knob(config, "vector_decode_multiplier" if is_decode else "vector_prefill_multiplier")
    current_vector_us = ideal_vector_us * vector_multiplier / occupancy
    sync_overhead_us = kv_tiles * _attention_knob(config, "sync_us_per_kv_tile")
    if is_decode:
        text = kernel_type.lower()
        if "flashattentionscore" in text:
            latency_floor_us = _attention_knob(config, "flash_decode_latency_floor_us")
        elif "fusedinfer" in text:
            latency_floor_us = _attention_knob(config, "fused_infer_decode_latency_floor_us")
        else:
            latency_floor_us = _attention_knob(config, "decode_latency_floor_us")
    elif spec.q_seq <= 512 and spec.kv_seq <= 512:
        latency_floor_us = _attention_knob(config, "short_prefill_latency_floor_us")
    else:
        latency_floor_us = _attention_knob(config, "prefill_latency_floor_us")
    return {
        "current_gm_bytes": current_gm_bytes,
        "current_hbm_us": current_hbm_us,
        "current_compute_us": current_compute_us,
        "current_vector_us": current_vector_us,
        "occupancy_efficiency": occupancy,
        "traffic_factor": traffic_factor,
        "q_block_tiles": q_tiles,
        "kv_block_tiles": kv_tiles,
        "work_tiles": work_tiles,
        "sync_overhead_us": sync_overhead_us,
        "latency_floor_us": latency_floor_us,
        "template_overhead_factor": template_factor,
    }


def estimate_attention_cost(
    spec: AttentionSpec,
    dtype: str,
    *,
    config: dict[str, Any] | str | Path | None = None,
    config_path: str | Path | None = None,
    output_dtype: str | None = None,
    kernel_type: str = "FusedInferAttentionScore",
    include_launch: bool = True,
) -> AttentionCostEstimate:
    resolved_config = _resolve_config(config, config_path)
    dtype = (dtype or "UNKNOWN").strip().upper()
    output_dtype = (output_dtype or dtype).strip().upper()
    peak_tflops = peak_for_dtype(resolved_config, dtype)
    pipeline_efficiency = calibration_value(resolved_config, "pipeline_efficiency_by_dtype", dtype, 1.0)
    pipeline_efficiency = max(pipeline_efficiency, 1e-9)
    launch_us = _attention_knob(resolved_config, "launch_overhead_us")
    launch_by_type = resolved_config.get("calibration", {}).get("launch_overhead_us_by_type", {})
    if kernel_type in launch_by_type:
        launch_us = float(launch_by_type[kernel_type])
    if not include_launch:
        launch_us = 0.0

    qk_flops = 2 * spec.batch * spec.q_heads * spec.q_seq * spec.kv_seq * spec.head_dim
    pv_flops = 2 * spec.batch * spec.q_heads * spec.q_seq * spec.kv_seq * spec.value_dim
    flops = qk_flops + pv_flops
    compute_us = None
    if peak_tflops:
        compute_us = flops / (peak_tflops * 1_000_000.0 * pipeline_efficiency)

    vector_ops = 4 * spec.score_elements + spec.output_elements
    vector_tops = float(resolved_config.get("attention_model", {}).get("vector_tops", resolved_config.get("aiv_num", 1) * 2.0))
    vector_eff = float(resolved_config.get("attention_model", {}).get("vector_efficiency", 0.5))
    vector_us = vector_ops / max(vector_tops * 1_000_000.0 * vector_eff, 1e-9)

    input_bytes = (spec.q_elements + spec.k_elements + spec.v_elements) * dtype_size(dtype)
    aux_bytes = spec.aux_elements * 4
    output_bytes = spec.output_elements * dtype_size(output_dtype)
    score_spill_factor = float(resolved_config.get("attention_model", {}).get("score_spill_factor", 0.0))
    score_bytes = int(spec.score_elements * 4 * score_spill_factor)
    gm_bytes_min = input_bytes + aux_bytes + output_bytes + score_bytes
    hbm_us = gm_bytes_min / max(float(resolved_config["hbm_bandwidth_tbps"]) * 1_000_000.0, 1e-9)

    kernel_terms = [vector_us, hbm_us]
    if compute_us is not None:
        kernel_terms.append(compute_us)
    lower_bound_us = max(kernel_terms)
    current = _kernel_aware_components(
        spec,
        resolved_config,
        dtype=dtype,
        output_dtype=output_dtype,
        kernel_type=kernel_type,
        ideal_compute_us=compute_us,
        ideal_vector_us=vector_us,
        gm_bytes_min=gm_bytes_min,
    )
    current_terms = [current["current_vector_us"], current["current_hbm_us"]]
    if current["current_compute_us"] is not None:
        current_terms.append(current["current_compute_us"])
    current_kernel_bound_us = max(current_terms)
    kernel_with_sync_us = current_kernel_bound_us * current["template_overhead_factor"] + current["sync_overhead_us"]
    total_us = launch_us + max(kernel_with_sync_us, current["latency_floor_us"])
    bound_type = _bound_type(current["current_compute_us"], current["current_vector_us"], current["current_hbm_us"])
    dominant = _dominant(launch_us, current["current_compute_us"], current["current_vector_us"], current["current_hbm_us"])
    if dominant == "launch":
        total_bound_type = "launch_bound"
    else:
        total_bound_type = bound_type
    tiling = replay_attention_tiling_strategy(spec, kernel_type, resolved_config)

    return AttentionCostEstimate(
        spec=spec,
        dtype=dtype,
        output_dtype=output_dtype,
        kernel_type=kernel_type,
        flops=flops,
        vector_ops=vector_ops,
        gm_bytes_min=gm_bytes_min,
        current_gm_bytes=current["current_gm_bytes"],
        compute_us=compute_us,
        vector_us=vector_us,
        hbm_us=hbm_us,
        current_compute_us=current["current_compute_us"],
        current_vector_us=current["current_vector_us"],
        current_hbm_us=current["current_hbm_us"],
        lower_bound_us=lower_bound_us,
        current_kernel_bound_us=current_kernel_bound_us,
        total_us=total_us,
        bound_type=total_bound_type,
        dominant_component=dominant,
        launch_overhead_us=launch_us,
        pipeline_efficiency=pipeline_efficiency,
        occupancy_efficiency=current["occupancy_efficiency"],
        traffic_factor=current["traffic_factor"],
        q_block_tiles=current["q_block_tiles"],
        kv_block_tiles=current["kv_block_tiles"],
        work_tiles=current["work_tiles"],
        sync_overhead_us=current["sync_overhead_us"],
        latency_floor_us=current["latency_floor_us"],
        template_overhead_factor=current["template_overhead_factor"],
        actual_tiling_source=tiling.actual_tiling_source,
        fallback_tiling_source=tiling.fallback_tiling_source,
        optimal_tiling_source=tiling.optimal_tiling_source,
        current_tiling_kind=tiling.current_tiling_kind,
        tiling_strategy=tiling.tiling_strategy,
        ops_transformer_source_file=tiling.ops_transformer_source_file,
        tiling_notes=tiling.notes,
    )


def estimate_attention(
    batch: int,
    q_heads: int,
    kv_heads: int,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: str,
    *,
    config: dict[str, Any] | str | Path | None = None,
    config_path: str | Path | None = None,
    value_dim: int | None = None,
    output_dtype: str | None = None,
    kernel_type: str = "FusedInferAttentionScore",
    include_launch: bool = True,
) -> AttentionCostEstimate:
    value_dim = value_dim or head_dim
    spec = AttentionSpec(
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        q_seq=q_seq,
        kv_seq=kv_seq,
        head_dim=head_dim,
        value_dim=value_dim,
        output_elements=batch * q_heads * q_seq * value_dim,
        q_elements=batch * q_heads * q_seq * head_dim,
        k_elements=batch * kv_heads * kv_seq * head_dim,
        v_elements=batch * kv_heads * kv_seq * value_dim,
        aux_elements=0,
        raw_aux_elements=0,
        score_elements=batch * q_heads * q_seq * kv_seq,
        layout="api",
        variant="attention",
        causal_or_masked=False,
    )
    return estimate_attention_cost(
        spec,
        dtype,
        config=config,
        config_path=config_path,
        output_dtype=output_dtype,
        kernel_type=kernel_type,
        include_launch=include_launch,
    )
