from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from op_eval.common import load_config

from .common import (
    MatmulSpec,
    QuantSpec,
    RuntimeKbEntry,
    TileEstimate,
    calibration_value,
    dtype_bitwidth,
    is_quant_kernel_type,
    is_quantized_data_dtype,
)
from .kernel_model import dominant_bottleneck, select_tile_estimate
from .quant_model import estimate_quant_cost, infer_nd2nz_operands
from .runtime_kb import load_runtime_kb


@dataclass(frozen=True)
class MatmulCostEstimate:
    """Public cost estimate returned by the matmul evaluator API.

    The object intentionally keeps both user-facing totals and intermediate
    components. Reports use these fields to explain whether a row is compute,
    memory, launch, format, quant, or fallback-tiling dominated.
    """

    spec: MatmulSpec
    dtype: str
    output_dtype: str
    kernel_type: str
    flops: int
    aligned_flops: int
    flops_cost_us: float | None
    memory_access_us: float
    total_us: float
    bound_type: str
    kernel_bound_type: str
    dominant_component: str
    launch_overhead_us: float
    template_overhead_us: float
    format_overhead_us: float
    pipeline_efficiency: float
    kernel_lower_bound_us: float
    gm_bytes_min: int
    gm_bytes_tiled: int
    tile: TileEstimate
    nd2nz_a: bool = False
    nd2nz_b: bool = False
    quant_spec: QuantSpec | None = None
    quant_compute_us: float | None = None
    quant_hbm_us: float | None = None
    quant_dequant_us: float = 0.0
    quant_gm_bytes_min: int | None = None
    quant_gm_bytes_tiled: int | None = None

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
        raise ValueError("estimate_matmul requires either config or config_path")
    return load_config(path)


def _resolve_runtime_kb(
    config: dict[str, Any],
    runtime_kb: dict[tuple[Any, ...], list[RuntimeKbEntry]] | None,
) -> dict[tuple[Any, ...], list[RuntimeKbEntry]]:
    if runtime_kb is not None:
        return runtime_kb
    if config.get("kernel_model", {}).get("runtime_kb", {}).get("enabled", False):
        return load_runtime_kb(config)
    return {}


def _normalize_dtype(dtype: str | None, default: str = "UNKNOWN") -> str:
    if not dtype:
        return default
    return dtype.strip().upper()


def _kernel_bound_type(compute_us: float | None, memory_us: float) -> str:
    if compute_us is None:
        return "unknown_compute_bound"
    if compute_us > memory_us * 1.2:
        return "compute_bound"
    if memory_us > compute_us * 1.2:
        return "memory_access_bound"
    return "balanced_bound"


def _total_bound_type(dominant_component: str, kernel_bound_type: str) -> str:
    if dominant_component == "launch":
        return "launch_bound"
    if dominant_component == "format":
        return "format_bound"
    if dominant_component == "hbm":
        return "memory_access_bound"
    if dominant_component == "compute":
        return "compute_bound"
    return kernel_bound_type


def _default_quant_mode(input_dtypes: list[str]) -> str:
    data_bits = [dtype_bitwidth(dtype) for dtype in input_dtypes[:2] if is_quantized_data_dtype(dtype)]
    return f"int{min(data_bits)}" if data_bits else "unknown"


def _small_m_matmul_v2_serial_us(
    spec: MatmulSpec,
    tile: TileEstimate,
    config: dict[str, Any],
    kernel_type: str,
) -> float:
    model = config.get("matmul_model", {}).get("small_m_matmul_v2", {})
    if not model.get("enabled", False):
        return 0.0
    if kernel_type not in set(model.get("applies_to", ["MatMulV2"])):
        return 0.0
    max_edge = int(model.get("max_m_or_n", 1))
    if min(spec.m, spec.n) > max_edge:
        return 0.0
    if spec.a_format != "ND" or spec.b_format != "ND" or spec.output_format != "ND":
        return 0.0
    if tile.source != "analytic_search":
        return 0.0
    max_l2_bytes = int(model.get("max_gm_bytes_for_l2_resident", config.get("l2_bytes", 0)))
    if max_l2_bytes > 0 and tile.gm_bytes_min > max_l2_bytes:
        return 0.0
    effective_tflops = float(model.get("effective_aligned_tflops", 0.0))
    if effective_tflops <= 0:
        return 0.0
    return tile.aligned_flops / (effective_tflops * 1_000_000.0)


def _quant_weight_nz_epilogue_us(
    spec: MatmulSpec,
    tile: TileEstimate,
    quant_spec: QuantSpec | None,
    config: dict[str, Any],
    kernel_type: str,
) -> float:
    model = config.get("quant_matmul", {}).get("weight_nz_epilogue", {})
    if not model.get("enabled", False):
        return 0.0
    if quant_spec is None or not quant_spec.is_quant:
        return 0.0
    if kernel_type not in set(model.get("applies_to", ["QuantBatchMatmulV3"])):
        return 0.0
    if spec.b_format != "FRACTAL_NZ":
        return 0.0
    if quant_spec.granularity not in set(model.get("granularities", ["per_channel_n"])):
        return 0.0
    if quant_spec.compute_path not in set(model.get("compute_paths", ["full_quant"])):
        return 0.0
    if min(spec.m, spec.n) > int(model.get("max_m_or_n", 4)):
        return 0.0
    min_n_tiles = int(model.get("min_n_tiles", 1))
    if tile.tile_n < min_n_tiles:
        return 0.0

    per_n_tile_us = float(model.get("per_n_tile_us", 0.0))
    per_k_tile_us = float(model.get("per_k_tile_us", 0.0))
    scale_bytes_per_n_tile = int(model.get("scale_bytes_per_n_tile", tile.base_n * 8))
    scale_replay_bytes = tile.tile_n * max(0, scale_bytes_per_n_tile)
    scale_replay_us = scale_replay_bytes / max(float(config["hbm_bandwidth_tbps"]) * 1_000_000.0, 1e-9)
    return tile.tile_n * per_n_tile_us + tile.tile_n * max(1, tile.tile_k) * per_k_tile_us + scale_replay_us


def estimate_matmul_cost(
    spec: MatmulSpec,
    dtype: str,
    *,
    config: dict[str, Any] | str | Path | None = None,
    config_path: str | Path | None = None,
    output_dtype: str | None = None,
    runtime_kb: dict[tuple[Any, ...], list[RuntimeKbEntry]] | None = None,
    input_dtypes: list[str] | None = None,
    kernel_type: str = "MatMulV3",
    quant_spec: QuantSpec | None = None,
    input_shapes: list[list[int]] | None = None,
    include_launch: bool = True,
) -> MatmulCostEstimate:
    """Estimate one matmul kernel from a logical spec and data type.

    `spec` is already the logical GEMM reconstructed from profiling
    shape/layout fields. This function then selects the best available kernel
    tiling source, applies quantization/format/launch terms, and returns the
    current-kernel estimate. The result is not the ideal physical lower bound;
    `tile.source` and report fields identify whether the estimate came from
    runtime KB, an ops-nn tiling replay approximation, or analytic fallback.

    Stable public fields for external callers are `flops_cost_us`,
    `memory_access_us`, `total_us`, and `bound_type`.
    """

    resolved_config = _resolve_config(config, config_path)
    resolved_runtime_kb = _resolve_runtime_kb(resolved_config, runtime_kb)
    dtype = _normalize_dtype(dtype)
    output_dtype = _normalize_dtype(output_dtype, dtype)
    input_dtypes = [_normalize_dtype(item) for item in (input_dtypes or [dtype, dtype])]

    tile = select_tile_estimate(spec, dtype, output_dtype, resolved_config, resolved_runtime_kb, input_dtypes, kernel_type)
    flops = 2 * spec.m * spec.n * spec.k * spec.batch
    launch_us = calibration_value(resolved_config, "launch_overhead_us_by_type", kernel_type, 0.0)
    if not include_launch:
        launch_us = 0.0

    pipeline_efficiency = calibration_value(resolved_config, "pipeline_efficiency_by_dtype", dtype, 1.0)
    pipeline_efficiency = max(pipeline_efficiency, 1e-9)
    nd2nz_a, nd2nz_b = infer_nd2nz_operands(spec, dtype)
    format_overhead_us = 0.0
    if nd2nz_a or nd2nz_b:
        format_overhead_us += (
            int(nd2nz_a) + int(nd2nz_b)
        ) * calibration_value(resolved_config, "format_overhead_us", "ND2NZ", 0.0)

    quant_compute_us: float | None = None
    quant_hbm_us: float | None = None
    quant_dequant_us = 0.0
    quant_gm_bytes_min: int | None = None
    quant_gm_bytes_tiled: int | None = None
    if quant_spec is not None and quant_spec.is_quant:
        (
            quant_compute_us,
            quant_hbm_us,
            quant_dequant_us,
            quant_gm_bytes_min,
            quant_gm_bytes_tiled,
        ) = estimate_quant_cost(
            spec,
            tile,
            quant_spec,
            input_shapes or [],
            input_dtypes,
            output_dtype,
            resolved_config,
        )

    if quant_spec is not None and quant_spec.is_quant:
        flops_cost_us = quant_compute_us
        memory_access_us = quant_hbm_us if quant_hbm_us is not None else tile.hbm_us
        gm_bytes_min = quant_gm_bytes_min if quant_gm_bytes_min is not None else tile.gm_bytes_min
        gm_bytes_tiled = quant_gm_bytes_tiled if quant_gm_bytes_tiled is not None else tile.gm_bytes_tiled
    else:
        flops_cost_us = tile.compute_us / pipeline_efficiency if tile.compute_us is not None else None
        memory_access_us = tile.hbm_us
        gm_bytes_min = tile.gm_bytes_min
        gm_bytes_tiled = tile.gm_bytes_tiled

    template_overhead_us = 0.0
    if quant_spec is None or not quant_spec.is_quant:
        template_overhead_us = _small_m_matmul_v2_serial_us(spec, tile, resolved_config, kernel_type)
    else:
        template_overhead_us = _quant_weight_nz_epilogue_us(spec, tile, quant_spec, resolved_config, kernel_type)

    kernel_lower_bound_us = max(value for value in (flops_cost_us, memory_access_us) if value is not None)
    kernel_lower_bound_us += template_overhead_us
    total_us = launch_us + kernel_lower_bound_us + format_overhead_us
    kernel_bound = _kernel_bound_type(flops_cost_us, memory_access_us)
    dominant_component = dominant_bottleneck(launch_us, flops_cost_us, memory_access_us, format_overhead_us)
    bound_type = _total_bound_type(dominant_component, kernel_bound)

    return MatmulCostEstimate(
        spec=spec,
        dtype=dtype,
        output_dtype=output_dtype,
        kernel_type=kernel_type,
        flops=flops,
        aligned_flops=tile.aligned_flops,
        flops_cost_us=flops_cost_us,
        memory_access_us=memory_access_us,
        total_us=total_us,
        bound_type=bound_type,
        kernel_bound_type=kernel_bound,
        dominant_component=dominant_component,
        launch_overhead_us=launch_us,
        template_overhead_us=template_overhead_us,
        format_overhead_us=format_overhead_us,
        pipeline_efficiency=pipeline_efficiency,
        kernel_lower_bound_us=kernel_lower_bound_us,
        gm_bytes_min=gm_bytes_min,
        gm_bytes_tiled=gm_bytes_tiled,
        tile=tile,
        nd2nz_a=nd2nz_a,
        nd2nz_b=nd2nz_b,
        quant_spec=quant_spec,
        quant_compute_us=quant_compute_us,
        quant_hbm_us=quant_hbm_us,
        quant_dequant_us=quant_dequant_us,
        quant_gm_bytes_min=quant_gm_bytes_min,
        quant_gm_bytes_tiled=quant_gm_bytes_tiled,
    )


def estimate_matmul(
    m: int,
    n: int,
    k: int,
    dtype: str,
    *,
    config: dict[str, Any] | str | Path | None = None,
    config_path: str | Path | None = None,
    batch: int = 1,
    output_dtype: str | None = None,
    trans_a: bool = False,
    trans_b: bool = False,
    a_format: str = "ND",
    b_format: str = "ND",
    output_format: str = "ND",
    a_storage_elements: int | None = None,
    b_storage_elements: int | None = None,
    output_storage_elements: int | None = None,
    runtime_kb: dict[tuple[Any, ...], list[RuntimeKbEntry]] | None = None,
    input_dtypes: list[str] | None = None,
    kernel_type: str = "MatMulV3",
    quant_mode: str | None = None,
    quant_granularity: str = "none",
    quant_compute_path: str | None = None,
    quant_aux_elements: int = 0,
    quant_aux_bytes: int = 0,
    include_launch: bool = True,
) -> MatmulCostEstimate:
    """Public one-call interface for estimating a matmul by shape and dtype."""

    spec = MatmulSpec(
        m=m,
        n=n,
        k=k,
        batch=batch,
        trans_a=trans_a,
        trans_b=trans_b,
        a_format=a_format,
        b_format=b_format,
        output_format=output_format,
        a_storage_elements=a_storage_elements,
        b_storage_elements=b_storage_elements,
        output_storage_elements=output_storage_elements,
    )
    normalized_inputs = [_normalize_dtype(item) for item in (input_dtypes or [dtype, dtype])]
    quant_spec = None
    if quant_mode or quant_compute_path or is_quant_kernel_type(kernel_type):
        quant_spec = QuantSpec(
            is_quant=True,
            mode=quant_mode or _default_quant_mode(normalized_inputs),
            granularity=quant_granularity,
            compute_path=quant_compute_path or "full_quant",
            aux_elements=quant_aux_elements,
            aux_bytes=quant_aux_bytes,
            notes="api_provided",
        )

    return estimate_matmul_cost(
        spec,
        dtype,
        config=config,
        config_path=config_path,
        output_dtype=output_dtype,
        runtime_kb=runtime_kb,
        input_dtypes=normalized_inputs,
        kernel_type=kernel_type,
        quant_spec=quant_spec,
        include_launch=include_launch,
    )
