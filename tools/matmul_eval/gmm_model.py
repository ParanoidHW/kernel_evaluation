from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .common import MatmulSpec, QuantSpec, ceil_align, ceil_div, dtype_size, peak_for_dtype
from .quant_model import quant_storage_bytes


@dataclass(frozen=True)
class GroupedMatmulScenarioEstimate:
    """Cost estimate for one GroupedMatmul routing scenario."""

    scenario: str
    active_experts: int
    tokens_per_active_expert: int
    gm_bytes: int
    compute_us: float | None
    hbm_us: float
    scheduler_us: float
    total_us: float
    core_efficiency: float
    work_tiles: int

    def to_dict(self, prefix: str) -> dict[str, Any]:
        return {f"{prefix}_{key}": value for key, value in asdict(self).items() if key != "scenario"}


@dataclass(frozen=True)
class GroupedMatmulEstimate:
    """Routing-aware estimates for a GroupedMatmul row.

    ops-transformer exposes the GroupedMatmul host tiling and kernel scheduler,
    but profiling CSVs do not carry groupList/tuningConfig. Therefore the model
    reports explicit routing scenarios and a source-visible group scheduler
    term, rather than inferring active experts from block_dim.
    """

    expert_count: int
    weight_elements_per_expert: int
    weight_bytes_per_expert: int
    balanced: GroupedMatmulScenarioEstimate
    extreme: GroupedMatmulScenarioEstimate

    def to_report_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "gmm_expert_count": self.expert_count,
            "gmm_weight_elements_per_expert": self.weight_elements_per_expert,
            "gmm_weight_bytes_per_expert": self.weight_bytes_per_expert,
        }
        fields.update(self.balanced.to_dict("gmm_balanced"))
        fields.update(self.extreme.to_dict("gmm_extreme"))
        return fields


def _expert_count_from_weight_shape(weight_shape: list[int]) -> int:
    if len(weight_shape) >= 5:
        return max(1, weight_shape[0])
    return 1


def _core_efficiency(spec: MatmulSpec, active_experts: int, tokens_per_expert: int, config: dict[str, Any]) -> tuple[float, int]:
    aic_num = max(1, int(config["aic_num"]))
    # GroupedMatmul schedules independent expert GEMMs. Approximate available
    # parallel work from active experts and N tiles; token tiles are usually
    # small for inference MoE but still matter when a single expert dominates.
    token_tiles = max(1, ceil_div(tokens_per_expert, 16))
    n_tiles = max(1, ceil_div(spec.n, 128))
    work_tiles = max(1, active_experts * token_tiles * n_tiles)
    rounds = max(1, ceil_div(work_tiles, aic_num))
    return min(1.0, work_tiles / max(rounds * aic_num, 1)), work_tiles


def _scenario_estimate(
    *,
    scenario: str,
    spec: MatmulSpec,
    config: dict[str, Any],
    dtype: str,
    output_dtype: str,
    input_dtypes: list[str],
    quant_spec: QuantSpec,
    expert_count: int,
    active_experts: int,
    weight_elements_per_expert: int,
    aux_bytes: int,
) -> GroupedMatmulScenarioEstimate:
    tokens_per_expert = max(1, ceil_div(spec.m, max(active_experts, 1)))
    core_eff, work_tiles = _core_efficiency(spec, active_experts, tokens_per_expert, config)
    output_elements = spec.output_storage_elements or spec.m * spec.n
    a_elements = spec.a_storage_elements or spec.m * spec.k
    a_dtype = input_dtypes[0] if input_dtypes else dtype
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else dtype
    out_size = dtype_size(output_dtype)
    weight_bytes = active_experts * quant_storage_bytes(weight_elements_per_expert, b_dtype)
    gm_bytes = quant_storage_bytes(a_elements, a_dtype) + weight_bytes + output_elements * out_size + aux_bytes
    hbm_us = gm_bytes / (float(config["hbm_bandwidth_tbps"]) * 1_000_000.0)

    peak = peak_for_dtype(config, a_dtype)
    if peak is None and quant_spec.is_quant:
        quant_cfg = config.get("quant_matmul", {})
        peak = quant_cfg.get("peak_tops", {}).get(a_dtype) or quant_cfg.get("peak_tops", {}).get(a_dtype.replace("DT_", ""))
    compute_us = None
    if peak is not None:
        aligned_m = active_experts * ceil_align(tokens_per_expert, 16)
        aligned_flops = 2 * aligned_m * ceil_align(spec.n, 16) * ceil_align(spec.k, 16)
        efficiency = 1.0
        if quant_spec.is_quant:
            quant_cfg = config.get("quant_matmul", {})
            efficiency = float(
                quant_cfg.get("pipeline_efficiency", {}).get(
                    a_dtype, quant_cfg.get("pipeline_efficiency", {}).get("default", 1.0)
                )
            )
            op_factor = float(quant_cfg.get("operation_factor", {}).get(quant_spec.compute_path, 1.0))
        else:
            op_factor = 1.0
        compute_us = aligned_flops * op_factor / (float(peak) * 1_000_000.0 * max(core_eff, 1e-9) * max(efficiency, 1e-9))

    # ops-transformer GMM kernels iterate over groupList in the kernel and
    # skip empty groups after reading their split value. The profiling CSV does
    # not expose groupList, so charge a small source-visible scheduling term
    # when the configured expert count exceeds available Cube cores.
    scheduler_us = max(0, expert_count - int(config["aic_num"])) * 0.04
    return GroupedMatmulScenarioEstimate(
        scenario=scenario,
        active_experts=active_experts,
        tokens_per_active_expert=tokens_per_expert,
        gm_bytes=gm_bytes,
        compute_us=compute_us,
        hbm_us=hbm_us,
        scheduler_us=scheduler_us,
        total_us=max(value for value in (compute_us, hbm_us) if value is not None) + scheduler_us,
        core_efficiency=core_eff,
        work_tiles=work_tiles,
    )


def estimate_grouped_matmul_bounds(
    spec: MatmulSpec,
    input_shapes: list[list[int]],
    input_dtypes: list[str],
    dtype: str,
    output_dtype: str,
    quant_spec: QuantSpec,
    config: dict[str, Any],
) -> GroupedMatmulEstimate | None:
    """Estimate balanced and extreme routing bounds for GroupedMatmul.

    The profiling CSV exposes expert weight storage but not the runtime
    `group_list` values. We therefore report two explicit scenarios:

    - balanced: tokens are spread across as many experts as possible.
    - extreme: all tokens route to a single expert.

    Both scenarios use the same logical token work but charge different expert
    weight traffic and parallel work tiles.
    """

    if len(input_shapes) < 2:
        return None
    expert_count = _expert_count_from_weight_shape(input_shapes[1])
    full_weight_elements = spec.b_storage_elements or spec.k * spec.n * expert_count
    weight_elements_per_expert = max(1, full_weight_elements // max(expert_count, 1))
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else dtype
    weight_bytes_per_expert = quant_storage_bytes(weight_elements_per_expert, b_dtype)
    aux_bytes = quant_spec.aux_bytes

    balanced_active = max(1, min(expert_count, spec.m))
    extreme_active = 1
    balanced = _scenario_estimate(
        scenario="balanced",
        spec=spec,
        config=config,
        dtype=dtype,
        output_dtype=output_dtype,
        input_dtypes=input_dtypes,
        quant_spec=quant_spec,
        expert_count=expert_count,
        active_experts=balanced_active,
        weight_elements_per_expert=weight_elements_per_expert,
        aux_bytes=aux_bytes,
    )
    extreme = _scenario_estimate(
        scenario="extreme_imbalance",
        spec=spec,
        config=config,
        dtype=dtype,
        output_dtype=output_dtype,
        input_dtypes=input_dtypes,
        quant_spec=quant_spec,
        expert_count=expert_count,
        active_experts=extreme_active,
        weight_elements_per_expert=weight_elements_per_expert,
        aux_bytes=aux_bytes,
    )
    return GroupedMatmulEstimate(
        expert_count=expert_count,
        weight_elements_per_expert=weight_elements_per_expert,
        weight_bytes_per_expert=weight_bytes_per_expert,
        balanced=balanced,
        extreme=extreme,
    )
