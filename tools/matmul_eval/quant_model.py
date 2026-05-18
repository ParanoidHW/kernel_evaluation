from __future__ import annotations

from typing import Any

from .common import *

def is_256b_aligned(inner_size: int, dtype_size_value: int) -> bool:
    return (inner_size * dtype_size_value) % 256 == 0


def is_nd2nz_on_the_way_supported(tensor_format: str, inner_size: int, dtype_size_value: int) -> bool:
    # From MatMulV3 SUPPORT_ND2NZ_GM2L0. These are byte lengths, not element counts.
    supported_bytes = {32, 64, 96, 128, 160, 192, 224, 256, 384}
    return tensor_format == "ND" and inner_size * dtype_size_value in supported_bytes


def need_nd2nz_for_operand(
    tensor_format: str,
    inner_size: int,
    outer_size: int,
    dtype: str,
    dtype_size_value: int,
) -> bool:
    if tensor_format != "ND" or dtype_size_value <= 0:
        return False

    support_on_the_way = is_nd2nz_on_the_way_supported(tensor_format, inner_size, dtype_size_value)
    inner_aligned = is_256b_aligned(inner_size, dtype_size_value)
    normal_nd2nz = (
        (not inner_aligned or inner_size > 65535)
        and not support_on_the_way
        and not (is_fp32_dtype(dtype) and inner_size < 65535)
    )

    will_fit_vnchw = (
        outer_size > 8192
        and inner_size > 1
        and (
            inner_size * dtype_size_value <= 192
            or (inner_size * dtype_size_value <= 384 and inner_size % 2 == 0)
            or (inner_size * dtype_size_value <= 512 and inner_size % 4 == 0)
        )
    )
    inner_equals_c0 = inner_size == 32 // dtype_size_value
    vnchw_nd2nz = will_fit_vnchw and not inner_aligned and not support_on_the_way and not inner_equals_c0
    return normal_nd2nz or vnchw_nd2nz


def infer_nd2nz_operands(spec: MatmulSpec, dtype: str) -> tuple[bool, bool]:
    elem_size = dtype_size(dtype)
    inner_a = spec.m if spec.trans_a else spec.k
    outer_a = spec.k if spec.trans_a else spec.m
    inner_b = spec.k if spec.trans_b else spec.n
    outer_b = spec.n if spec.trans_b else spec.k
    nd2nz_a = need_nd2nz_for_operand(spec.a_format, inner_a, outer_a, dtype, elem_size)
    nd2nz_b = need_nd2nz_for_operand(spec.b_format, inner_b, outer_b, dtype, elem_size)
    return nd2nz_a, nd2nz_b


def quant_storage_bytes(elements: int, dtype: str) -> int:
    bitwidth = dtype_bitwidth(dtype)
    if bitwidth < 8:
        return ceil_div(elements * bitwidth, 8)
    return elements * dtype_size(dtype)


def infer_quant_spec(
    row: dict[str, str],
    spec: MatmulSpec,
    input_shapes: list[list[int]],
) -> QuantSpec:
    kernel_type = row.get("Type", "")
    input_dtypes = split_semicolon_values(row.get("Input Data Types"))
    output_dtype = output_dtype_from_row(row)
    if not is_quant_kernel_type(kernel_type) and not any(is_quantized_data_dtype(dtype) for dtype in input_dtypes[:2]):
        return QuantSpec(is_quant=False)

    a_dtype = input_dtypes[0] if input_dtypes else "UNKNOWN"
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else "UNKNOWN"
    aux_shapes = input_shapes[2:] if len(input_shapes) > 2 else []
    aux_dtypes = input_dtypes[2:] if len(input_dtypes) > 2 else []
    aux_elements = sum(num_elements(shape) for shape in aux_shapes)
    aux_bytes = 0
    for shape, dtype in zip(aux_shapes, aux_dtypes):
        aux_bytes += num_elements(shape) * dtype_size(dtype)

    quant_data_inputs = sum(1 for dtype in (a_dtype, b_dtype) if is_quantized_data_dtype(dtype))
    has_scale = any(is_scale_dtype(dtype) for dtype in aux_dtypes)
    input_type_names = [dtype.upper().replace("DT_", "") for dtype in input_dtypes]
    min_data_bits = min(dtype_bitwidth(a_dtype), dtype_bitwidth(b_dtype))
    if any(dtype in {"MXFP8", "HIFLOAT8", "HIF8"} for dtype in input_type_names):
        mode = "mxfp8"
    elif any(dtype in {"FLOAT8", "FLOAT8_E4M3FN", "FLOAT8_E5M2", "FP8"} for dtype in input_type_names):
        mode = "fp8"
    elif any(dtype in {"FLOAT4", "FLOAT4_E2M1"} for dtype in input_type_names):
        mode = "float4"
    else:
        mode = f"int{min_data_bits}"

    kernel_type_lower = kernel_type.lower()
    if "weightquant" in kernel_type_lower and quant_data_inputs >= 1 and has_scale:
        compute_path = "weight_only_quant_with_dequant"
    elif "weightquant" in kernel_type_lower and quant_data_inputs >= 1:
        compute_path = "weight_only_quant"
    elif quant_data_inputs == 2 and has_scale and output_dtype in {"FLOAT16", "DT_FLOAT16", "DT_BF16", "BFLOAT16", "FLOAT", "DT_FLOAT"}:
        compute_path = "full_quant_with_dequant"
    elif quant_data_inputs == 2:
        compute_path = "full_quant"
    else:
        compute_path = "fake_quant_or_mixed"

    granularity = "unknown"
    notes: list[str] = []
    is_qbm_v3 = "quantbatchmatmulv3" in kernel_type_lower or "quantmatmulv3" in kernel_type_lower
    if is_qbm_v3:
        scale_shape = aux_shapes[0] if aux_shapes else []
        pertoken_shape = aux_shapes[3] if len(aux_shapes) >= 4 else []
        scale_is_channel = len(scale_shape) == 1 and scale_shape[0] in {1, spec.n}
        pertoken_is_m = len(pertoken_shape) == 1 and pertoken_shape[0] in {spec.m, spec.batch * spec.m}
        if pertoken_is_m and scale_is_channel:
            granularity = "per_token_per_channel"
        elif pertoken_is_m:
            granularity = "per_token_m"
        elif scale_is_channel:
            granularity = "per_channel_n" if scale_shape[0] == spec.n else "per_tensor"
        if granularity != "unknown":
            notes.append("qbm_v3_scale_offset_order")

    for shape in aux_shapes:
        if is_qbm_v3 and granularity != "unknown":
            break
        if len(shape) == 1:
            dim = shape[0]
            if dim == spec.n and dim != spec.m:
                granularity = "per_channel_n"
            elif dim == spec.m and dim != spec.n:
                granularity = "per_token_m"
            elif dim == spec.n and dim == spec.m:
                granularity = "per_channel_n_or_per_token_m"
                notes.append("scale_shape_equals_m_and_n")
            elif dim == 1:
                granularity = "per_tensor"
            elif spec.n % dim == 0 or spec.m % dim == 0:
                granularity = "per_group_or_block"
            else:
                granularity = "vector_scale_unknown_axis"
        elif len(shape) >= 2:
            if shape[-1] == spec.n and shape[-2] in {1, spec.m}:
                granularity = "per_token_per_channel"
            elif shape[-1] != spec.n and spec.n % shape[-1] == 0:
                granularity = "per_group"
            else:
                granularity = "tensor_scale"

    if not aux_shapes:
        granularity = "none"
    if has_scale and granularity == "unknown":
        notes.append("scale_dtype_present_but_granularity_unknown")
    if compute_path.startswith("full_quant") and has_scale:
        notes.append("int_accumulate_then_dequant")
    if compute_path.startswith("weight_only_quant"):
        notes.append("weight_dequant_on_matmul_path")

    deduped_notes = list(dict.fromkeys(notes))
    return QuantSpec(
        is_quant=True,
        mode=mode,
        granularity=granularity,
        compute_path=compute_path,
        aux_elements=aux_elements,
        aux_bytes=aux_bytes,
        notes="|".join(deduped_notes),
    )


def estimate_quant_cost(
    spec: MatmulSpec,
    tile: TileEstimate,
    quant_spec: QuantSpec,
    input_shapes: list[list[int]],
    input_dtypes: list[str],
    output_dtype: str,
    config: dict[str, Any],
) -> tuple[float | None, float, float, int, int]:
    quant_cfg = config.get("quant_matmul", {})
    output_elements = spec.output_storage_elements or spec.batch * spec.m * spec.n
    a_elements = spec.a_storage_elements or spec.batch * spec.m * spec.k
    b_elements = spec.b_storage_elements or spec.batch * spec.k * spec.n
    a_dtype = input_dtypes[0] if input_dtypes else "UNKNOWN"
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else "UNKNOWN"
    out_size = dtype_size(output_dtype)

    aux_bytes = quant_spec.aux_bytes
    quant_gm_bytes_min = (
        quant_storage_bytes(a_elements, a_dtype)
        + quant_storage_bytes(b_elements, b_dtype)
        + aux_bytes
        + output_elements * out_size
    )

    scale_replay = float(quant_cfg.get("scale_replay_factor", 1.0))
    quant_gm_bytes_tiled = int(
        quant_storage_bytes(a_elements, a_dtype) * tile.tile_n
        + quant_storage_bytes(b_elements, b_dtype) * tile.tile_m
        + aux_bytes * scale_replay
        + output_elements * out_size
    )
    l2_bytes = int(config.get("l2_bytes", 0))
    if l2_bytes > 0 and quant_gm_bytes_min <= l2_bytes:
        quant_gm_bytes_tiled = quant_gm_bytes_min
    elif l2_bytes > 0:
        redundant = max(0, quant_gm_bytes_tiled - quant_gm_bytes_min)
        l2_pressure = max(0.0, 1.0 - l2_bytes / max(quant_gm_bytes_min, 1))
        quant_gm_bytes_tiled = int(quant_gm_bytes_min + redundant * l2_pressure)

    hbm_us = quant_gm_bytes_tiled / (float(config["hbm_bandwidth_tbps"]) * 1_000_000.0)

    peak_tops = peak_for_dtype(config, a_dtype)
    if peak_tops is None:
        peak_tops = quant_cfg.get("peak_tops", {}).get(a_dtype)
    if peak_tops is None:
        peak_tops = quant_cfg.get("peak_tops", {}).get(a_dtype.replace("DT_", ""))

    compute_us: float | None = None
    if peak_tops is not None:
        efficiency = float(quant_cfg.get("pipeline_efficiency", {}).get(a_dtype, quant_cfg.get("pipeline_efficiency", {}).get("default", 1.0)))
        efficiency = max(efficiency, 1e-9)
        op_factor = float(quant_cfg.get("operation_factor", {}).get(quant_spec.compute_path, 1.0))
        compute_us = tile.aligned_flops * op_factor / (float(peak_tops) * 1_000_000.0 * tile.core_efficiency * efficiency)

    dequant_per_elem_us = float(quant_cfg.get("dequant_us_per_output_element", 0.0))
    dequant_us = (
        output_elements * dequant_per_elem_us
        if quant_spec.compute_path in {"full_quant_with_dequant", "weight_only_quant_with_dequant"}
        else 0.0
    )
    if compute_us is not None:
        compute_us += dequant_us

    return compute_us, hbm_us, dequant_us, quant_gm_bytes_min, quant_gm_bytes_tiled


