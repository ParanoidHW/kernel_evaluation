from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from op_eval.common import dtype_size, num_elements


MATMUL_TYPES = {
    "matmul",
    "matmulv2",
    "matmulv3",
    "batchmatmul",
    "batchmatmulv2",
    "batchmatmulv3",
    "transposebatchmatmul",
    "quantbatchmatmulv3",
    "groupedmatmul",
}

ATTENTION_TOKENS = (
    "attention",
    "attentionscore",
    "flashattentionscore",
    "fusedinferattentionscore",
    "promptflashattention",
    "increflashattention",
    "pagedattention",
    "kvquantsparseflashattention",
)

COMMUNICATION_TOKENS = (
    "hcom",
    "allreduce",
    "allgather",
    "alltoall",
    "reducescatter",
    "broadcastaicpukernel",
    "allreduceaicpukernel",
    "allgathermatmul",
)

LAYOUT_MEMORY_TYPES = {
    "cast",
    "transpose",
    "transdata",
    "tensormove",
    "slice",
    "stridedsliced",
    "asstrided",
    "concatd",
    "concatv2d",
    "splitvd",
    "pack",
    "tile",
    "memset",
}

ELEMENTWISE_TYPES = {
    "add",
    "mul",
    "sub",
    "neg",
    "realdiv",
    "pows",
    "greaterequal",
    "greater",
    "less",
    "equal",
    "zeroslike",
    "oneslike",
    "fill",
    "muls",
    "sigmoid",
    "broadcastto",
    "selectv2",
    "clipbyvaluev2",
    "cos",
    "sin",
}

REDUCTION_TYPES = {"reducesum", "reducesumd", "reducemean", "reduceall", "softmaxv2"}

NORM_ACTIVATION_TYPES = {
    "rmsnorm",
    "layernormv3",
    "add_rmsnorm",
    "addrmsnorm",
    "inplaceaddrmsnorm",
    "addrmsnormcast",
    "swish",
    "gelu",
    "swiglu",
    "gegluv2",
    "dequantswigluquant",
    "groupnormsilu",
}

INDEX_SCATTER_TYPES = {
    "gatherv2",
    "gatherv3",
    "scatter",
    "scatterupdate",
    "scatterndupdate",
    "topkv2",
    "argmaxv2",
    "index",
    "moegatingtopksoftmax",
    "moegatingtopk",
    "moegatingtopkhash",
    "moeinitrouting",
    "moeinitroutingv3",
    "moecomputerexperttokens",
    "moefinalizeroutingv2",
    "moererouting",
    "moedistributedispatchv2",
    "moedistributecombinev2",
}

CV_TYPES = {
    "conv3dv2",
    "resizebicubicv2",
    "resizebilinearv2",
    "resizenearestneighborv2",
    "gridsample",
    "gridsample2d",
    "gridsample3d",
    "roialign",
    "nmswithmask",
    "nonmaxsuppressionv3",
    "nonmaxsuppressionv6",
}


@dataclass(frozen=True)
class OtherOpSpec:
    op_type: str
    op_family: str
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]
    input_dtypes: list[str]
    output_dtypes: list[str]
    input_formats: list[str]
    output_formats: list[str]
    input_elements: list[int]
    output_elements: list[int]
    input_bytes: list[int]
    output_bytes: list[int]
    logical_elements: int
    source_repo: str
    source_path: str
    source_strategy: str
    layout_pattern: str
    missing_attrs: str = ""


def normalize_type(op_type: str) -> str:
    return op_type.replace("_", "").replace("-", "").lower()


def is_other_ops_row(row: dict[str, Any]) -> bool:
    op_type = str(row.get("Type", ""))
    text = f"{row.get('Name', '')} {op_type}".lower()
    normalized = normalize_type(op_type)
    if normalized in MATMUL_TYPES:
        return False
    if any(token.lower() in normalized for token in ATTENTION_TOKENS):
        return False
    if any(token in text for token in COMMUNICATION_TOKENS):
        return False
    return bool(op_type)


LAYOUT_SOURCE_PATHS = {
    "cast": "ops-math/math/cast",
    "tensormove": "ops-math/conversion/tensor_move",
    "transdata": "ops-math/conversion/trans_data",
    "transpose": "ops-math/conversion/transpose",
    "slice": "ops-math/conversion/slice",
    "stridedsliced": "ops-math/conversion/strided_slice",
    "asstrided": "ops-math/conversion/as_strided",
    "concatd": "ops-math/conversion/concat",
    "concatv2d": "ops-math/conversion/concat",
    "splitvd": "ops-math/conversion/split",
    "pack": "ops-math/conversion/pack",
    "tile": "ops-math/math/tile",
    "memset": "ops-math/conversion/mem_set",
}

ELEMENTWISE_SOURCE_PATHS = {
    "broadcastto": "ops-math/conversion/broadcast_to",
    "clipbyvaluev2": "ops-math/conversion/clip_by_value_v2",
    "fill": "ops-math/conversion/fill",
    "zeroslike": "ops-math/conversion/zeros_like",
    "oneslike": "ops-math/math/ones_like",
    "realdiv": "ops-math/math/real_div",
    "selectv2": "ops-math/math/select_v2",
    "cos": "ops-math/math/cos",
    "sin": "ops-math/math/sin",
}

REDUCTION_SOURCE_PATHS = {
    "reducesum": "ops-math/math/reduce_sum",
    "reducesumd": "ops-math/math/reduce_sum",
    "reducemean": "ops-math/math/reduce_mean",
    "reduceall": "ops-math/math/reduce_all",
    "softmaxv2": "ops-nn/activation/softmax_v2",
}

NORM_ACTIVATION_SOURCE_PATHS = {
    "rmsnorm": "ops-nn/norm/rms_norm",
    "layernormv3": "ops-nn/norm/layer_norm_v3",
    "add_rmsnorm": "ops-nn/norm/add_rms_norm",
    "addrmsnorm": "ops-nn/norm/add_rms_norm",
    "inplaceaddrmsnorm": "ops-nn/norm/inplace_add_rms_norm",
    "addrmsnormcast": "ops-nn/norm/add_rms_norm_cast",
    "swish": "ops-nn/activation/swish",
    "gelu": "ops-nn/activation/gelu",
    "swiglu": "ops-nn/activation/swiglu",
    "gegluv2": "ops-nn/activation/geglu_v2",
    "dequantswigluquant": "ops-nn/activation/swiglu",
    "groupnormsilu": "ops-nn/norm/group_norm_silu",
}


def classify_op_family(op_type: str) -> tuple[str, str, str]:
    normalized = normalize_type(op_type)
    if normalized in LAYOUT_MEMORY_TYPES:
        return "layout_memory", "ops-math", LAYOUT_SOURCE_PATHS.get(normalized, "ops-math/conversion")
    if normalized in ELEMENTWISE_TYPES:
        return "elementwise_vector", "ops-math", ELEMENTWISE_SOURCE_PATHS.get(normalized, "ops-math/math")
    if normalized in REDUCTION_TYPES:
        repo = "ops-nn" if normalized == "softmaxv2" else "ops-math"
        return "reduction", repo, REDUCTION_SOURCE_PATHS.get(normalized, "ops-math/math")
    if normalized in NORM_ACTIVATION_TYPES:
        return "norm_activation", "ops-nn", NORM_ACTIVATION_SOURCE_PATHS.get(normalized, "ops-nn/norm_or_activation")
    if normalized in INDEX_SCATTER_TYPES:
        return "index_scatter_routing", "ops-nn/ops-transformer", "ops-nn/index_or_ops-transformer/moe"
    if normalized in CV_TYPES:
        return "cv_regular", "ops-cv", "ops-cv/image_or_objdetect"
    return "unsupported_other", "", ""


def build_spec(
    op_type: str,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    input_dtypes: list[str],
    output_dtypes: list[str],
    input_formats: list[str],
    output_formats: list[str],
) -> OtherOpSpec | None:
    family, source_repo, source_path = classify_op_family(op_type)
    if family == "unsupported_other":
        return None
    input_elements = [num_elements(shape) for shape in input_shapes]
    output_elements = [num_elements(shape) for shape in output_shapes]
    if input_dtypes and len(input_dtypes) > len(input_elements):
        for _ in range(len(input_dtypes) - len(input_elements)):
            input_elements.append(1)
    if not input_elements and not output_elements:
        return None
    input_bytes = [
        elems * dtype_size(input_dtypes[index] if index < len(input_dtypes) else "")
        for index, elems in enumerate(input_elements)
    ]
    output_bytes = [
        elems * dtype_size(output_dtypes[index] if index < len(output_dtypes) else "")
        for index, elems in enumerate(output_elements)
    ]
    logical_elements = max(output_elements or [0], default=0)
    if logical_elements == 0:
        logical_elements = max(input_elements or [0], default=0)
    missing_attrs = infer_missing_attrs(op_type, family)
    source_strategy = infer_source_strategy(
        op_type,
        family,
        input_elements,
        logical_elements,
        input_formats,
        output_formats,
        missing_attrs,
    )
    layout_pattern = infer_layout_pattern(op_type, family, input_shapes, output_shapes, input_formats, output_formats)
    return OtherOpSpec(
        op_type=op_type,
        op_family=family,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        input_dtypes=input_dtypes,
        output_dtypes=output_dtypes,
        input_formats=input_formats,
        output_formats=output_formats,
        input_elements=input_elements,
        output_elements=output_elements,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        logical_elements=logical_elements,
        source_repo=source_repo,
        source_path=source_path,
        source_strategy=source_strategy,
        layout_pattern=layout_pattern,
        missing_attrs=missing_attrs,
    )


def infer_missing_attrs(op_type: str, family: str) -> str:
    normalized = normalize_type(op_type)
    missing: list[str] = []
    if normalized == "transpose":
        missing.append("perm")
    if normalized in {"slice", "stridedsliced"}:
        missing.extend(["begin", "size_or_end", "stride"])
    if normalized in {"concatd", "concatv2d", "splitvd", "pack"}:
        missing.append("axis")
    if normalized == "tile":
        missing.append("multiples")
    if normalized == "memset":
        missing.append("fill_value")
    if normalized == "asstrided":
        missing.extend(["size", "stride", "storage_offset"])
    if family == "index_scatter_routing":
        missing.append("indices_or_routing_values")
    return "|".join(missing)


def infer_source_strategy(
    op_type: str,
    family: str,
    input_elements: list[int],
    logical_elements: int,
    input_formats: list[str],
    output_formats: list[str],
    missing_attrs: str,
) -> str:
    normalized = normalize_type(op_type)
    if family == "layout_memory":
        if normalized == "cast":
            return "linear_ub_cast"
        if normalized == "tensormove":
            return "linear_ub_copy"
        if normalized == "transdata":
            formats = {fmt.upper() for fmt in input_formats + output_formats}
            if any("FRACTAL" in fmt or "NZ" in fmt for fmt in formats):
                return "format_transform_nz_nd_simt"
            if any("5HD" in fmt or "C1HWC0" in fmt for fmt in formats):
                return "format_transform_5hd_simt"
            return "format_transform_simt"
        if normalized == "transpose":
            return "transpose_nddma_vconv_missing_perm" if missing_attrs else "transpose_nddma_vconv"
        if normalized in {"slice", "stridedsliced"}:
            return "slice_move_align_or_nddma_missing_offsets" if missing_attrs else "slice_move_align_or_nddma"
        if normalized == "asstrided":
            return "as_strided_gather_or_move_align_missing_stride" if missing_attrs else "as_strided_gather_or_move_align"
        if normalized in {"concatd", "concatv2d"}:
            return "concat_axis_strategy_missing_axis" if missing_attrs else "concat_axis_strategy"
        if normalized == "splitvd":
            return "split_axis_strategy_missing_axis" if missing_attrs else "split_axis_strategy"
        if normalized == "pack":
            return "pack_to_concat_missing_axis" if missing_attrs else "pack_to_concat"
        if normalized == "tile":
            return "tile_broadcast_copy_missing_multiples" if missing_attrs else "tile_broadcast_copy"
        if normalized == "memset":
            return "memset_output_fill"
    if family == "elementwise_vector":
        if normalized in {"cos", "sin", "sigmoid"}:
            return "elementwise_transcendental_vector_pipeline"
        if normalized in {"pows", "pow", "realdiv"}:
            return "elementwise_expensive_math_vector_pipeline"
        if normalized in {"zeroslike", "oneslike", "fill"}:
            return "elementwise_fill_vector_pipeline"
        if input_elements and any(elems == 1 for elems in input_elements) and logical_elements > 1:
            return "elementwise_scalar_broadcast_vector_pipeline"
        if input_elements and any(elems not in {1, logical_elements} for elems in input_elements):
            return "elementwise_broadcast_vector_pipeline"
        return "elementwise_vector_pipeline"
    if family == "reduction":
        if normalized == "softmaxv2":
            return "softmax_reduce_exp_sum_normalize"
        if normalized == "reducemean":
            return "reduce_tree_with_scale"
        return "reduce_tree"
    if family == "norm_activation":
        if normalized in {"swish", "gelu", "swiglu", "gegluv2", "dequantswigluquant"}:
            return "activation_vector_pipeline"
        if normalized in {"add_rmsnorm", "addrmsnorm", "inplaceaddrmsnorm", "addrmsnormcast"}:
            return "rmsnorm_residual_fusion"
        if normalized == "rmsnorm":
            return "rmsnorm_reduce_scale"
        if normalized == "layernormv3":
            return "layernorm_mean_var_scale"
        if normalized == "groupnormsilu":
            return "groupnorm_reduce_silu"
        return "norm_activation_source_pending"
    if family == "index_scatter_routing":
        return "index_scatter_missing_runtime_values"
    if family == "cv_regular":
        return "cv_source_pending"
    return "unsupported"


def infer_layout_pattern(
    op_type: str,
    family: str,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    input_formats: list[str],
    output_formats: list[str],
) -> str:
    if family != "layout_memory":
        return ""
    normalized = normalize_type(op_type)
    if normalized in {"cast", "tensormove"}:
        return "linear"
    if normalized == "transdata":
        return "format_change" if input_formats != output_formats else "format_preserve"
    if normalized == "transpose":
        return "rank_permutation"
    if normalized in {"slice", "stridedsliced", "asstrided"}:
        return "strided_region"
    if normalized in {"concatd", "concatv2d", "splitvd", "pack"}:
        return "axis_segment"
    if normalized == "tile":
        return "broadcast_tile"
    if normalized == "memset":
        return "output_fill"
    if input_shapes and output_shapes and input_shapes[0] == output_shapes[0]:
        return "shape_preserve"
    return "layout_transform"
