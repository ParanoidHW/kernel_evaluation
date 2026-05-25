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
}

ELEMENTWISE_TYPES = {
    "add",
    "mul",
    "sub",
    "neg",
    "realdiv",
    "pows",
    "greaterequal",
    "less",
    "zeroslike",
    "oneslike",
    "fill",
    "muls",
    "sigmoid",
    "broadcastto",
    "selectv2",
    "clipbyvaluev2",
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


def classify_op_family(op_type: str) -> tuple[str, str, str]:
    normalized = normalize_type(op_type)
    if normalized in LAYOUT_MEMORY_TYPES:
        return "layout_memory", "ops-math", "ops-math/conversion"
    if normalized in ELEMENTWISE_TYPES:
        return "elementwise_vector", "ops-math", "ops-math/math"
    if normalized in REDUCTION_TYPES:
        return "reduction", "ops-math", "ops-math/math"
    if normalized in NORM_ACTIVATION_TYPES:
        return "norm_activation", "ops-nn", "ops-nn/norm_or_activation"
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
    if normalized == "asstrided":
        missing.extend(["size", "stride", "storage_offset"])
    if family == "index_scatter_routing":
        missing.append("indices_or_routing_values")
    return "|".join(missing)
