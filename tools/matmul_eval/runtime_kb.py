from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import *

def decode_tiling_enable(value: int | None) -> dict[str, int | None]:
    if value is None or value < 0:
        return {
            "split_core": None,
            "full_load": None,
            "fix_opti": None,
            "special_opti": None,
        }
    return {
        "split_core": value % 10,
        "full_load": (value // 10) % 10,
        "fix_opti": (value // 1000) % 10,
        "special_opti": (value // 10000) % 10,
    }


def tiling_full_load_name(value: int | None) -> str:
    return {0: "NONE_FULL_LOAD", 1: "A_FULL_LOAD", 2: "B_FULL_LOAD"}.get(value, "")


def runtime_kb_dtype_code(dtype: str) -> int | None:
    normalized = dtype.upper().replace("DT_", "")
    candidates = [dtype.upper(), normalized]
    if normalized == "BF16":
        candidates.extend(["BFLOAT16", "DT_BF16"])
    for candidate in candidates:
        if candidate in GE_DTYPE_RUNTIME_KB:
            return GE_DTYPE_RUNTIME_KB[candidate]
    return None


def runtime_kb_format_code(tensor_format: str) -> int | None:
    return GE_FORMAT_RUNTIME_KB.get(tensor_format)


def runtime_kb_aligned_spec(spec: MatmulSpec, dtype: str) -> tuple[int, int, int, bool, bool, bool]:
    reduce_align = 8 if is_fp32_dtype(dtype) else 16
    aligned_m = ceil_align(spec.m, 16)
    aligned_n = ceil_align(spec.n, 16)
    aligned_k = ceil_align(spec.k, reduce_align)
    return aligned_m, aligned_n, aligned_k, spec.m == aligned_m, spec.n == aligned_n, spec.k == aligned_k


def runtime_kb_key_from_parts(
    a_dtype_code: int,
    b_dtype_code: int,
    out_dtype_code: int,
    a_format_code: int,
    b_format_code: int,
    out_format_code: int,
    m: int,
    n: int,
    k: int,
    m_align: bool,
    n_align: bool,
    k_align: bool,
    trans_a: bool,
    trans_b: bool,
    bias: bool,
) -> tuple[Any, ...]:
    return (
        a_dtype_code,
        b_dtype_code,
        out_dtype_code,
        a_format_code,
        b_format_code,
        out_format_code,
        m,
        n,
        k,
        bool(m_align),
        bool(n_align),
        bool(k_align),
        bool(trans_a),
        bool(trans_b),
        bool(bias),
    )


def runtime_kb_key_from_entry(info: dict[str, Any]) -> tuple[Any, ...]:
    return runtime_kb_key_from_parts(
        int(info.get("a_dtype", -1)),
        int(info.get("b_dtype", -1)),
        int(info.get("out_dtype", -1)),
        int(info.get("a_format", -1)),
        int(info.get("b_format", -1)),
        int(info.get("out_format", -1)),
        int(info.get("m", -1)),
        int(info.get("n", -1)),
        int(info.get("k", -1)),
        bool(info.get("m_align_flag", False)),
        bool(info.get("n_align_flag", False)),
        bool(info.get("k_align_flag", False)),
        bool(info.get("trans_a_flag", False)),
        bool(info.get("trans_b_flag", False)),
        bool(info.get("bias_flag", False)),
    )


def runtime_kb_key_from_row(
    spec: MatmulSpec,
    input_dtypes: list[str],
    output_dtype: str,
) -> tuple[Any, ...] | None:
    if spec.batch != 1:
        return None
    a_dtype = input_dtypes[0] if input_dtypes else "UNKNOWN"
    b_dtype = input_dtypes[1] if len(input_dtypes) > 1 else a_dtype
    a_dtype_code = runtime_kb_dtype_code(a_dtype)
    b_dtype_code = runtime_kb_dtype_code(b_dtype)
    out_dtype_code = runtime_kb_dtype_code(output_dtype)
    a_format_code = runtime_kb_format_code(spec.a_format)
    b_format_code = runtime_kb_format_code(spec.b_format)
    out_format_code = runtime_kb_format_code(spec.output_format)
    if None in {a_dtype_code, b_dtype_code, out_dtype_code, a_format_code, b_format_code, out_format_code}:
        return None
    aligned_m, aligned_n, aligned_k, m_align, n_align, k_align = runtime_kb_aligned_spec(spec, a_dtype)
    bias = len(input_dtypes) >= 3 and is_scale_dtype(input_dtypes[2])
    return runtime_kb_key_from_parts(
        a_dtype_code,
        b_dtype_code,
        out_dtype_code,
        a_format_code,
        b_format_code,
        out_format_code,
        aligned_m,
        aligned_n,
        aligned_k,
        m_align,
        n_align,
        k_align,
        spec.trans_a,
        spec.trans_b,
        bias,
    )


def expand_config_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if any(char in pattern for char in "*?[]"):
            paths.extend(sorted(Path.cwd().glob(pattern)))
        elif pattern_path.exists():
            paths.append(pattern_path)
    return sorted(set(paths))


def load_runtime_kb(config: dict[str, Any]) -> dict[tuple[Any, ...], list[RuntimeKbEntry]]:
    kernel_model = config.get("kernel_model", {})
    runtime_cfg = kernel_model.get("runtime_kb", {})
    if not runtime_cfg.get("enabled", False):
        return {}

    index: dict[tuple[Any, ...], list[RuntimeKbEntry]] = {}
    for path in expand_config_paths(list(runtime_cfg.get("matmul_v3_paths", []))):
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                info = record.get("info_dict", {})
                knowledge = record.get("knowledge", {})
                key = runtime_kb_key_from_entry(info)
                entry = RuntimeKbEntry(
                    source_file=display_path(path),
                    entry_id=str(record.get("id", f"{path.name}:{line_no}")),
                    key=key,
                    info=info,
                    knowledge=knowledge,
                )
                index.setdefault(key, []).append(entry)
    return index


