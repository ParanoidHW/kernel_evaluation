from __future__ import annotations

import math
from typing import Any

from .common import *
from .runtime_kb import decode_tiling_enable, runtime_kb_key_from_row, tiling_full_load_name

def candidate_base_values(limit: int) -> list[int]:
    max_value = max(16, min(512, ceil_align(limit, 16)))
    values = list(range(16, max_value + 1, 16))
    preferred = [64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 336]
    return sorted(set(values + [value for value in preferred if value <= max_value]))


def source_base_values(limit: int, max_base: int = ADV_BASIC_BLOCK_256, align: int = ADV_BASIC_BLOCK_16) -> list[int]:
    align = max(ADV_BASIC_BLOCK_16, align)
    max_value = max(ADV_BASIC_BLOCK_16, min(max_base, ceil_align(limit, ADV_BASIC_BLOCK_16)))
    values = [value for value in range(align, max_value + 1, align) if value % ADV_BASIC_BLOCK_16 == 0]
    if ADV_BASIC_BLOCK_16 not in values:
        values.append(ADV_BASIC_BLOCK_16)
    return sorted(set(values))


def storage_element_counts(spec: MatmulSpec) -> tuple[int, int, int]:
    logical_a = spec.batch * spec.m * spec.k
    logical_b = spec.batch * spec.k * spec.n
    logical_c = spec.batch * spec.m * spec.n
    return (
        spec.a_storage_elements or logical_a,
        spec.b_storage_elements or logical_b,
        spec.output_storage_elements or logical_c,
    )


def l2_aware_gm_bytes(gm_bytes_min: int, gm_bytes_tiled_raw: int, config: dict[str, Any]) -> int:
    l2_bytes = int(config.get("l2_bytes", 0))
    if l2_bytes > 0 and gm_bytes_min <= l2_bytes:
        return gm_bytes_min
    if l2_bytes > 0:
        redundant = max(0, gm_bytes_tiled_raw - gm_bytes_min)
        l2_pressure = max(0.0, 1.0 - l2_bytes / max(gm_bytes_min, 1))
        return int(gm_bytes_min + redundant * l2_pressure)
    return gm_bytes_tiled_raw


def asw_window_len(aic_num: int) -> int:
    sqrt_num = int(math.sqrt(max(aic_num, 1)))
    for factor in range(sqrt_num, 0, -1):
        if aic_num % factor == 0:
            return factor
    return 1


def advanced_base_k(
    spec: MatmulSpec,
    elem_size: int,
    l0a: int,
    base_m: int,
    base_n: int,
) -> int:
    k_value_align = ceil_align(spec.k, ADV_BASIC_BLOCK_16)
    max_base_k = l0a // ADV_DB_SIZE // max(elem_size, 1) // max(base_m, base_n, 1)
    if k_value_align <= max_base_k:
        return max(ADV_BASIC_BLOCK_16, k_value_align)
    if spec.trans_a and not spec.trans_b:
        return max(ADV_BASIC_BLOCK_16, floor_align(max_base_k, ADV_BASIC_BLOCK_16))
    align_256b = max(ADV_BASIC_BLOCK_16, ADV_BASIC_BLOCK_K_256_BYTE // max(elem_size, 1))
    if max_base_k * elem_size >= ADV_BASIC_BLOCK_K_256_BYTE:
        return max(ADV_BASIC_BLOCK_16, floor_align(max_base_k, align_256b))
    for candidate in (128, 64, 32, 16):
        if max_base_k >= candidate:
            return candidate
    return ADV_BASIC_BLOCK_16


def advanced_cal_l1_tiling(
    spec: MatmulSpec,
    elem_size: int,
    l1: int,
    base_m: int,
    base_n: int,
    base_k: int,
    single_core_k: int,
) -> tuple[int, int, int, int, int, int]:
    depth_a1 = max(1, l1 // ADV_DB_SIZE // max(base_m, 1) // max(base_k, 1) // max(elem_size, 1))
    depth_b1 = max(1, l1 // ADV_DB_SIZE // max(base_n, 1) // max(base_k, 1) // max(elem_size, 1))
    depth_a_size = depth_a1 * base_m * base_k * elem_size
    depth_b_size = depth_b1 * base_n * base_k * elem_size
    if depth_a_size + depth_b_size > l1:
        if base_m <= base_n:
            depth_a1 = max(depth_a1 // ADV_DB_SIZE, 1)
        else:
            depth_b1 = max(depth_b1 // ADV_DB_SIZE, 1)

    step_ka = max(depth_a1 // ADV_DB_SIZE, 1)
    step_kb = max(depth_b1 // ADV_DB_SIZE, 1)
    if (
        base_m == ADV_BASIC_BLOCK_256
        and base_n == ADV_BASIC_BLOCK_256
        and spec.m % ADV_BASIC_BLOCK_16 == 0
        and spec.n % ADV_BASIC_BLOCK_16 == 0
        and spec.k % ADV_BASIC_BLOCK_16 == 0
        and single_core_k <= ADV_BASIC_BLOCK_256
    ):
        step_ka = min(step_ka, 2)
        step_kb = min(step_kb, 2)

    if step_ka >= step_kb:
        step_ka = max((step_ka // max(step_kb, 1)) * step_kb, 1)
    else:
        step_kb = max((step_kb // max(step_ka, 1)) * step_ka, 1)
    return step_ka * ADV_DB_SIZE, step_kb * ADV_DB_SIZE, 1, 1, step_ka, step_kb


def advanced_step_small_k(
    spec: MatmulSpec,
    dtype: str,
    elem_size: int,
    base_k: int,
    step_ka: int,
    step_kb: int,
    is_bl1_full_load: bool,
) -> int:
    step_big_k = step_kb if is_bl1_full_load else step_ka
    step_small_k = step_ka if is_bl1_full_load else step_kb
    is_trans = spec.trans_a if is_bl1_full_load else spec.trans_b
    step_small_k = max(step_small_k, 1)
    is_small_tail = (step_big_k % step_small_k) / step_small_k <= 0.25
    is_small_tail = (is_small_tail and not is_trans) or base_k * elem_size >= ADV_BASIC_BLOCK_K_256_BYTE
    if is_fp32_dtype(dtype):
        return 1
    if is_small_tail:
        return 2
    return step_small_k


def advanced_l0c2out(
    spec: MatmulSpec,
    output_dtype: str,
    aic_num: int,
    aiv_num: int,
    single_core_m: int,
    single_core_n: int,
    dtype: str,
) -> str:
    is_valid_mkn = spec.k <= ADV_BASIC_BLOCK_256 and spec.m >= ADV_BASIC_BLOCK_256
    m_cnt = ceil_div(spec.m, max(single_core_m, 1))
    n_cnt = ceil_div(spec.n, max(single_core_n, 1))
    is_multi_round = m_cnt * n_cnt >= 2 * aic_num
    c_size = dtype_size(output_dtype)
    is_unaligned_n = spec.n * c_size % 128 != 0 and spec.n * c_size > ADV_BASIC_BLOCK_256
    fixpipe_bound = is_valid_mkn and is_multi_round and is_unaligned_n
    if not fixpipe_bound or aiv_num != aic_num * 2:
        return "ON_THE_FLY"
    if dtype in {"FLOAT16", "DT_FLOAT16", "DT_BF16", "BFLOAT16"}:
        return "ND_FIXPIPE_1_1"
    return "ND_FIXPIPE_1_2"


def advanced_stream_k_kind(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
    kernel_type: str,
) -> str | None:
    aic_num = int(config["aic_num"])
    aiv_num = int(config.get("aiv_num", aic_num * 2))
    elem_size = dtype_size(dtype)
    if aiv_num != aic_num * 2 or spec.a_format != "ND":
        return None

    align_value = ADV_BLOCK_BYTE_SIZE if is_fp32_dtype(dtype) else ADV_BASIC_BLOCK_256
    k_threshold_sk = max(ADV_STREAM_K_MIN_K_THRESHOLD, aic_num * ADV_BASIC_BLOCK_K_256_BYTE) // max(elem_size, 1)
    k_large_enough = ceil_align(spec.k, ADV_BASIC_BLOCK_256) >= k_threshold_sk

    if is_batch_matmul_kernel_type(kernel_type):
        if is_fp32_dtype(dtype) and spec.k > 2_000_000:
            return None
        if not k_large_enough:
            return None
        m_cnt = ceil_div(spec.m, align_value)
        n_cnt = ceil_div(spec.n, align_value)
        if spec.batch * m_cnt * n_cnt <= max(1, aic_num // 2):
            return "batch_stream_k_sk"
        return None

    if k_large_enough:
        m_cnt = ceil_div(spec.m, align_value)
        n_cnt = ceil_div(spec.n, align_value)
        if m_cnt * n_cnt <= max(1, aic_num // 2):
            return "stream_k_sk"

    k_threshold_dpsk = max(ADV_STREAM_K_MIN_K_THRESHOLD, aic_num * ADV_BASIC_BLOCK_K_128_BYTE) // max(elem_size, 1)
    if spec.m % ADV_BASIC_BLOCK_256 == 0 and spec.n % ADV_BASIC_BLOCK_256 == 0 and spec.k >= k_threshold_dpsk:
        m_cnt = ceil_div(spec.m, ADV_BASIC_BLOCK_256)
        n_cnt = ceil_div(spec.n, ADV_BASIC_BLOCK_256)
        total_mn = m_cnt * n_cnt
        remainder = total_mn % aic_num
        if total_mn >= aic_num and remainder != 0 and remainder <= aic_num // 2:
            return "stream_k_dpsk"
    return None


def advanced_stream_k_tile(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    kernel_type: str,
    kind: str,
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    base_m = ADV_BASIC_BLOCK_256
    base_n = ADV_BASIC_BLOCK_256
    m_cnt = ceil_div(spec.m, base_m)
    n_cnt = ceil_div(spec.n, base_n)

    if kind == "batch_stream_k_sk":
        blocks_per_batch = max(1, aic_num // max(spec.batch, 1))
        if m_cnt > blocks_per_batch // 3 and m_cnt < blocks_per_batch // 2:
            m_cnt = max(1, blocks_per_batch // 2)
        if n_cnt > blocks_per_batch // 3 and n_cnt < blocks_per_batch // 2:
            n_cnt = max(1, blocks_per_batch // 2)
        mn_per_batch = max(1, m_cnt * n_cnt)
        k_cnt = max(1, blocks_per_batch // mn_per_batch)
    else:
        total_mn = max(1, m_cnt * n_cnt)
        if total_mn <= max(1, aic_num // 2):
            if m_cnt > aic_num // 3 and m_cnt < aic_num // 2:
                m_cnt = max(1, aic_num // 2)
            if n_cnt > aic_num // 3 and n_cnt < aic_num // 2:
                n_cnt = max(1, aic_num // 2)
            total_mn = max(1, m_cnt * n_cnt)
            k_cnt = max(1, aic_num // total_mn)
        else:
            remainder = max(1, total_mn % aic_num)
            k_cnt = max(1, aic_num // remainder)

    base_m = ceil_align(ceil_div(spec.m, max(m_cnt, 1)), ADV_BASIC_BLOCK_16)
    base_n = ceil_align(ceil_div(spec.n, max(n_cnt, 1)), ADV_BASIC_BLOCK_16)
    single_core_m = base_m
    single_core_n = base_n
    single_core_k = ceil_div(spec.k, max(k_cnt, 1))
    if spec.b_format != "ND":
        k_align = ADV_BASIC_BLOCK_16 if elem_size == 2 or (elem_size == 4 and not spec.trans_b) else ADV_BASIC_BLOCK_16 // 2
        single_core_k = ceil_align(single_core_k, max(k_align, 1))

    base_k_align = ADV_BASIC_BLOCK_128 // max(elem_size, 1) if (not spec.trans_a or spec.trans_b) else ADV_BASIC_BLOCK_16
    k_value_max = floor_align(
        int(config["l0a_bytes"]) // ADV_DB_SIZE // max(elem_size, 1) // max(base_m, base_n, 1),
        max(base_k_align, 1),
    )
    base_k = max(ADV_BASIC_BLOCK_16, min(single_core_k, max(k_value_max, ADV_BASIC_BLOCK_16)))
    depth_a1, depth_b1, step_m, step_n, step_ka, step_kb = advanced_cal_l1_tiling(
        spec, elem_size, int(config["l1_bytes"]), base_m, base_n, base_k, single_core_k
    )
    if base_m == base_n and depth_b1 == depth_a1 * 2:
        depth_a1 = depth_a1 * 2
        depth_b1 = max(1, depth_b1 // 2)
        step_kb = max(1, depth_b1 // ADV_DB_SIZE)
        step_ka = max(1, depth_a1 // ADV_DB_SIZE)

    return make_tile_estimate_from_source_tiling(
        spec=spec,
        dtype=dtype,
        output_dtype=output_dtype,
        config=config,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        single_core_m=single_core_m,
        single_core_n=single_core_n,
        single_core_k=single_core_k,
        used_core_num=min(aic_num, max(1, spec.batch * m_cnt * n_cnt * k_cnt)),
        core_work_tiles=max(1, spec.batch * m_cnt * n_cnt * k_cnt),
        source="advanced_tiling_heuristic",
        tiling_strategy=kind,
        full_load="NONE_FULL_LOAD",
        l0c2out=advanced_l0c2out(
            spec, output_dtype, aic_num, int(config.get("aiv_num", aic_num * 2)), base_m, base_n, dtype
        ),
        asw_window=asw_window_len(aic_num),
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        step_m=step_m,
        step_n=step_n,
        step_ka=step_ka,
        step_kb=step_kb,
        l1_buffer_num=ADV_DB_SIZE,
        ub_db=ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 <= int(config.get("ub_bytes", 0)) else 1,
        tiling_split_core=0,
        tiling_full_load=0,
    )


def advanced_balance_rate_with_tail(
    spec: MatmulSpec,
    used_core_num: int,
    base_m: int,
    base_n: int,
) -> float:
    total_round = spec.batch * ceil_div(spec.m, base_m) * ceil_div(spec.n, base_n)
    if total_round <= 0 or used_core_num <= 0:
        return 0.0
    main_round = ceil_div(total_round, used_core_num) - 1
    tail_blocks = total_round - used_core_num * main_round
    if tail_blocks <= 0:
        return 1.0
    if main_round == 0 or (base_m * base_n) // tail_blocks < ADV_MIN_TAIL_BLOCK_SIZE or spec.batch != 1:
        return (spec.batch * spec.m * spec.n / used_core_num) / ((main_round + 1) * base_m * base_n)
    tail_split_sqrt = max(1, int(math.sqrt(tail_blocks)))
    offset = (tail_blocks - tail_split_sqrt * tail_split_sqrt) // tail_split_sqrt + 1
    tail_round = 1.0 / (tail_split_sqrt * (tail_split_sqrt + offset - 1))
    return (spec.m * spec.n / used_core_num) / ((main_round + tail_round) * base_m * base_n)


def advanced_max_base_with_limit(
    spec: MatmulSpec,
    elem_size: int,
    config: dict[str, Any],
    base_mn_buffer_limit: int,
    base_align_unit: int,
    is_right_matrix: bool,
    is_memory_bound: bool,
) -> int:
    shape_value = spec.n if is_right_matrix else spec.m
    k_align_value = ceil_align(spec.k, ADV_BASIC_BLOCK_16)
    k_limit_value = ADV_BASIC_BLOCK_16 if is_memory_bound else ADV_BASIC_BLOCK_K_128_BYTE // max(elem_size, 1)
    min_k_l0_bytes = min(k_limit_value, k_align_value) * elem_size
    l0_size = int(config["l0b_bytes"] if is_right_matrix else config["l0a_bytes"])
    max_base_mn_with_buffer = base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // ADV_BASIC_BLOCK_16
    max_base_block = min(
        l0_size // ADV_DB_SIZE // max(min_k_l0_bytes, 1),
        max_base_mn_with_buffer,
    )
    k_align_unit = (
        (ADV_BASIC_BLOCK_K_256_BYTE if is_memory_bound and spec.batch == 1 else ADV_BASIC_BLOCK_K_256_BYTE * 2)
        // max(elem_size, 1)
        if (not spec.trans_a or spec.trans_b)
        else ADV_BASIC_BLOCK_16
    )
    max_base_mn_with_k_inner = int(config["l1_bytes"]) // (
        2 * ADV_DB_SIZE * max(elem_size, 1) * max(1, min(k_align_unit, k_align_value))
    )
    max_base_block = min(max_base_block, max_base_mn_with_k_inner)
    max_base_block = min(ceil_align(shape_value, base_align_unit), floor_align(max_base_block, base_align_unit))
    return max(ADV_BASIC_BLOCK_16, max_base_block)


def advanced_batch_rebalance_base(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
) -> tuple[int, int, int, int, float]:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    base_mn_buffer_limit = int(config["l0c_bytes"])
    core_freq_ghz = float(config.get("kernel_model", {}).get("advanced_tiling", {}).get("core_freq_ghz", 1.65))
    l2_rate = float(config.get("kernel_model", {}).get("advanced_tiling", {}).get("l2_rate", 100.0))
    hbm_bw = float(config["hbm_bandwidth_tbps"])
    l2_bw = core_freq_ghz * aic_num * l2_rate / 1024.0
    compute_power = core_freq_ghz * 8.0 * aic_num
    cmr = (spec.m + spec.n) / max(spec.m * spec.n, 1)
    l2_cache_usage = max(spec.batch * (spec.m + spec.n) * spec.k * elem_size / max(int(config.get("l2_bytes", 1)), 1), 1.0)
    cube_bound_edge = (
        (l2_bw / max(compute_power, 1e-9))
        + l2_cache_usage * (1 - l2_bw / max(hbm_bw, 1e-9)) * cmr
        - (1 + l2_bw / max(hbm_bw, 1e-9)) / max(spec.k, 1)
    )

    base_m_best = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
    base_n_best = max(
        ADV_BASIC_BLOCK_16,
        min(
            ceil_align(spec.n, ADV_BASIC_BLOCK_16),
            floor_align(base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // max(base_m_best, 1), ADV_BASIC_BLOCK_16),
        ),
    )
    is_memory_bound = (1.0 / base_m_best + 1.0 / base_n_best) > cube_bound_edge
    inner_align_unit = ADV_BASIC_BLOCK_128 if is_memory_bound else ADV_BASIC_BLOCK_64
    fixp_bound_edge = (spec.m * spec.n * hbm_bw) / max((spec.m + spec.n) * l2_bw, 1e-9)
    base_m_align_unit = inner_align_unit // max(elem_size, 1) if spec.trans_a else ADV_BASIC_BLOCK_16
    base_n_align_unit = (
        ADV_BASIC_BLOCK_K_256_BYTE // max(elem_size, 1)
        if spec.k < fixp_bound_edge
        else (ADV_BASIC_BLOCK_16 if spec.trans_b else inner_align_unit // max(elem_size, 1))
    )
    base_m_align_unit = max(ADV_BASIC_BLOCK_16, base_m_align_unit)
    base_n_align_unit = max(ADV_BASIC_BLOCK_16, base_n_align_unit)
    max_base_m = advanced_max_base_with_limit(
        spec, elem_size, config, base_mn_buffer_limit, base_m_align_unit, False, is_memory_bound
    )
    max_base_n = advanced_max_base_with_limit(
        spec, elem_size, config, base_mn_buffer_limit, base_n_align_unit, True, is_memory_bound
    )

    best_base_m = max(ADV_BASIC_BLOCK_16, min(max_base_m, ADV_BASIC_BLOCK_256))
    best_base_n = max(
        ADV_BASIC_BLOCK_16,
        min(max_base_n, floor_align(base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // best_base_m, base_n_align_unit)),
    )
    best_cube_param = 1.0 / best_base_m + 1.0 / best_base_n
    cube_bound_edge *= ADV_CUBE_BOUND_RATIO
    best_balance = advanced_balance_rate_with_tail(spec, aic_num, best_base_m, best_base_n)

    for cur_base_m in range(max_base_m, 0, -base_m_align_unit):
        cur_base_m = floor_align(cur_base_m, base_m_align_unit)
        if cur_base_m < ADV_BASIC_BLOCK_16:
            continue
        cur_max_base_n = min(max_base_n, floor_align(base_mn_buffer_limit // ADV_DATA_SIZE_FP32 // cur_base_m, base_n_align_unit))
        for cur_base_n in range(cur_max_base_n, 0, -base_n_align_unit):
            cur_base_n = floor_align(cur_base_n, base_n_align_unit)
            if cur_base_n < ADV_BASIC_BLOCK_16:
                continue
            cur_cube_param = 1.0 / cur_base_m + 1.0 / cur_base_n
            cur_balance = advanced_balance_rate_with_tail(spec, aic_num, cur_base_m, cur_base_n)
            if best_balance >= 0.9 and cur_cube_param > best_cube_param and cur_cube_param > cube_bound_edge:
                continue
            cube_bound_cond = cur_cube_param <= cube_bound_edge and cur_balance > best_balance
            current_score = cur_cube_param / max(cur_balance, 1e-9)
            best_score = best_cube_param / max(best_balance, 1e-9)
            balance_cond = current_score < best_score or (abs(current_score - best_score) < 1e-9 and cur_balance > best_balance)
            if cube_bound_cond or balance_cond:
                best_base_m = cur_base_m
                best_base_n = cur_base_n
                best_cube_param = cur_cube_param
                best_balance = cur_balance

    best_base_m = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), best_base_m)
    best_base_n = min(ceil_align(spec.n, ADV_BASIC_BLOCK_16), best_base_n)
    base_k = advanced_base_k(spec, elem_size, int(config["l0a_bytes"]), best_base_m, best_base_n)
    m_core = ceil_div(spec.m, best_base_m)
    n_core = ceil_div(spec.n, best_base_n)
    used_core = min(spec.batch * m_core * n_core, aic_num)
    return best_base_m, best_base_n, base_k, used_core, best_balance


def advanced_matmul_base(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    if ceil_div(spec.m, ADV_BASIC_BLOCK_256) * ceil_div(spec.n, ADV_BASIC_BLOCK_256) >= aic_num:
        base_m = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
        base_n = min(ceil_align(spec.n, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
    else:
        best: tuple[float, int, int] | None = None
        for base_m_candidate in source_base_values(spec.m):
            for base_n_candidate in source_base_values(spec.n):
                if base_m_candidate * base_n_candidate * ADV_DATA_SIZE_FP32 > int(config["l0c_bytes"]):
                    continue
                m_tiles = ceil_div(spec.m, base_m_candidate)
                n_tiles = ceil_div(spec.n, base_n_candidate)
                mn_tiles = m_tiles * n_tiles
                rounds = ceil_div(mn_tiles, aic_num)
                core_eff = mn_tiles / max(rounds * aic_num, 1)
                redundant = 1.0 / base_m_candidate + 1.0 / base_n_candidate
                tail_waste = (m_tiles * base_m_candidate * n_tiles * base_n_candidate) / max(spec.m * spec.n, 1)
                score = (1.0 / max(core_eff, 1e-9)) + 0.35 * redundant * ADV_BASIC_BLOCK_256 + 0.15 * tail_waste
                if best is None or score < best[0]:
                    best = (score, base_m_candidate, base_n_candidate)
        if best is None:
            base_m = min(ceil_align(spec.m, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
            base_n = min(ceil_align(spec.n, ADV_BASIC_BLOCK_16), ADV_BASIC_BLOCK_256)
        else:
            _, base_m, base_n = best
    base_k = advanced_base_k(spec, elem_size, int(config["l0a_bytes"]), base_m, base_n)
    used_core = min(aic_num, ceil_div(spec.m, base_m) * ceil_div(spec.n, base_n))
    return base_m, base_n, base_k, used_core


def apply_advanced_full_load(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    base_m: int,
    base_n: int,
    base_k: int,
    depth_a1: int,
    depth_b1: int,
    step_ka: int,
    step_kb: int,
) -> tuple[int, int, int, int, int, int, int, int, str]:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    l1 = int(config["l1_bytes"])
    l0a = int(config["l0a_bytes"])
    l0b = int(config["l0b_bytes"])
    l0c = int(config["l0c_bytes"])
    asw_window = asw_window_len(aic_num)
    l0c2out = advanced_l0c2out(
        spec, output_dtype, aic_num, int(config.get("aiv_num", aic_num * 2)), base_m, base_n, dtype
    )
    m_cnt = ceil_div(spec.m, base_m)
    n_cnt = ceil_div(spec.n, base_n)
    is_single_round = m_cnt * n_cnt <= aic_num

    full_load = "NONE_FULL_LOAD"
    single_core_m = base_m
    single_core_n = base_n
    step_m = 1
    step_n = 1

    m_aligned = ceil_align(spec.m, ADV_BASIC_BLOCK_16)
    n_aligned = ceil_align(spec.n, ADV_BASIC_BLOCK_16)
    k_aligned_a = ceil_align(spec.k, ADV_BASIC_BLOCK_16 if spec.trans_a else max(1, ADV_BLOCK_BYTE_SIZE // elem_size))
    k_aligned_b = ceil_align(spec.k, max(1, ADV_BLOCK_BYTE_SIZE // elem_size) if spec.trans_b else ADV_BASIC_BLOCK_16)
    max_step = max(1, asw_window - 1)

    al1_size = k_aligned_a * m_aligned * elem_size
    a_l1_full_mte2 = spec.m * aic_num + spec.n * m_cnt
    base_mte2_for_a = spec.m * n_cnt + spec.n * m_cnt
    al1_ok = (
        l0c2out == "ON_THE_FLY"
        and spec.n >= ADV_CACHELINE
        and not is_single_round
        and spec.m < max_step * base_m
        and al1_size <= l1 * 3 // 4
        and not (spec.m > ADV_BASIC_BLOCK_256 and base_mte2_for_a < 1.2 * a_l1_full_mte2)
    )

    bl1_size = k_aligned_b * n_aligned * elem_size
    b_l1_full_mte2 = spec.n * aic_num + spec.m * n_cnt
    base_mte2_for_b = spec.m * n_cnt + spec.n * m_cnt
    bl1_ok = (
        spec.m >= ADV_CACHELINE
        and not is_single_round
        and spec.n < max_step * base_n
        and bl1_size <= l1 * 3 // 4
        and not (spec.n > ADV_BASIC_BLOCK_256 and base_mte2_for_b < 1.2 * b_l1_full_mte2)
    )

    if al1_ok:
        if m_aligned * base_k * elem_size * ADV_DB_SIZE <= l0a:
            base_m = m_aligned
        else:
            base_m = min(m_aligned, base_m)
        step_m = ceil_div(spec.m, base_m)
        step_ka = ceil_div(spec.k, base_k)
        step_kb = advanced_step_small_k(spec, dtype, elem_size, base_k, step_ka, step_kb, False)
        if ceil_div(spec.n, base_n) < aic_num:
            base_n = max(ADV_BASIC_BLOCK_16, ceil_align(ceil_div(spec.n, aic_num), ADV_BASIC_BLOCK_16))
        depth_b1 = ADV_DB_SIZE * step_kb
        depth_a1 = step_m * step_ka
        a_l1_size = ceil_align(spec.k, ADV_BASIC_BLOCK_16) * m_aligned * elem_size
        b_l1_load_size = base_k * depth_b1 * base_n * elem_size
        while base_n > ADV_BASIC_BLOCK_16 and b_l1_load_size > l1 - a_l1_size:
            if step_kb == min(step_ka, 2):
                base_n = ceil_align(max(ADV_BASIC_BLOCK_16, base_n >> 1), ADV_BASIC_BLOCK_16)
            step_kb = min(step_ka, 2)
            depth_b1 = ADV_DB_SIZE * step_kb
            b_l1_load_size = depth_b1 * base_n * base_k * elem_size
        single_core_m = spec.m
        single_core_n = base_n
        db_l0c = ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 * ADV_DB_SIZE <= l0c else 1
        if (
            (spec.trans_b and (base_n * elem_size) % ADV_BASIC_BLOCK_K_256_BYTE == 0)
            or (base_n * elem_size) % (ADV_BASIC_BLOCK_K_256_BYTE * 2) == 0
        ) and db_l0c <= 1:
            base_n = ceil_align(max(ADV_BASIC_BLOCK_16, base_n >> 1), ADV_BASIC_BLOCK_16)
            single_core_n = base_n
        full_load = "A_FULL_LOAD"
    elif bl1_ok:
        if n_aligned * base_k * elem_size * ADV_DB_SIZE <= l0b:
            base_n = n_aligned
        else:
            base_n = min(n_aligned, base_n)
        step_n = ceil_div(spec.n, base_n)
        step_kb = ceil_div(spec.k, base_k)
        step_ka = advanced_step_small_k(spec, dtype, elem_size, base_k, step_ka, step_kb, True)
        if ceil_div(spec.m, base_m) < aic_num:
            base_m = max(ADV_BASIC_BLOCK_16, ceil_align(ceil_div(spec.m, aic_num), ADV_BASIC_BLOCK_16))
        depth_a1 = ADV_DB_SIZE * step_ka
        depth_b1 = step_n * step_kb
        b_l1_size = ceil_align(spec.k, ADV_BASIC_BLOCK_16) * n_aligned * elem_size
        a_l1_load_size = base_k * depth_a1 * base_m * elem_size
        while base_m > ADV_BASIC_BLOCK_16 and a_l1_load_size > l1 - b_l1_size:
            if step_ka == min(step_kb, 2):
                base_m = ceil_align(max(ADV_BASIC_BLOCK_16, base_m >> 1), ADV_BASIC_BLOCK_16)
            step_ka = min(step_kb, 2)
            depth_a1 = ADV_DB_SIZE * step_ka
            a_l1_load_size = depth_a1 * base_m * base_k * elem_size
        single_core_n = spec.n
        single_core_m = base_m
        if (not spec.trans_a or (base_m * elem_size) % (ADV_BASIC_BLOCK_K_256_BYTE * 2) == 0) and (
            base_m * base_n * ADV_DATA_SIZE_FP32 * ADV_DB_SIZE > l0c
        ):
            base_m = ceil_align(max(ADV_BASIC_BLOCK_16, base_m >> 1), ADV_BASIC_BLOCK_16)
            single_core_m = base_m
        full_load = "B_FULL_LOAD"

    return base_m, base_n, single_core_m, single_core_n, depth_a1, depth_b1, step_m, step_n, full_load


def make_tile_estimate_from_source_tiling(
    *,
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    base_m: int,
    base_n: int,
    base_k: int,
    single_core_m: int,
    single_core_n: int,
    single_core_k: int,
    used_core_num: int,
    core_work_tiles: int,
    source: str,
    tiling_strategy: str,
    full_load: str,
    l0c2out: str,
    asw_window: int,
    depth_a1: int,
    depth_b1: int,
    step_m: int,
    step_n: int,
    step_ka: int,
    step_kb: int,
    l1_buffer_num: int,
    ub_db: int,
    tiling_split_core: int,
    tiling_full_load: int,
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    peak_tflops = peak_for_dtype(config, dtype)
    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    a_storage, b_storage, c_storage = storage_element_counts(spec)
    gm_bytes_min = a_storage * elem_size + b_storage * elem_size + c_storage * out_size

    tile_m = max(1, ceil_div(spec.m, max(single_core_m, 1)))
    tile_n = max(1, ceil_div(spec.n, max(single_core_n, 1)))
    tile_k = max(1, ceil_div(spec.k, max(single_core_k, 1)))
    mn_tile_count = tile_m * tile_n * spec.batch
    tile_count = mn_tile_count * tile_k
    rounds = max(1, ceil_div(max(core_work_tiles, mn_tile_count), aic_num))
    core_eff = min(1.0, max(core_work_tiles, mn_tile_count) / max(rounds * aic_num, 1))
    aligned_flops = (
        2
        * tile_m
        * single_core_m
        * tile_n
        * single_core_n
        * tile_k
        * single_core_k
        * spec.batch
    )
    tail_eff = true_flops / aligned_flops if aligned_flops else 0.0
    gm_bytes_tiled_raw = tile_n * a_storage * elem_size + tile_m * b_storage * elem_size + c_storage * out_size
    if "stream_k" in tiling_strategy and tile_k > 1:
        stream_k_factor = float(
            config.get("kernel_model", {}).get("advanced_tiling", {}).get("stream_k_reduction_traffic_factor", 0.25)
        )
        gm_bytes_tiled_raw += int((tile_k - 1) * c_storage * ADV_DATA_SIZE_FP32 * stream_k_factor)
    gm_bytes_tiled = l2_aware_gm_bytes(gm_bytes_min, gm_bytes_tiled_raw, config)
    hbm_us = gm_bytes_tiled / (float(config["hbm_bandwidth_tbps"]) * 1_000_000.0)
    compute_us = None
    if peak_tflops is not None:
        compute_us = aligned_flops / (peak_tflops * 1_000_000.0 * max(core_eff, 1e-9))
    lower_bound_us = max(value for value in (compute_us, hbm_us) if value is not None)
    return TileEstimate(
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        db_l0c=ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 * ADV_DB_SIZE <= int(config["l0c_bytes"]) else 1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        mn_tile_count=mn_tile_count,
        tile_count=tile_count,
        used_core_num=min(aic_num, max(1, used_core_num)),
        core_efficiency=core_eff,
        tail_efficiency=tail_eff,
        aligned_flops=aligned_flops,
        gm_bytes_min=gm_bytes_min,
        gm_bytes_tiled_raw=gm_bytes_tiled_raw,
        gm_bytes_tiled=gm_bytes_tiled,
        compute_us=compute_us,
        hbm_us=hbm_us,
        lower_bound_us=lower_bound_us,
        source=source,
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        step_m=step_m,
        step_n=step_n,
        step_ka=step_ka,
        step_kb=step_kb,
        tiling_strategy=tiling_strategy,
        full_load=full_load,
        l0c2out=l0c2out,
        asw_window_len=asw_window,
        l1_buffer_num=l1_buffer_num,
        ub_db=ub_db,
        tiling_split_core=tiling_split_core,
        tiling_full_load=tiling_full_load,
        tiling_fix_opti=0,
        tiling_special_opti=0,
    )


def estimate_tile_from_advanced_tiling(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    kernel_type: str,
) -> TileEstimate:
    stream_k = advanced_stream_k_kind(spec, dtype, config, kernel_type)
    if stream_k is not None:
        return advanced_stream_k_tile(spec, dtype, output_dtype, config, kernel_type, stream_k)

    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    if is_batch_matmul_kernel_type(kernel_type):
        base_m, base_n, base_k, used_core, _ = advanced_batch_rebalance_base(spec, dtype, config)
        tiling_strategy = "batch_asw_basic_rebalance"
    else:
        base_m, base_n, base_k, used_core = advanced_matmul_base(spec, dtype, config)
        tiling_strategy = "basic_aswt"

    single_core_k = spec.k
    depth_a1, depth_b1, step_m, step_n, step_ka, step_kb = advanced_cal_l1_tiling(
        spec, elem_size, int(config["l1_bytes"]), base_m, base_n, base_k, single_core_k
    )
    full_load = "NONE_FULL_LOAD"
    if not is_batch_matmul_kernel_type(kernel_type):
        base_m, base_n, single_core_m, single_core_n, depth_a1, depth_b1, step_m, step_n, full_load = apply_advanced_full_load(
            spec, dtype, output_dtype, config, base_m, base_n, base_k, depth_a1, depth_b1, step_ka, step_kb
        )
        if full_load == "A_FULL_LOAD":
            tiling_strategy = "basic_aswt_al1_full_load"
            step_ka = max(1, depth_a1 // max(step_m, 1))
            step_kb = max(1, depth_b1 // ADV_DB_SIZE)
        elif full_load == "B_FULL_LOAD":
            tiling_strategy = "basic_aswt_bl1_full_load"
            step_ka = max(1, depth_a1 // ADV_DB_SIZE)
            step_kb = max(1, depth_b1 // max(step_n, 1))
        else:
            single_core_m = base_m
            single_core_n = base_n
    else:
        single_core_m = base_m
        single_core_n = base_n

    l0c2out = advanced_l0c2out(
        spec, output_dtype, aic_num, int(config.get("aiv_num", aic_num * 2)), single_core_m, single_core_n, dtype
    )
    core_work_tiles = max(1, spec.batch * ceil_div(spec.m, max(single_core_m, 1)) * ceil_div(spec.n, max(single_core_n, 1)))
    tiling_full_load = {"NONE_FULL_LOAD": 0, "A_FULL_LOAD": 1, "B_FULL_LOAD": 2}.get(full_load, 0)
    l1_tensor_size = base_k * max(step_ka, step_kb) * (base_m + base_n) * elem_size
    l1_buffer_num = ADV_BASIC_L1_BUFFER_NUM if l1_tensor_size * ADV_BASIC_L1_BUFFER_NUM <= int(config["l1_bytes"]) else ADV_DB_SIZE
    return make_tile_estimate_from_source_tiling(
        spec=spec,
        dtype=dtype,
        output_dtype=output_dtype,
        config=config,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        single_core_m=single_core_m,
        single_core_n=single_core_n,
        single_core_k=single_core_k,
        used_core_num=used_core,
        core_work_tiles=core_work_tiles,
        source="advanced_tiling_heuristic",
        tiling_strategy=tiling_strategy,
        full_load=full_load,
        l0c2out=l0c2out,
        asw_window=asw_window_len(aic_num),
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        step_m=step_m,
        step_n=step_n,
        step_ka=step_ka,
        step_kb=step_kb,
        l1_buffer_num=l1_buffer_num,
        ub_db=ADV_DB_SIZE if base_m * base_n * ADV_DATA_SIZE_FP32 <= int(config.get("ub_bytes", 0)) else 1,
        tiling_split_core=0,
        tiling_full_load=tiling_full_load,
    )


def estimate_tile(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    l0a = int(config["l0a_bytes"])
    l0b = int(config["l0b_bytes"])
    l0c = int(config["l0c_bytes"])
    hbm_bandwidth_tbps = float(config["hbm_bandwidth_tbps"])
    peak_tflops = peak_for_dtype(config, dtype)

    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    logical_a_elements = spec.batch * spec.m * spec.k
    logical_b_elements = spec.batch * spec.k * spec.n
    logical_c_elements = spec.batch * spec.m * spec.n
    a_storage_elements = spec.a_storage_elements or logical_a_elements
    b_storage_elements = spec.b_storage_elements or logical_b_elements
    output_storage_elements = spec.output_storage_elements or logical_c_elements
    gm_bytes_min = (
        a_storage_elements * elem_size
        + b_storage_elements * elem_size
        + output_storage_elements * out_size
    )

    best: TileEstimate | None = None
    for base_m in candidate_base_values(spec.m):
        for base_n in candidate_base_values(spec.n):
            # L0C accumulates FP32 partial sums.
            if base_m * base_n * 4 > l0c:
                continue
            db_l0c = 2 if base_m * base_n * 4 * 2 <= l0c else 1

            max_k_a = floor_align(l0a // max(1, 2 * elem_size * base_m), 16)
            max_k_b = floor_align(l0b // max(1, 2 * elem_size * base_n), 16)
            max_base_k = min(max_k_a, max_k_b)
            if max_base_k < 16:
                continue

            base_k = min(ceil_align(spec.k, 16), max_base_k)
            base_k = max(16, floor_align(base_k, 16))

            tile_m = ceil_div(spec.m, base_m)
            tile_n = ceil_div(spec.n, base_n)
            tile_k = ceil_div(spec.k, base_k)
            mn_tile_count = tile_m * tile_n * spec.batch
            tile_count = mn_tile_count * tile_k
            used_core_num = min(aic_num, max(1, mn_tile_count))
            rounds = max(1, ceil_div(mn_tile_count, aic_num))
            core_eff = mn_tile_count / (rounds * aic_num)

            aligned_m = tile_m * base_m
            aligned_n = tile_n * base_n
            aligned_k = tile_k * base_k
            aligned_flops = 2 * aligned_m * aligned_n * aligned_k * spec.batch
            tail_eff = true_flops / aligned_flops if aligned_flops else 0.0

            # Raw repeated traffic if every tile reread hits GM. The effective
            # HBM estimate below is L2-aware, matching the kernel's L2 cache
            # decision path rather than pessimistically charging every repeat.
            gm_bytes_tiled_raw = (
                tile_n * a_storage_elements * elem_size
                + tile_m * b_storage_elements * elem_size
                + output_storage_elements * out_size
            )
            l2_bytes = int(config.get("l2_bytes", 0))
            if l2_bytes > 0 and gm_bytes_min <= l2_bytes:
                gm_bytes_tiled = gm_bytes_min
            elif l2_bytes > 0:
                redundant = max(0, gm_bytes_tiled_raw - gm_bytes_min)
                l2_pressure = max(0.0, 1.0 - l2_bytes / max(gm_bytes_min, 1))
                gm_bytes_tiled = int(gm_bytes_min + redundant * l2_pressure)
            else:
                gm_bytes_tiled = gm_bytes_tiled_raw
            hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1_000_000.0)
            if peak_tflops is None:
                compute_us = None
                lower_bound_us = hbm_us
            else:
                compute_us = aligned_flops / (peak_tflops * 1_000_000.0 * max(core_eff, 1e-9))
                lower_bound_us = max(compute_us, hbm_us)

            estimate = TileEstimate(
                base_m=base_m,
                base_n=base_n,
                base_k=base_k,
                db_l0c=db_l0c,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                mn_tile_count=mn_tile_count,
                tile_count=tile_count,
                used_core_num=used_core_num,
                core_efficiency=core_eff,
                tail_efficiency=tail_eff,
                aligned_flops=aligned_flops,
                gm_bytes_min=gm_bytes_min,
                gm_bytes_tiled_raw=gm_bytes_tiled_raw,
                gm_bytes_tiled=gm_bytes_tiled,
                compute_us=compute_us,
                hbm_us=hbm_us,
                lower_bound_us=lower_bound_us,
            )
            if best is None or estimate.lower_bound_us < best.lower_bound_us:
                best = estimate

    if best is None:
        # Fallback should be rare; it keeps the evaluator usable for odd shapes.
        aligned_m = ceil_align(spec.m, 16)
        aligned_n = ceil_align(spec.n, 16)
        aligned_k = ceil_align(spec.k, 16)
        aligned_flops = 2 * aligned_m * aligned_n * aligned_k * spec.batch
        hbm_us = gm_bytes_min / (hbm_bandwidth_tbps * 1_000_000.0)
        compute_us = None
        if peak_tflops is not None:
            compute_us = aligned_flops / (peak_tflops * 1_000_000.0)
        return TileEstimate(
            base_m=16,
            base_n=16,
            base_k=16,
            db_l0c=1,
            tile_m=ceil_div(spec.m, 16),
            tile_n=ceil_div(spec.n, 16),
            tile_k=ceil_div(spec.k, 16),
            mn_tile_count=ceil_div(spec.m, 16) * ceil_div(spec.n, 16) * spec.batch,
            tile_count=ceil_div(spec.m, 16) * ceil_div(spec.n, 16) * ceil_div(spec.k, 16) * spec.batch,
            used_core_num=min(aic_num, ceil_div(spec.m, 16) * ceil_div(spec.n, 16) * spec.batch),
            core_efficiency=1.0,
            tail_efficiency=(2 * spec.m * spec.n * spec.k * spec.batch) / aligned_flops,
            aligned_flops=aligned_flops,
            gm_bytes_min=gm_bytes_min,
            gm_bytes_tiled_raw=gm_bytes_min,
            gm_bytes_tiled=gm_bytes_min,
            compute_us=compute_us,
            hbm_us=hbm_us,
            lower_bound_us=max(value for value in (compute_us, hbm_us) if value is not None),
        )
    return best


def estimate_tile_from_runtime_kb(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    entry: RuntimeKbEntry,
) -> TileEstimate:
    aic_num = int(config["aic_num"])
    elem_size = dtype_size(dtype)
    out_size = dtype_size(output_dtype)
    hbm_bandwidth_tbps = float(config["hbm_bandwidth_tbps"])
    peak_tflops = peak_for_dtype(config, dtype)
    info = entry.info
    knowledge = entry.knowledge

    aligned_m = int(info["m"])
    aligned_n = int(info["n"])
    aligned_k = int(info["k"])
    base_m = max(1, int(knowledge.get("baseM", aligned_m or 1)))
    base_n = max(1, int(knowledge.get("baseN", aligned_n or 1)))
    base_k = max(1, int(knowledge.get("baseK", aligned_k or 1)))
    single_core_m = max(1, int(knowledge.get("singleCoreM", base_m)))
    single_core_n = max(1, int(knowledge.get("singleCoreN", base_n)))
    single_core_k = max(1, int(knowledge.get("singleCoreK", aligned_k or base_k)))

    tile_m = max(1, ceil_div(aligned_m, single_core_m))
    tile_n = max(1, ceil_div(aligned_n, single_core_n))
    tile_k = max(1, ceil_div(aligned_k, single_core_k))
    mn_tile_count = tile_m * tile_n * spec.batch
    tile_count = mn_tile_count * tile_k
    used_core_num = max(1, min(aic_num, int(knowledge.get("usedCoreNum", min(aic_num, mn_tile_count)))))
    core_eff = min(1.0, used_core_num / max(aic_num, 1))

    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    aligned_flops = 2 * tile_m * single_core_m * tile_n * single_core_n * tile_k * single_core_k * spec.batch
    tail_eff = true_flops / aligned_flops if aligned_flops else 0.0

    logical_a_elements = spec.batch * spec.m * spec.k
    logical_b_elements = spec.batch * spec.k * spec.n
    logical_c_elements = spec.batch * spec.m * spec.n
    a_storage_elements = spec.a_storage_elements or logical_a_elements
    b_storage_elements = spec.b_storage_elements or logical_b_elements
    output_storage_elements = spec.output_storage_elements or logical_c_elements
    gm_bytes_min = (
        a_storage_elements * elem_size
        + b_storage_elements * elem_size
        + output_storage_elements * out_size
    )
    gm_bytes_tiled_raw = (
        tile_n * a_storage_elements * elem_size
        + tile_m * b_storage_elements * elem_size
        + output_storage_elements * out_size
    )
    l2_bytes = int(config.get("l2_bytes", 0))
    if l2_bytes > 0 and gm_bytes_min <= l2_bytes:
        gm_bytes_tiled = gm_bytes_min
    elif l2_bytes > 0:
        redundant = max(0, gm_bytes_tiled_raw - gm_bytes_min)
        l2_pressure = max(0.0, 1.0 - l2_bytes / max(gm_bytes_min, 1))
        gm_bytes_tiled = int(gm_bytes_min + redundant * l2_pressure)
    else:
        gm_bytes_tiled = gm_bytes_tiled_raw

    hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1_000_000.0)
    compute_us = None
    if peak_tflops is not None:
        compute_us = aligned_flops / (peak_tflops * 1_000_000.0 * max(core_eff, 1e-9))
    lower_bound_us = max(value for value in (compute_us, hbm_us) if value is not None)
    tiling_enable = int(knowledge.get("tilingEnable", -1))
    decoded_tiling = decode_tiling_enable(tiling_enable)
    return TileEstimate(
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        db_l0c=int(knowledge.get("dbL0C", 1)),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        mn_tile_count=mn_tile_count,
        tile_count=tile_count,
        used_core_num=used_core_num,
        core_efficiency=core_eff,
        tail_efficiency=tail_eff,
        aligned_flops=aligned_flops,
        gm_bytes_min=gm_bytes_min,
        gm_bytes_tiled_raw=gm_bytes_tiled_raw,
        gm_bytes_tiled=gm_bytes_tiled,
        compute_us=compute_us,
        hbm_us=hbm_us,
        lower_bound_us=lower_bound_us,
        source="runtime_kb_exact",
        runtime_kb_id=entry.entry_id,
        runtime_kb_file=entry.source_file,
        tiling_enable=tiling_enable,
        depth_a1=int(knowledge.get("depthA1", 0)),
        depth_b1=int(knowledge.get("depthB1", 0)),
        step_m=int(knowledge.get("stepM", 0)),
        step_n=int(knowledge.get("stepN", 0)),
        step_ka=int(knowledge.get("stepKa", 0)),
        step_kb=int(knowledge.get("stepKb", 0)),
        l2_m_tile=int(knowledge.get("l2MTileCnt", 0)),
        l2_n_tile=int(knowledge.get("l2NTileCnt", 0)),
        tiling_strategy="runtime_kb",
        full_load=tiling_full_load_name(decoded_tiling["full_load"]),
        tiling_split_core=decoded_tiling["split_core"],
        tiling_full_load=decoded_tiling["full_load"],
        tiling_fix_opti=decoded_tiling["fix_opti"],
        tiling_special_opti=decoded_tiling["special_opti"],
    )


def advanced_tiling_notes(
    spec: MatmulSpec,
    dtype: str,
    config: dict[str, Any],
    kernel_type: str,
    tile: TileEstimate | None = None,
) -> str:
    kernel_model = config.get("kernel_model", {})
    advanced_cfg = kernel_model.get("advanced_tiling", {})
    if not advanced_cfg.get("enabled", False):
        return "disabled"
    if not is_ops_nn_v3_kernel_type(kernel_type):
        return "not_ops_nn_v3"

    notes: list[str] = ["advanced_soc"]
    if tile is not None:
        if tile.tiling_strategy:
            notes.append(tile.tiling_strategy)
        if tile.full_load and tile.full_load != "NONE_FULL_LOAD":
            notes.append(tile.full_load.lower())
        if tile.l0c2out and tile.l0c2out != "ON_THE_FLY":
            notes.append(tile.l0c2out.lower())
    aic_num = int(config.get("aic_num", 1))
    m_cnt_128 = ceil_div(spec.m, 128)
    n_cnt_128 = ceil_div(spec.n, 128)
    mn_cnt = m_cnt_128 * n_cnt_128 * spec.batch
    if advanced_stream_k_kind(spec, dtype, config, kernel_type) is not None:
        notes.append("stream_k_capable_by_shape")
    elif spec.k >= aic_num * 384 and mn_cnt < max(1, aic_num // 2) and not (not spec.trans_a and spec.trans_b):
        notes.append("multi_core_splitk_candidate")
    if aic_num == 20:
        notes.append("aic20_l2_conflict_factor")
    elif aic_num == 24:
        notes.append("aic24_factor_table")
    if spec.a_format == "FRACTAL_NZ" or spec.b_format == "FRACTAL_NZ":
        notes.append("nz_layout")
    return "|".join(notes)


def select_tile_estimate(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
    runtime_kb: dict[tuple[Any, ...], list[RuntimeKbEntry]],
    input_dtypes: list[str],
    kernel_type: str,
) -> TileEstimate:
    use_ops_nn_v3_model = is_ops_nn_v3_kernel_type(kernel_type)
    key = runtime_kb_key_from_row(spec, input_dtypes, output_dtype) if use_ops_nn_v3_model else None
    if key is not None and not is_batch_matmul_kernel_type(kernel_type) and key in runtime_kb:
        entries = runtime_kb[key]
        estimates = [estimate_tile_from_runtime_kb(spec, dtype, output_dtype, config, entry) for entry in entries]
        return min(estimates, key=lambda estimate: estimate.lower_bound_us)

    if use_ops_nn_v3_model and config.get("kernel_model", {}).get("advanced_tiling", {}).get("enabled", False):
        return estimate_tile_from_advanced_tiling(spec, dtype, output_dtype, config, kernel_type)

    estimate = estimate_tile(spec, dtype, output_dtype, config)
    return estimate


def ideal_kernel_bounds(
    spec: MatmulSpec,
    dtype: str,
    output_dtype: str,
    config: dict[str, Any],
) -> tuple[float | None, float, float, int]:
    peak_tflops = peak_for_dtype(config, dtype)
    true_flops = 2 * spec.m * spec.n * spec.k * spec.batch
    compute_us = None
    if peak_tflops is not None:
        compute_us = true_flops / (peak_tflops * 1_000_000.0)
    logical_a_elements = spec.batch * spec.m * spec.k
    logical_b_elements = spec.batch * spec.k * spec.n
    logical_c_elements = spec.batch * spec.m * spec.n
    a_storage_elements = spec.a_storage_elements or logical_a_elements
    b_storage_elements = spec.b_storage_elements or logical_b_elements
    output_storage_elements = spec.output_storage_elements or logical_c_elements
    gm_bytes_min = (
        a_storage_elements * dtype_size(dtype)
        + b_storage_elements * dtype_size(dtype)
        + output_storage_elements * dtype_size(output_dtype)
    )
    hbm_us = gm_bytes_min / (float(config["hbm_bandwidth_tbps"]) * 1_000_000.0)
    lower = max(value for value in (compute_us, hbm_us) if value is not None)
    return compute_us, hbm_us, lower, gm_bytes_min


def dominant_bottleneck(launch_us: float, compute_us: float | None, hbm_us: float, format_us: float) -> str:
    components = {"launch": launch_us, "hbm": hbm_us, "format": format_us}
    if compute_us is not None:
        components["compute"] = compute_us
    return max(components.items(), key=lambda item: item[1])[0]


