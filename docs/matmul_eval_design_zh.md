# MatMul Kernel 评估设计文档

## 范围

本文档描述 `tools/matmul_eval/` 中普通 MatMul、BatchMatMul、TransposeBatchMatMul 和量化 MatMul 的评估方案。`GroupedMatmul` 已拆分为独立设计文档 `gmm_eval_design_zh.md`，本文件只保留与 GMM 共享的实现入口说明，不再把 GMM 精度口径混入普通 MatMul。

当前默认 `op_kind=matmul` 会评估 `Type` 中包含 matmul/bmm 相关 token 的行，但默认排除：

- `GroupedMatmul`
- `AllGatherMatmul`

可通过 `--include-gmm` 和 `--include-allgather` 显式纳入兼容报告。推荐 GMM 使用 `--op-kind grouped_matmul` 独立分析。

## 当前实现入口

- `tools/matmul_eval/common.py`：shape/format/dtype 解析、`MatmulSpec`、`TileEstimate`、`QuantSpec`。
- `tools/matmul_eval/kernel_model.py`：runtime KB、advanced tiling heuristic、analytic search、L2-aware GM 流量。
- `tools/matmul_eval/quant_model.py`：低比特 dtype、量化粒度、dequant、量化 HBM/compute 成本。
- `tools/matmul_eval/runtime_kb.py`：ops-nn runtime knowledge-base 加载和 key 匹配。
- `tools/matmul_eval/api.py`：`MatmulCostEstimate` 和 `estimate_matmul_cost(...)`。
- `tools/matmul_eval/evaluator.py`：profiling 逐行评估、report 字段、diagnosis、校准建议。

命令入口：

```bash
python3 tools/eval_ops.py --op-kind matmul \
  --profiling <profiling_dir_or_csv> \
  --config <configs/ascend_*.json> \
  --output matmul_eval_report.csv \
  --unresolved-output matmul_eval_unresolved.csv
```

## 源码依据

当前本地源码目录为 `ops-nn/matmul`。已纳入模型的主要路径：

- `ops-nn/matmul/mat_mul_v3`
- `ops-nn/matmul/batch_mat_mul_v3`
- `ops-nn/matmul/common/cmct`
- `ops-nn/matmul/quant_batch_matmul_v3`
- `ops-nn/matmul/weight_quant_batch_matmul_v2`
- `ops-nn/matmul/transpose_batch_mat_mul`

关键源码机制：

- `FRACTAL_NZ` shape 还原采用 MatMulV3 host tiling 中 `GetInputDims` 类语义。
- MatMulV3 runtime KB 提供 exact tiling 命中路径。
- 910C/A3 类平台可走 advanced tiling heuristic，覆盖 ASWT、full-load、Stream-K、fixpipe output 等源码策略。
- 910B4 当前按 checked source 走非 advanced host tiling，优先 runtime KB，未命中时使用 analytic fallback。
- 小 M/N MatMulV2 的 current-kernel overhead 来自公共 MatMul 路径中 `MatmulToMul` policy 与 checked block MMAD 路径的流水开销特征。

## profiling 解析规则

行过滤只看 `Type`：

```text
matmul
mat_mul
batchmatmul
bmm
```

解析输入：

- `Input Shapes`
- `Output Shapes`
- `Input Formats`
- `Output Formats`
- `Input Data Types`
- `Output Data Types`

输出逻辑规格 `MatmulSpec`：

- `m`
- `n`
- `k`
- `batch`
- `trans_a`
- `trans_b`
- `a_format`
- `b_format`
- `output_format`
- `a_storage_elements`
- `b_storage_elements`
- `output_storage_elements`

普通 MatMul 会枚举 `trans_a/trans_b` 候选，用 output shape 给候选打分。解析失败时写入 unresolved 报告。

## Shape 与 Format 语义

format 标准化规则：

```text
FRACTAL_NZ 或 NZ -> FRACTAL_NZ
其他 format -> ND
```

ND：

```text
matrix_dim0 = shape[-2]
matrix_dim1 = shape[-1]
batch_dims = shape[:-2]
```

FRACTAL_NZ：

```text
matrix_dim0 = storage[-3] * storage[-2]
matrix_dim1 = storage[-4] * storage[-1]
batch_dims = storage[:-4]
```

K 维匹配：

- 精确相等直接接受。
- 如果一侧是 `FRACTAL_NZ` 且较大维度等于另一侧 16 对齐后的值，则接受为 logical K。

output M/N 匹配：

- 精确相等直接接受。
- `FRACTAL_NZ` output 允许 16 对齐后的匹配。

物理存储元素直接来自 storage shape 的乘积，用于 HBM 字节计算。逻辑 FLOPs 用推断后的 M/N/K/batch。

## 物理下界与 current-kernel 字段

对一条已解析 MatMul：

```text
true_flops = 2 * M * N * K * batch

gm_bytes_min =
  A_storage_elements * dtype_size(A) +
  B_storage_elements * dtype_size(B) +
  C_storage_elements * dtype_size(C)
```

`ideal_kernel_bounds(...)` 输出理想物理下界：

```text
ideal_compute_us = true_flops / (peak_tflops * 1e6)
ideal_hbm_us = gm_bytes_min / (hbm_bandwidth_tbps * 1e6)
ideal_lower_bound_us = max(ideal_compute_us, ideal_hbm_us)
```

current-kernel 估计会使用当前 tiling 的对齐 FLOPs 和 tiled GM 流量：

```text
aligned_flops = 2 * tile_m * single_core_m * tile_n * single_core_n * tile_k * single_core_k * batch
compute_us = aligned_flops / (peak_tflops * 1e6 * core_efficiency)
hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1e6)
kernel_lower_bound_us = max(compute_us / pipeline_efficiency, hbm_us) + template_overhead_us
estimated_us = launch_overhead_us + kernel_lower_bound_us + format_overhead_us
```

报告中同时保留：

- `ideal_lower_bound_us`：物理下界。
- `current_kernel_bound_us`：当前 tiling 的理论 kernel body 下界。
- `estimated_us`：当前 kernel 估计，包含 launch/format/template 等 current-kernel 项。
- `best_kernel_us`：理想下界上叠加 launch/pipeline 后的参考值。

## Tiling 来源语义

当前区分三类 tiling 来源：

```text
runtime_kb_exact
advanced_tiling_heuristic
analytic_search
```

语义拆分：

- `runtime_kb_exact`：来自 ops-nn runtime knowledge-base 的 exact tiling，报告为 `actual_tiling`。
- `advanced_tiling_heuristic`：基于 ops-nn advanced host tiling 逻辑的源码约束近似，报告为 `actual_tiling`，但不是二进制 tiling replay。
- `analytic_search`：真实 tiling 不可得时的 fallback 解析搜索，报告为 `fallback_tiling`。
- `physical_lower_bound`：所有路径的 `optimal_tiling_source`。

## runtime KB

runtime KB 由 `configs/*.json::kernel_model.runtime_kb.files` 指定。当前 key 来自源码 `info_dict` 规格，包括：

- A/B/C dtype
- A/B/C format
- 对齐后的 M/N/K
- 对齐 flag
- 转置 flag
- bias flag

命中后读取：

- `baseM/baseN/baseK`
- `singleCoreM/singleCoreN/singleCoreK`
- `usedCoreNum`
- `depthA1/depthB1`
- `stepKa/stepKb`
- `l2MTileCnt/l2NTileCnt`
- `tilingEnable`

`tilingEnable` 被解码为：

```text
个位   = split-core 模板
十位   = AL1/BL1 full-load 模板
千位   = fixpipe/output 优化
万位   = special opti
```

## advanced tiling heuristic

advanced tiling 只在配置开启时使用。当前 910C 配置开启，910B4 关闭。

已建模的源码策略：

- `basic_aswt`
- `basic_aswt_al1_full_load`
- `basic_aswt_bl1_full_load`
- `batch_asw_basic_rebalance`
- `stream_k_sk`
- `batch_stream_k_sk`
- `stream_k_dpsk`

主要约束：

- base M/N/K 以 16 或源码常量对齐。
- L0A/L0B/L0C/L1/UB 容量限制候选。
- ASW window 由 `aic_num` 的因子估算。
- full-load 需要满足 L1 容量、single-round、MTE2 访问收益等源码条件。
- `l0c2out` 可为 `ON_THE_FLY`、`ND_FIXPIPE_1_1`、`ND_FIXPIPE_1_2`。
- Stream-K 对小 MN、大 K 形状额外计入 partial-C/reduction 流量因子。

## analytic fallback

`analytic_search` 用于 runtime KB 未命中且 advanced tiling 不适用的路径。它不是源码 exact tiling，只提供确定性估计：

- base M/N 为 16 倍数，并包含常用候选值。
- L0C 约束决定是否可双缓冲。
- base K 根据 L0A/L0B 容量估算。
- score 使用当前候选的 `max(compute_us, hbm_us)`。
- HBM 重复流量通过 L2 pressure 折减。

所有 fallback 行都会在报告中显示：

```text
current_tiling_kind = fallback_tiling
fallback_tiling_source = analytic_search
confidence <= medium
```

## L2-aware GM 流量

当前模型先计算 pessimistic tiled GM 流量：

```text
gm_bytes_tiled_raw =
  tile_N * A_storage_bytes +
  tile_M * B_storage_bytes +
  C_storage_bytes
```

再使用 `l2_aware_gm_bytes(...)` 折减重复流量：

- 若 `gm_bytes_min <= l2_bytes`，认为重复 tile 读取可由 L2 承接，`gm_bytes_tiled = gm_bytes_min`。
- 若超过 L2，只对冗余部分按 L2 pressure 折减。
- 若没有 L2 配置，使用 raw tiled bytes。

## ND2NZ 与格式开销

运行时 ND2NZ 由 `tools/matmul_eval/quant_model.py::infer_nd2nz_operands(...)` 判断：

- 只有 ND operand 可能需要运行时转换。
- 若 `inner_size * dtype_size` 属于 `{32,64,96,128,160,192,224,256,384}`，认为可走 GM-to-L0 on-the-way。
- 否则内轴非 256B 对齐或过长时可能触发普通 ND2NZ。
- 大 outer、小 inner、未对齐时可能触发 VNCHW 类路径。

若 `nd2nz_a` 或 `nd2nz_b` 为真：

```text
format_overhead_us += operand_count * calibration.format_overhead_us.ND2NZ
```

默认该项可以为 0，避免在没有证据时把格式转换硬编码为固定耗时。

## 量化 MatMul

满足以下任一条件进入量化路径：

- kernel type 包含 `Quant`
- A/B 输入 dtype 是 INT8、INT4、FP8、MXFP8、HIFLOAT8 等低比特类型

输出字段：

- `quant_mode`
- `quant_granularity`
- `quant_compute_path`
- `quant_aux_elements`
- `quant_aux_bytes`
- `quant_dequant_us`
- `quant_gm_bytes_min`
- `quant_gm_bytes_tiled`

`quant_compute_path` 包括：

- `full_quant`
- `full_quant_with_dequant`
- `weight_only_quant`
- `weight_only_quant_with_dequant`
- `fake_quant_or_mixed`

量化 HBM：

```text
quant_A_bytes = quant_storage_bytes(A_elements, A_dtype)
quant_B_bytes = quant_storage_bytes(B_elements, B_dtype)
quant_aux_bytes = sum(aux_elements * aux_dtype_size)
quant_output_bytes = output_elements * output_dtype_size
```

量化 compute：

```text
quant_compute_us =
  aligned_flops * operation_factor /
  (peak_tops * 1e6 * core_efficiency * quant_pipeline_efficiency)
```

若路径包含 dequant，则加：

```text
quant_dequant_us = output_elements * dequant_us_per_output_element
```

## small-M/N MatMulV2 current-kernel 项

`template_overhead_us` 当前只对配置显式开启的 small-M/N MatMulV2 生效。触发条件：

- `matmul_model.small_m_matmul_v2.enabled = true`
- kernel type 在 `applies_to` 中，当前用于 `MatMulV2`
- `min(M,N) <= max_m_or_n`
- A/B/output format 都是 ND
- tiling 来源为 `analytic_search`
- GM footprint 不超过 `max_gm_bytes_for_l2_resident`
- 配置提供 `effective_aligned_tflops`

该项按：

```text
template_overhead_us = aligned_flops / (effective_aligned_tflops * 1e6)
```

它表达 current-kernel 小 M/N 路径仍可能承担 L1/L0 copy、MMAD/fixpipe、同步流水等成本，不改变 `ideal_lower_bound_us`。

## QuantBatchMatmulV3 Weight-NZ epilogue 项

`QuantBatchMatmulV3` 在 arch35 Weight-NZ、per-channel scale、small-M full-quant 路径上存在 current-kernel 模板开销。源码依据：

- `ops-nn/matmul/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_basic_api_cmct.h` 使用 `BlockMmadA8W8FixpipeQuant`。
- `ops-nn/matmul/common/cmct/block/block_mmad_a8w8_fixpipe_quant.h` 中该路径设置 `disableGemv = true`。
- per-channel scale 会按 N tile 执行 `CopyX2ScaleInL1`，并在 fixpipe 输出阶段消费 scale/quant 参数。
- 该开销属于当前 kernel 的模板流水、event 同步和 epilogue 重放，不属于物理下界；因此只加到 `template_overhead_us`，不改变 `ideal_lower_bound_us`。

配置入口：

- `quant_matmul.weight_nz_epilogue.enabled`
- `applies_to`：当前用于 `QuantBatchMatmulV3`
- `granularities`：当前用于 `per_channel_n`
- `compute_paths`：当前只用于 `full_quant`，避免影响已经由 dequant/output HBM 覆盖的大 BF16 输出路径
- `max_m_or_n`、`min_n_tiles`：限制 small-M 且 N tile 足够多的 Weight-NZ 路径
- `per_n_tile_us`：按 N tile 计入 scale/fixpipe/event epilogue 成本
- `per_k_tile_us`：保留接口，当前 910C 配置为 0；K 循环主体仍由 compute/HBM 项表达
- `scale_bytes_per_n_tile`：按 N tile 补充 scale GM->L1 的显式流量

当前 910C 配置用于解释 ds3.2 `M=4,N=4096,K=7168,INT32 output` 的 full-quant Weight-NZ 低估；不会套到 `full_quant_with_dequant` BF16 输出路径。

## 报告字段

resolved MatMul 报告包含：

- 来源：`file`、`line`、`name`、`type`
- 逻辑规格：`m/n/k/batch/trans_a/trans_b`
- format/storage：`a_format`、`b_format`、`output_format`、`*_storage_elements`、`storage_padding_ratio`
- dtype/quant：`dtype`、`output_dtype`、`quant_*`
- tiling：`kernel_tiling_source`、`actual_tiling_source`、`fallback_tiling_source`、`current_tiling_kind`、`tiling_strategy`
- runtime KB：`runtime_kb_id`、`runtime_kb_file`
- tile 参数：`base_m/base_n/base_k`、`tile_m/tile_n/tile_k`、`depth_a1/depth_b1`、`step_m/step_n/step_ka/step_kb`
- source strategy：`full_load`、`l0c2out`、`asw_window_len`、`tiling_*`
- 成本分量：`compute_us`、`hbm_us`、`flops_cost_us`、`memory_access_us`、`launch_overhead_us`、`format_overhead_us`、`template_overhead_us`
- 下界与对比：`current_kernel_bound_us`、`ideal_lower_bound_us`、`best_kernel_us`
- 结果：`estimated_us`、`residual_us`、`duration_over_estimate`、`diagnosis`、`confidence`

## diagnosis 与 confidence

主要 diagnosis 标签：

- `runtime_kb_exact`
- `advanced_tiling_heuristic`
- `fallback_tiling`
- `stream_k`
- `al1_full_load` / `bl1_full_load`
- `fixpipe_output`
- `quant_matmul`
- `full_quant_dequant`
- `weight_only_quant` / `weight_only_quant_dequant`
- `fake_or_mixed_quant`
- `weight_nz` / `fractal_nz`
- `runtime_nd2nz`
- `layout_padding`
- `small_m_overhead`
- `small_m_matmul_v2_serial_pipeline`
- `low_tile_count`
- `low_cube_utilization`
- `compute_bound` / `memory_bound` / `balanced_bound`
- `launch_bound`
- `large_residual`

confidence 规则：

- 默认 high。
- fallback tiling 降为 medium。
- `M <= 4` 或 tile 数少于 `aic_num` 降为 low。
- unknown compute peak 降为 medium。
- GMM 兼容行固定 low，推荐改用独立 GMM 文档和报告口径。

## 校准项

只允许全局项，不允许 per-shape 拟合：

- `calibration.launch_overhead_us_by_type`
- `calibration.pipeline_efficiency_by_dtype`
- `calibration.format_overhead_us.ND2NZ`
- `quant_matmul.peak_tops`
- `quant_matmul.pipeline_efficiency`
- `quant_matmul.operation_factor`
- `quant_matmul.dequant_us_per_output_element`
- `matmul_model.small_m_matmul_v2` 中的平台级 current-kernel 小 M/N 参数

`--suggest-calibration` 目前只支持 matmul：

- launch 建议来自低 tile 数残差的低分位数。
- pipeline 建议来自大 shape、高 cube 利用率样本的高分位数。

## 已知限制

- `advanced_tiling_heuristic` 不是 exact host tiling replay。
- `analytic_search` 是 fallback，不应当被解释为实际 kernel tiling。
- BatchMatMulV3 的 iter-batch、merge-batch 等特殊路径仍是近似。
- AllGatherMatmul 默认排除，当前不建模通信。
- 量化路径仍依赖配置中的有效 TOPS 和 pipeline 参数。
- 若 profiling 导出的 `FRACTAL_NZ` shape 不是 storage shape，当前 NZ 还原需要额外元数据辅助。

## 后续任务

- 增加 MatMulV2/BatchMatMulV2 host tiling replay。
- 补齐 BatchMatMulV3 iter-batch/merge-batch 的源码策略拆分。
- 扩大量化 MatMul 的 FP8/MXFP8/per-group 样本验证。
- 对 fallback tail 输出更细粒度的源码缺口分类。
