# MatMul 评估工具设计

## 范围

该工具用于分析 Ascend 上导出的 profiling CSV，并对 matmul 类 kernel 的实际开销进行可解释估算。当前实现范围：

- 硬件配置：Ascend 910B4 和 Ascend 910C。
- 910B4 HBM 带宽：0.8 TB/s。
- 910C HBM 带宽：1.6 TB/s。
- 910C BF16/FP16 峰值：按可见设备 400 TFLOPS。
- HF32：不启用。
- 默认纳入：`MatMul`、`MatMulV2`、`MatMulV3`、`BatchMatMulV2`。
- 默认排除：`GroupedMatmul`、`AllGatherMatmul`。

`GroupedMatmul` 默认排除，因为专家分组权重不能直接等价成普通单 GEMM 或 batch GEMM。`AllGatherMatmul` 默认排除，因为通信开销可能主导实测时间。

当显式使用 `--include-gmm` 纳入 `GroupedMatmul` 时，解析器按专家分组语义处理：

- 第一输入 `x` 的 token 数作为总 `M`。
- 第二输入 `weight` 的 `FRACTAL_NZ` 首维视为专家数，不作为普通 batch 乘入 FLOPs。
- 逻辑 FLOPs 按实际 token 总量估算，而不是按“专家数 * token 数”估算。
- 权重物理存储仍保留完整专家权重大小，但 profiling CSV 通常只有 `group_list` 的形状，没有每个专家实际 token 分布，因此无法精确判断本次 kernel 访问了多少专家权重切片。

因此 `GroupedMatmul` 当前仍是低置信度路径：如果报告出现 `ideal_lower_bound_us > duration_us`，优先检查是否把全部专家权重流量计入了单次执行，而真实 kernel 只访问了非空专家或命中了 L2/权重缓存。

`GroupedMatmul` 也可以通过独立算子族入口评估：

```bash
python3 tools/eval_ops.py --op-kind grouped_matmul \
  --profiling <profiling> \
  --config <config.json> \
  --output grouped_matmul_eval_report.csv \
  --unresolved-output grouped_matmul_eval_unresolved.csv
```

独立报告保留普通 matmul 字段用于兼容，同时额外输出 routing 场景边界：

- `gmm_balanced_*`：理想均衡场景，tokens 尽量均匀分布到可用专家，权重流量按活跃专家数计入。
- `gmm_extreme_*`：极端负载不均衡场景，所有 tokens 落到单个专家，权重流量最小但并行度最低。
- `gmm_duration_position`：实测时间位于两个边界内、低于边界或高于边界。

这两个场景不是拟合值，而是缺少 `group_list` 实际值时的可解释上下界。若实测高于均衡边界，说明还存在模板调度、同步、atomic/merge、额外搬运或源码路径未建模；若实测低于极端边界，则优先检查 shape/流量解析。

## 建模原则

工具不对历史样例做插值拟合，而是使用 kernel-aware 的半解析模型：

1. 从 profiling 行解析 shape、format、dtype 和计数器。
2. 按 MatMulV3 tiling 代码中的 shape 规则解释 `FRACTAL_NZ`。
3. 根据 shape、dtype、cache 大小和 AI Core 数重构 tiling、对齐、padding 和并行效率。
4. 通过计算峰值和 HBM 带宽计算物理下界。
5. 只保留少量全局校准项，避免按 shape 拟合参数。

这样可以保证输出结果可解释，也能避免被少量 profiling 样本过拟合。

## 硬件配置

配置文件按硬件目标拆分：

- `configs/ascend_910b4.json`：20 个 AI Core，0.8 TB/s HBM，无 HF32 BF16/FP16 峰值 240 TFLOPS。
- `configs/ascend_910c.json`：从 910C profiling 的 Block Dim 推断 24 个 AI Core，1.6 TB/s HBM，无 HF32 BF16/FP16 峰值 400 TFLOPS。

对新增昇腾 profiling，如果 README 或元数据没有明确平台，可用 `Block Dim`/`Block Num` 做初步交叉判断：

- Cube 类算子看 `aic_num`，例如 `MatMul`、`BatchMatMul`、`GroupedMatmul`、`QuantBatchMatmul` 和主要走 Cube 的 attention/FA 子路径。
- Vector 类算子看 `aiv_num`，例如 `Cast`、`Transpose`、`RotaryMul`、`Gather/Scatter`、activation、normalization、routing 和部分 fusion。
- 910B4 的典型证据是 Cube 最大 block dim 约 `20`，Vector 最大 block dim 约 `40`。
- 910C/A3 的典型证据是 Cube 最大 block dim 约 `24`，Vector 最大 block dim 约 `48`。
- 不要用全文件最大 block dim 直接判断 matmul/FA 的 Cube 核数；全局最大值常由 Vector 算子贡献。

当前已知或用户给定的信息：

- `aic_num`：用于 tile/core-efficiency 建模的 AI Core 数。
- `aiv_num`：Vector Core 数。
- `hbm_bandwidth_tbps`：HBM 带宽，单位 TB/s。

当前可调整的假设：

- `l0a_bytes`、`l0b_bytes`、`l0c_bytes`、`l1_bytes`、`l2_bytes`、`ub_bytes`：cache/buffer 大小。
- `peak_tflops`：无 HF32 时各 dtype 的理论峰值。
- `calibration`：kernel 启动开销、流水效率、可选格式转换开销。

如果后续能拿到可靠的 CANN platform 配置或官方目标硬件指标，应优先更新这些配置项，而不是修改模型逻辑。

## Profiling 提取

每条 matmul 行会提取：

- 标识信息：来源文件、CSV 行号、`Name`、`Type`。
- 时间信息：`Duration(us)`、`aicore_time(us)`、`aic_mac_time(us)`。
- 硬件计数器：`Block Dim`、`Mix Block Dim`、`cube_utilization(%)`、AIC MAC/MTE/Fixpipe 比例。
- 规格输入：`Input Shapes`、`Output Shapes`、`Input Formats`、`Output Formats`、输入/输出 dtype。
- 推断后的 GEMM：`M`、`N`、`K`、`batch`、`transA`、`transB`。
- 存储布局信息：A/B/output 的标准化 format 和物理存储元素个数。

解析器不会假设 B 一定是 `[K,N]`。当前实现会枚举 `transA/transB`，并用 output shape 给候选打分。

行过滤只根据算子 `Type` 判断是否属于 matmul，不使用 `Name`。这样会排除 `Type=Mul` 但名称中带有 `Matmul` 的 profiling 行，避免把 elementwise mul 误当作 matmul 评估。

## Shape 和 Format 推断

format 先标准化：

```text
FRACTAL_NZ 或 NZ -> FRACTAL_NZ
其他 format -> ND
```

这与 MatMulV3/BatchMatMulV3 kernel 的行为一致：编译期 format 宏会把 `FORMAT_FRACTAL_NZ` 映射到 `CubeFormat::NZ`，其他 format 统一按 `CubeFormat::ND` 处理。

ND 存储的矩阵维度：

```text
matrix_dim0 = shape[-2]
matrix_dim1 = shape[-1]
batch_dims = shape[:-2]
```

`FRACTAL_NZ` 存储的矩阵维度按 host tiling 中的 `GetInputDims` 规则还原：

```text
matrix_dim0 = storage[-3] * storage[-2]
matrix_dim1 = storage[-4] * storage[-1]
batch_dims = storage[:-4]
```

候选推断规则：

- 对每组 `transA/transB` 枚举一个候选。
- K 维需要精确匹配，或者在 `FRACTAL_NZ` 侧允许 16 元素对齐后的匹配。
- output 的 M/N 需要精确匹配，或者对 `FRACTAL_NZ` output 允许 16 元素对齐后的匹配。
- 选择得分最高的候选，并对非转置布局给一个很小的偏好。

## FRACTAL_NZ / Weight-NZ 处理

`FRACTAL_NZ` 被当作存储布局处理，不被当作一个拟合出来的时间类别。

当前 profiling 中的一个例子：

```text
x1 shape    = [1, 2816], format ND
x2 shape    = [2048, 176, 16, 16], format FRACTAL_NZ
out shape   = [1, 32768], format ND
x2 logical  = [176 * 16, 2048 * 16] = [2816, 32768]
GEMM spec   = M=1, N=32768, K=2816, batch=1
```

物理存储元素数直接来自 storage shape：

```text
A_storage_elements = product(A_storage_shape)
B_storage_elements = product(B_storage_shape)
C_storage_elements = product(C_storage_shape)
```

这意味着预排布的 Weight-NZ 会按实际物理字节计入 HBM 流量。已经是 `FRACTAL_NZ` 的权重不会额外加 `format_overhead_us`，格式转换开销只留给运行时 ND2NZ 路径。

之前 unresolved 的三条 `ND;FRACTAL_NZ` 行现在都会解析为 `M=1, N=32768, K=2816`。这类小 M、大 N、Weight-NZ 的场景主要受大权重矩阵 HBM 读取限制，而不是 compute 限制。

## 运行时 ND2NZ 检测

运行时 ND2NZ 转换按确定性规则检测。当前实现采用 MatMulV3 风格的近似规则：

- 只有 ND operand 可能需要运行时 ND2NZ。
- 如果 `inner_size * dtype_size` 属于 `{32, 64, 96, 128, 160, 192, 224, 256, 384}`，认为可走 GM-to-L0 on-the-way 转换。
- 否则，如果内轴不是 256B 对齐，或者 `inner_size > 65535`，可能触发普通 ND2NZ；FP32 无 HF32 的部分场景例外。
- 当外轴较大且内轴较小、未对齐时，可能触发 VNCHW 风格 ND2NZ。

输出字段为 `nd2nz_a` 和 `nd2nz_b`。如果任一为真：

```text
format_overhead_us = ND2NZ operand 数量 * calibration.format_overhead_us.ND2NZ
```

默认 `ND2NZ` 开销为 0，除非用户在配置中提供全局值。`calibration.format_overhead_us.FRACTAL_NZ` 目前保留在配置里做兼容，但当前代码不会对已经预排布的 NZ 输入收取该项。

## 解析开销模型

对于一条已解析 GEMM：

```text
true_flops = 2 * M * N * K * batch
aligned_flops = 2 * aligned_M * aligned_N * aligned_K * batch
compute_us = aligned_flops / (peak_tflops * 1e6 * core_eff)

gm_bytes_min =
  A_storage_elements * input_dtype_size +
  B_storage_elements * input_dtype_size +
  C_storage_elements * output_dtype_size

gm_bytes_tiled_raw =
  tile_N * A_storage_bytes +
  tile_M * B_storage_bytes +
  C_storage_bytes

hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1e6)
lower_bound_us = max(compute_us, hbm_us)
estimated_us = launch_overhead_us + max(compute_us / pipeline_efficiency, hbm_us) + format_overhead_us
```

`launch_overhead_us` 来自 `configs/ascend_910b4.json::calibration.launch_overhead_us_by_type`。当前默认值为 0.0，表示不把平台/运行时相关的固定开销硬编码进模型；对小规格或低 tile 数算子，应使用 `--suggest-calibration` 从低 tile 残差中得到初始估计，再写回配置。

`storage_padding_ratio` 定义为：

```text
physical_storage_elements / logical_storage_elements
```

该字段用于标识 layout 或 padding 带来的物理存储膨胀。

## Tiling 近似

tiling 搜索是确定性的：

- `baseM` 和 `baseN` 候选为 16 的倍数，并额外加入 64、80、96、128、192、256、320、336 等偏好值。
- 单个候选的最大 extent 当前限制为 512。
- L0C 单缓冲约束：`baseM * baseN * 4 <= l0c_bytes`。
- 如果双缓冲 L0C 也放得下，则 `db_l0c=2`，否则为 1。
- `baseK` 由 L0A/L0B 容量推导，并按 16 对齐。
- 候选评分使用物理下界 `max(compute_us, hbm_us)`。

AI Core 并行效率按 M/N/batch tile 数估计：

```text
mn_tile_count = ceil(M/baseM) * ceil(N/baseN) * batch
rounds = ceil(mn_tile_count / aic_num)
core_eff = mn_tile_count / (rounds * aic_num)
```

tail 效率：

```text
tail_eff = true_flops / aligned_flops
```

HBM 估算考虑 L2：

- 如果 `gm_bytes_min <= l2_bytes`，重复 tile 读取视为 L2 命中，HBM 使用 `gm_bytes_min`。
- 如果超过 L2，则对 `gm_bytes_tiled_raw - gm_bytes_min` 的冗余流量施加一个确定性的 L2 pressure 项。

## Kernel 实现集成

工具现在区分三类 tiling 来源：

- `runtime_kb_exact`：只用于 ops-nn `MatMulV3` 的单 batch 精确命中。
- `advanced_tiling_heuristic`：只用于开启 advanced tiling 的 ops-nn `MatMulV3` / `BatchMatMulV3`。
- `analytic_search`：用于非 V3 kernel、910B4 非 advanced 路径、量化 kernel 的基础 tile 估算。

`runtime_kb` 文件按 JSON Lines 解析，key 来自源码中的 `info_dict` 规格：A/B/C dtype、format、对齐后的 `M/N/K`、对齐 flag、转置 flag 和 bias flag。BF16 按源码行为归一到 FLOAT16 code；FP32 的 K 对齐按 8，FP16/BF16 按 16。精确命中后直接读取 `knowledge` 中的 `baseM/baseN/baseK/singleCoreM/singleCoreN/singleCoreK/usedCoreNum/depthA1/depthB1/stepKa/stepKb/l2MTileCnt/l2NTileCnt`，并解码 `tilingEnable`：

```text
个位   = split-core 模板
十位   = AL1/BL1 full-load 模板
千位   = fixpipe/output 优化
万位   = special opti
```

910C 配置按 `ascend910_93` / `DAV_3510` 处理 advanced tiling；910B4 配置关闭 advanced tiling，因为当前 ops-nn 源码中 910B4 走非 advanced host tiling 路径。advanced 路径实现了源码中的主要决策：

- `MatMulV3`：先检查 Stream-K 条件，再走 Basic ASWT；默认 `baseM=256/baseN=256/baseK=128B/dtype`，按 L0/L1 约束计算 `baseK/depth/step`，并按源码条件判断 AL1/BL1 full-load 和 `L0C2Out`。
- `BatchMatMulV3`：先检查 batch Stream-K，再用 `GetRebalanceBlock` 的核心思想做 ASW rebalance：使用 HBM/L2/core-freq 估算 cube-bound edge，结合尾块负载均衡选择 `baseM/baseN/baseK`。
- `Stream-K`：对小 MN、大 K 场景按源码阈值建模，并额外计入可配置的 partial-C/reduction 流量因子。

新增输出字段用于解释当前 kernel：

- `kernel_tiling_source`，以及显式语义拆分字段 `actual_tiling_source`、`fallback_tiling_source`、`optimal_tiling_source`、`current_tiling_kind`。
- `tiling_strategy`、`full_load`、`l0c2out`。
- `base_m/base_n/base_k`、`depth_a1/depth_b1`、`step_m/step_n/step_ka/step_kb`。
- `runtime_kb_id/runtime_kb_file` 和解码后的 `tiling_split_core/tiling_full_load/tiling_fix_opti/tiling_special_opti`。
- `current_kernel_bound_us/current_theoretical_tflops`：按当前 kernel tiling 的理论下界。
- `ideal_lower_bound_us/ideal_tflops`：不考虑 tiling padding、重复搬运和启动开销的物理下界。
- `best_kernel_us/best_kernel_tflops`：在理想 kernel 下界上加全局启动/流水项后的最优可达估计。
- `kernel_gap_to_best/current_gap_to_ideal/bottleneck`：当前 kernel 与最优/理想下界的差距和主导瓶颈。

## Quant Matmul 处理

量化 matmul 使用独立路径，不再沿用普通浮点 matmul 模型。满足以下任一条件就会进入量化路径：kernel type 中包含 `Quant`，或者 A/B 输入 dtype 是 `INT8`、`INT4`、`MXFP8` 等低 bit 数据类型。

评估器会推断：

- `quant_mode`：根据 A/B dtype 判断 `int8`、`int4`、`mxfp8` 等模式。
- `quant_compute_path`：`full_quant`、`full_quant_with_dequant` 或 `fake_quant_or_mixed`。
- `quant_granularity`：根据 scale 等辅助输入 shape 推断量化粒度。
- `quant_aux_bytes`：scale/offset 等辅助输入流量。

全量化指 A 和 B 都是低 bit tensor，并且主 matmul 路径按低 bit cube 吞吐建模。如果 A/B 是低 bit，额外带 FLOAT scale，输出是 FP16/BF16/FP32，则标记为 `full_quant_with_dequant`：主路径按 integer accumulate 估算，同时认为存在 scale/dequant/output conversion。若只有部分输入是低 bit 或规格不够明确，则标记为 `fake_quant_or_mixed`。

量化粒度根据 scale shape 推断：

- 标量或 `[1]`：`per_tensor`。
- 一维 scale 等于 `N`：`per_channel_n`。
- 一维 scale 等于 `M`：`per_token_m`。
- 如果 `M == N` 且 scale 为 `[M] == [N]`，默认只能标记为 `per_channel_n_or_per_token_m`，因为仅凭 shape 无法判断轴；但 `QuantBatchMatmulV3` 会按源码 API 输入顺序 `scale, offset, bias, pertokenScale` 解析，四输入场景中的 `[N]` scale/offset 会标记为 `per_channel_n`。
- 如果 scale shape 能整除 `M` 或 `N`，标记为 `per_group_or_block`。
- FP8/MXFP8 会体现在 `quant_mode` 中，但具体 block size 需要 profiling CSV 之外的元数据。

量化 HBM 字节按低 bit 存储重新计算：

```text
quant_A_bytes = A_elements * bitwidth(A) / 8
quant_B_bytes = B_elements * bitwidth(B) / 8
quant_aux_bytes = sum(product(aux_shape) * aux_dtype_size)
quant_output_bytes = C_elements * output_dtype_size
```

量化 compute 时间使用 `configs/ascend_910b4.json::quant_matmul` 中的显式参数：

```text
quant_compute_us =
  aligned_flops * operation_factor /
  (peak_tops * 1e6 * core_eff * quant_pipeline_efficiency)
```

当前 `QuantBatchMatmulV3` 样例的输入是 `INT8;INT8;FLOAT;FLOAT`，输出是 `FLOAT16`，两个辅助输入 shape 都是 `[4096]`。按源码 API 顺序这两个辅助输入是 `scale` 和 `offset`，工具会推断为 `int8`、`full_quant_with_dequant`、`per_channel_n`。

## 诊断标签

输出中的 `diagnosis` 会包含以下标签：

- `quant_matmul`：使用了量化 matmul 路径。
- `runtime_kb_exact`：使用 runtime_kb 精确 tiling。
- `advanced_tiling_heuristic`：使用源码约束的 advanced tiling 估算。
- `stream_k`：当前 V3 规格满足 Stream-K 模板。
- `al1_full_load`、`bl1_full_load`：当前 V3 规格进入 L1 full-load 模板。
- `fixpipe_output`：advanced tiling 判断输出走 fixpipe 优化。
- `full_quant_dequant`：低 bit matmul 后接浮点输出转换。
- `weight_only_quant`、`weight_only_quant_dequant`：weight-only quant matmul 路径。
- `fake_or_mixed_quant`：伪量化或混合量化路径。
- `weight_nz`：B 为 `FRACTAL_NZ`。
- `fractal_nz`：A 或 output 为 `FRACTAL_NZ`。
- `runtime_nd2nz`：检测到运行时 ND2NZ。
- `layout_padding`：物理存储元素数比逻辑元素数高 5% 以上。
- `small_m_overhead`：`M <= 4`。
- `low_tile_count`：M/N/batch tile 数小于 AI Core 数。
- `low_cube_utilization`：profiling 中 cube 利用率低于 80%。
- `compute_bound`、`memory_bound`、`balanced_bound`：解析模型判断的主导瓶颈。
- `large_residual`：实测时间超过估计时间 5 倍。

当 M 很小或 tile 数很少时，confidence 会降为 low，因为启动、调度、vector/fixpipe 和内存延迟可能主导实测时间。

## 校准项

只建议校准全局项：

- `launch_overhead_us_by_type`：按 kernel type 的全局启动/调度开销。
- `pipeline_efficiency_by_dtype`：大 shape、compute-heavy 场景下的持续峰值比例。
- `format_overhead_us.ND2NZ`：检测到运行时 ND2NZ 时的可选全局转换开销。
- `quant_matmul.peak_tops`：低 bit 有效计算吞吐。
- `quant_matmul.pipeline_efficiency`：低 bit 全局流水效率。
- `quant_matmul.operation_factor`：full/fake quant 路径的成本倍率。
- `quant_matmul.dequant_us_per_output_element`：可选 dequant/output conversion 项。

当前自动建议逻辑很简单：

- 启动开销来自低 tile 数残差的低分位数。
- 流水效率来自大 shape、高 cube 利用率样本的高分位数。

启动开销拟合是按 kernel type 的全局项，不是 per-shape 曲线。当前逻辑在低 tile 数样本中计算 `duration_us - lower_bound_us` 的低分位数，得到绝对启动/调度开销；低分位数用于避免把 cache miss、tail 或非典型访存误吸收到启动开销里。流水效率则使用大 shape、高 cube 利用率样本的 `achieved_tflops / peak_tflops` 高分位数。

当前已写入配置的校准值：

- 910B4 启动开销：`BatchMatMul=2.35824us`、`MatMul=3.6392us`、`MatMulV2=3.1942us`。
- 910B4 持续效率：`DT_BF16=0.922814`、`FLOAT16=0.976532`、`FLOAT=0.99724`、量化 `INT8=0.522674`。
- 910C 启动开销：`MatMulV2=7.61184us`；当前 910C 样例没有低 tile 数 `MatMulV3` 校准行，因此 `MatMulV3` 仍为 `0.0`。
- 910C 持续效率：`DT_BF16=0.772807`，基于用户指定的 400 TFLOPS 峰值。

不要拟合 per-shape 系数。如果残差很大，应优先找对应的 kernel 机制，例如 tail imbalance、运行时格式转换、非连续输入、通信融合或 GMM 语义，而不是直接加经验曲线。

## 当前验证结果

当前 910B4 profiling 样例验证结果：

```text
resolved_matmul_rows = 1255
unresolved_rows = 0
```

当前 910C profiling 样例使用 `configs/ascend_910c.json` 的验证结果：

```text
resolved_matmul_rows = 672
unresolved_rows = 0
MatMulV2 rows = 416
MatMulV3 rows = 256
MatMulV2 median actual / estimate ~= 1.05
MatMulV3 median actual / estimate ~= 0.98
MatMulV3 tiling source = advanced_tiling_heuristic
MatMulV2 tiling source = analytic_search
MatMulV3 median current_gap_to_ideal ~= 1.04
```

原先 3 条 unresolved 都是 `ND;FRACTAL_NZ` Weight-NZ 行，现在解析为：

```text
M=1, N=32768, K=2816, batch=1
B_storage_elements = 2048 * 176 * 16 * 16 = 92274688
estimated_us ~= 234.4
measured_us ~= 246-248
diagnosis = weight_nz|small_m_overhead|memory_bound
```

回归检查：旧版已经解析的 990 条，在加入 format-aware 解析后，`M/N/K/batch/transA/transB` 没有变化。

当前低 bit profiling 文件包含 11 条 `QuantBatchMatmulV3`：

```text
Input dtypes = INT8;INT8;FLOAT;FLOAT
Output dtype = FLOAT16
M=4096, N=4096, K=12800
quant_mode = int8
quant_compute_path = full_quant_with_dequant
quant_granularity = per_channel_n
median actual / estimate ~= 1.00
median absolute percentage error ~= 1.0%
```

## CLI

每次运行应使用一个 SoC 目录和匹配配置。目录输入会递归扫描，因此直接传顶层 `example_profilings` 会把 910B4/910C 行混到同一套硬件配置下。

910B4 报告：

```bash
python3 tools/eval_ops.py \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report_910b4.csv \
  --unresolved-output matmul_eval_unresolved_910b4.csv
```

910C 报告：

```bash
python3 tools/eval_ops.py \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output matmul_eval_report_910c.csv \
  --unresolved-output matmul_eval_unresolved_910c.csv
```

校准建议需要使用匹配的 SoC 目录和配置：

```bash
python3 tools/eval_ops.py \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --suggest-calibration \
  --calibration-output matmul_eval_calibration_suggested_910b4.json
```

可选参数：

- `--include-gmm`：在普通 `matmul` 报告中纳入 `GroupedMatmul`，并附带 GMM routing 边界字段；独立分析推荐使用 `--op-kind grouped_matmul`。
- `--include-allgather`：纳入 `AllGatherMatmul`。

## 已知限制

- 该模型不是 CANN tiling 的完整复刻，而是用确定性约束近似 kernel 行为；`runtime_kb_exact` 命中时才等价于读取源码生成的精确 tiling 知识。
- 当前 profiling 中的 910C MatMulV3 shape 没有 runtime_kb 精确命中，因此使用 `advanced_tiling_heuristic`。
- `BatchMatMulV3` advanced full-load/iter-batch/merge-batch 特殊路径仍是近似，后续如果有对应 profiling 样例应继续细化。
- GMM 默认不混入普通 MatMul 口径；当前提供独立 `grouped_matmul` 入口和低置信度 routing 边界模型，但没有真实 `group_list` 时不能声称精确复刻单次专家分布。
- AllGatherMatmul 默认排除，当前不建模通信。
- cache 大小和峰值 TFLOPS 当前是配置假设，拿到官方目标硬件数据后应更新配置。
- 运行时 ND2NZ 检测目前使用 MatMulV3 风格条件，对 BatchMatMulV3 的 multi-batch 特殊路径还可以进一步细化。
- Quant matmul 支持会区分模式和粒度，但仍依赖有效 `peak_tops` 和 pipeline 参数。MXFP8、per-group 等更多量化模式需要更多 profiling 样例验证。
- 如果 profiling 导出的 `FRACTAL_NZ` shape 不是 storage shape，而是 origin shape，则当前 NZ 还原规则需要额外元数据辅助。
