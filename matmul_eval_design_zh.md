# MatMul 评估工具设计

## 范围

该工具用于分析 Ascend 910B4 上导出的 profiling CSV，并对 matmul 类 kernel 的实际开销进行可解释估算。当前实现范围：

- 硬件：Ascend 910B4。
- HBM 带宽：0.8 TB/s。
- HF32：不启用。
- 默认纳入：`MatMul`、`MatMulV2`、`MatMulV3`、`BatchMatMulV2`。
- 默认排除：`GroupedMatmul`、`AllGatherMatmul`。

`GroupedMatmul` 默认排除，因为专家分组权重不能直接等价成普通单 GEMM 或 batch GEMM。`AllGatherMatmul` 默认排除，因为通信开销可能主导实测时间。

## 建模原则

工具不对历史样例做插值拟合，而是使用 kernel-aware 的半解析模型：

1. 从 profiling 行解析 shape、format、dtype 和计数器。
2. 按 MatMulV3 tiling 代码中的 shape 规则解释 `FRACTAL_NZ`。
3. 根据 shape、dtype、cache 大小和 AI Core 数重构 tiling、对齐、padding 和并行效率。
4. 通过计算峰值和 HBM 带宽计算物理下界。
5. 只保留少量全局校准项，避免按 shape 拟合参数。

这样可以保证输出结果可解释，也能避免被少量 profiling 样本过拟合。

## 硬件配置

配置文件为 `configs/ascend_910b4.json`。

当前已知或用户给定的信息：

- `aic_num`：AI Core 数。当前 profiling 中 AI_CORE matmul 的 `Block Dim` 约为 20，因此默认取 20。
- `aiv_num`：Vector Core 数。默认取 40。
- `hbm_bandwidth_tbps`：0.8。

当前可调整的假设：

- `l0a_bytes`、`l0b_bytes`、`l0c_bytes`、`l1_bytes`、`l2_bytes`、`ub_bytes`：cache/buffer 大小。
- `peak_tflops`：无 HF32 时各 dtype 的理论峰值。
- `calibration`：kernel 启动开销、流水效率、可选格式转换开销。

如果后续能拿到可靠的 CANN platform 配置或官方 910B4 指标，应优先更新这些配置项，而不是修改模型逻辑。

## Profiling 提取

每条 matmul 行会提取：

- 标识信息：来源文件、CSV 行号、`Name`、`Type`。
- 时间信息：`Duration(us)`、`aicore_time(us)`、`aic_mac_time(us)`。
- 硬件计数器：`Block Dim`、`Mix Block Dim`、`cube_utilization(%)`、AIC MAC/MTE/Fixpipe 比例。
- 规格输入：`Input Shapes`、`Output Shapes`、`Input Formats`、`Output Formats`、输入/输出 dtype。
- 推断后的 GEMM：`M`、`N`、`K`、`batch`、`transA`、`transB`。
- 存储布局信息：A/B/output 的标准化 format 和物理存储元素个数。

解析器不会假设 B 一定是 `[K,N]`。当前实现会枚举 `transA/transB`，并用 output shape 给候选打分。

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

## 诊断标签

输出中的 `diagnosis` 会包含以下标签：

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

当前自动建议逻辑很简单：

- 启动开销来自低 tile 数残差的低分位数。
- 流水效率来自大 shape、高 cube 利用率样本的高分位数。

不要拟合 per-shape 系数。如果残差很大，应优先找对应的 kernel 机制，例如 tail imbalance、运行时格式转换、非连续输入、通信融合或 GMM 语义，而不是直接加经验曲线。

## 当前验证结果

当前 profiling 样例验证结果：

```text
resolved_matmul_rows = 993
unresolved_rows = 0
```

原先 3 条 unresolved 都是 `ND;FRACTAL_NZ` Weight-NZ 行，现在解析为：

```text
M=1, N=32768, K=2816, batch=1
B_storage_elements = 2048 * 176 * 16 * 16 = 92274688
estimated_us ~= 230.8
measured_us ~= 246-248
diagnosis = weight_nz|small_m_overhead|memory_bound
```

回归检查：旧版已经解析的 990 条，在加入 format-aware 解析后，`M/N/K/batch/transA/transB` 没有变化。

## CLI

默认报告：

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report.csv \
  --unresolved-output matmul_eval_unresolved.csv
```

校准建议：

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings \
  --config configs/ascend_910b4.json \
  --suggest-calibration \
  --calibration-output matmul_eval_calibration_suggested.json
```

可选参数：

- `--include-gmm`：纳入 `GroupedMatmul`。
- `--include-allgather`：纳入 `AllGatherMatmul`。

## 已知限制

- 该模型不是 CANN tiling 的完整复刻，而是用确定性约束近似 kernel 行为。
- GMM 默认排除，当前没有按 grouped expert GEMM 建模。
- AllGatherMatmul 默认排除，当前不建模通信。
- cache 大小和峰值 TFLOPS 当前是配置假设，拿到官方 910B4 数据后应更新配置。
- 运行时 ND2NZ 检测目前使用 MatMulV3 风格条件，对 BatchMatMulV3 的 multi-batch 特殊路径还可以进一步细化。
- 如果 profiling 导出的 `FRACTAL_NZ` shape 不是 storage shape，而是 origin shape，则当前 NZ 还原规则需要额外元数据辅助。
