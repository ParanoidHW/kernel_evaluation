# 当前 Profiling 评估差距

本文记录当前已有 profiling 报告在“大 shape / 计算核较充分使用”条件下的评估差距。该视角用于观察 tiling、kernel 实现和硬件建模问题，避免被小 shape 的启动开销主导。

## 过滤口径

- 排除 `duration_us < 10us` 的样本。这类样本按启动开销主导处理，不作为 tiling 策略准确率的主要判断依据。
- 大 shape / 核打满样本定义为：
  - `block_dim` 或 `mix_block_dim >= 0.8 * aic_num`，或
  - `cube_utilization_pct >= 70`。
- 误差指标：
  - `relative_error = abs(estimated_us - duration_us) / duration_us`。
  - `duration_over_estimate > 1` 表示低估耗时，`< 1` 表示高估耗时。
- 如果 `ideal_lower_bound_us > duration_us`，优先判为解析、物理流量或硬件配置假设问题，不应通过经验系数拟合。

## 当前汇总

以下表格中的 `matmul` 仍使用兼容字段 `estimated_us`，因此 `GroupedMatmul` 的旧误差会显得很大；GMM 需要看后面的 routing-bound 独立口径。

| report | rows | large | max | p95 | median | DOE median | LB violations | top type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base_910b4_matmul | 1463 | 1161 | 19.088 | 6.215 | 0.100 | 1.054 | 209 | GroupedMatmul |
| base_910b4_attention | 246 | 139 | 0.259 | 0.208 | 0.089 | 0.919 | 0 | FusedInferAttentionScore |
| base_910c_matmul | 672 | 588 | 0.542 | 0.240 | 0.055 | 1.041 | 0 | MatMulV2 |
| base_910c_attention | 256 | 160 | 0.116 | 0.099 | 0.025 | 1.026 | 0 | FlashAttentionScore |
| ds32_matmul | 1060 | 420 | 0.689 | 0.651 | 0.165 | 1.198 | 0 | QuantBatchMatmulV3 |
| ds32_attention | 90 | 90 | 1.840 | 1.821 | 1.719 | 0.368 | 80 | KvQuantSparseFlashAttention |
| gemma_matmul | 888 | 778 | 8.856 | 6.781 | 0.099 | 1.054 | 180 | GroupedMatmul |
| gemma_attention | 90 | 75 | 0.416 | 0.236 | 0.152 | 0.868 | 0 | FusedInferAttentionScore |
| qwen7b_matmul | 483 | 483 | 0.811 | 0.584 | 0.390 | 0.719 | 483 | MatMulV2 |
| qwen7b_attention | 96 | 96 | 0.468 | 0.459 | 0.416 | 1.714 | 0 | FusedInferAttentionScore |
| longcat_matmul | 279 | 87 | 19.008 | 15.571 | 0.130 | 0.976 | 28 | GroupedMatmul |
| longcat_attention | 28 | 0 | - | - | - | - | 0 | - |

## GroupedMatmul 独立口径

`GroupedMatmul` 缺少真实 `groupList` 和 `tuningConfigOptional`，因此独立报告不把普通 MatMul 的 `estimated_us` 当作最终判断，而是输出两种场景边界。源码依据来自 `ops-transformer/gmm/grouped_matmul`：host tiling 设置 `groupListType/singleN/usedCoreNum`，kernel 内逐 group 读取 `groupList` 并用 `count % coreNum` 轮转调度，`Block Dim` 不能反推活跃专家数。

- `gmm_balanced_*`：理想均衡，tokens 尽量分布到更多专家。
- `gmm_extreme_*`：极端不均衡，所有 tokens 落到单个专家。
- `gmm_routing_bound_error`：实测落在两边界内记为 `0`，否则计算到最近边界的相对距离。

| report | rows | large | bound max | bound p95 | bound median | positions |
|---|---:|---:|---:|---:|---:|---|
| grouped_matmul_910b4 | 208 | 187 | 0.180 | 0.074 | 0.000 | 171 within, 16 above |
| longcat_grouped_matmul_910b4 | 28 | 28 | 0.179 | 0.157 | 0.040 | 11 within, 17 above |
| gemma_grouped_matmul_910b4 | 180 | 160 | 0.000 | 0.000 | 0.000 | 160 within |
| ds32_grouped_matmul_910c | 120 | 120 | 0.176 | 0.165 | 0.090 | 120 above |

## 结论

- 910C attention 大 shape 结果最好：`FlashAttentionScore/FusedInferAttentionScore` 的大 shape max 约 `11.6%`，p95 约 `9.9%`，当前 source strategy replay 对长 prefill 已可作为相对可信基线。
- 910C matmul 大 shape 总体可用，但 `MatMulV2` 的 `M=1` decode-like 路径仍有约 `54%` 最大误差；这不是 `<10us` 启动主导，而是小 M 但长 K/N 的 kernel 路径问题。
- 910B4 普通 attention 大 shape/较充分使用样本 max 约 `25.9%`，主要是 `FusedInferAttentionScore` decode-like 路径。
- `GroupedMatmul` 的普通 MatMul 兼容估计仍会产生巨大误差和下界违反，但独立 routing-bound 口径已把问题收敛到可解释范围：Gemma 大样本全部位于边界内，Longcat 在加入源码可见的 groupList 调度项后 max 约 `17.9%`，DS3.2 高于边界约 `17.6%`。剩余差距需要真实 `groupList/tuningConfigOptional`、专家调度/同步、merge/atomic 或更具体 GMM kernel 分支解释，不能用经验系数拟合。
- `ds3.2` 的 `KvQuantSparseFlashAttention` 是当前 attention 最大差距，max 约 `184%`，且有大量物理下界违反。该路径需要专门结合 `kv_quant_sparse_flash_attention` 源码和 tiling 处理，不能继续套普通 FA 模型。
- `qwen7b MatMulV2` 在当前 910B4 配置下所有大 shape 样本都出现物理下界违反，说明硬件带宽/平台配置、shape 解析或物理存储假设至少有一项不匹配。该问题优先级高于普通拟合。
- `longcat_attention` 没有进入大 shape/核打满集合，当前不作为 tiling 准确率核心样本。

## 下一步优先级

1. 检查 `qwen7b MatMulV2` 的平台和物理流量假设：它按 Cube/Vector block dim 匹配 910B4，但报告中 `ideal_lower_bound_us > duration_us` 全量出现，需要核对 HBM 带宽、profiling 单位、shape 存储解释和 MatMulV2 kernel 路径。
2. 为 `KvQuantSparseFlashAttention` 建立专用模型：读取 `ops-transformer` 中 `kv_quant_sparse_flash_attention` 的 host tiling 和 kernel 分支，拆分 kv quant、sparse、MLA absorb、mask/aux、workspace 和实际访问流量。
3. 对 910C/910B4 `MatMulV2 M=1` 长 K/N 路径做源码/tiling 检查，避免继续用普通 fallback tiling 解释 decode-like GEMV/GEMM 边界路径。
4. 继续收敛 `GroupedMatmul` above-bound 样本：优先寻找真实 `groupList`、`groupListType` 和 `tuningConfigOptional`；没有这些运行时输入前，区间模型比单点估计更符合源码语义。
