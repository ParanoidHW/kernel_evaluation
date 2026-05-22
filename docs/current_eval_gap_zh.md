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

## 结论

- 910C attention 大 shape 结果最好：`FlashAttentionScore/FusedInferAttentionScore` 的大 shape max 约 `11.6%`，p95 约 `9.9%`，当前 source strategy replay 对长 prefill 已可作为相对可信基线。
- 910C matmul 大 shape 总体可用，但 `MatMulV2` 的 `M=1` decode-like 路径仍有约 `54%` 最大误差；这不是 `<10us` 启动主导，而是小 M 但长 K/N 的 kernel 路径问题。
- 910B4 普通 attention 大 shape/较充分使用样本 max 约 `25.9%`，主要是 `FusedInferAttentionScore` decode-like 路径。
- `GroupedMatmul` 是 910B4/gemma/longcat matmul 的最大误差来源，且存在大量 `ideal_lower_bound_us > duration_us`。当前 `kernel_details.csv` 缺少实际 `group_list` 值，无法知道活跃专家数量和每专家 token 分布；不能用全专家权重流量直接当作单次执行 GM 流量。
- `ds3.2` 的 `KvQuantSparseFlashAttention` 是当前 attention 最大差距，max 约 `184%`，且有大量物理下界违反。该路径需要专门结合 `kv_quant_sparse_flash_attention` 源码和 tiling 处理，不能继续套普通 FA 模型。
- `qwen7b MatMulV2` 在当前 910B4 配置下所有大 shape 样本都出现物理下界违反，说明硬件带宽/平台配置、shape 解析或物理存储假设至少有一项不匹配。该问题优先级高于普通拟合。
- `longcat_attention` 没有进入大 shape/核打满集合，当前不作为 tiling 准确率核心样本。

## 下一步优先级

1. 检查 `qwen7b MatMulV2` 的平台和物理流量假设：它按 Cube/Vector block dim 匹配 910B4，但报告中 `ideal_lower_bound_us > duration_us` 全量出现，需要核对 HBM 带宽、profiling 单位、shape 存储解释和 MatMulV2 kernel 路径。
2. 为 `GroupedMatmul` 建立低置信度的 routing-aware 模型边界：没有 `group_list` 值时，不应把全专家权重流量计入单次执行；需要读取源码和 profiling 可用字段，决定是否能估计活跃专家或只能标残留。
3. 为 `KvQuantSparseFlashAttention` 建立专用模型：读取 `ops-transformer` 中 `kv_quant_sparse_flash_attention` 的 host tiling 和 kernel 分支，拆分 kv quant、sparse、MLA absorb、mask/aux、workspace 和实际访问流量。
4. 对 910C/910B4 `MatMulV2 M=1` 长 K/N 路径做源码/tiling 检查，避免继续用普通 fallback tiling 解释 decode-like GEMV/GEMM 边界路径。
