# 当前 Profiling 评估差距

本文记录当前已有 profiling 报告在“大 shape / 计算核较充分使用”条件下的评估差距。该视角用于定位 tiling、kernel 实现和硬件建模问题，避免被小 shape 的启动开销主导。

## 基线来源

当前文档使用两层结果：

- 全量基线：`eval_results/20260522T081103Z_0a7ccb1/eval_summary.csv`
- qwen3-7b/qwen7b 最新增量：`eval_results/20260522T085206Z_55f097d/eval_summary.csv`
- `eval_results/LATEST` 当前指向：`20260522T085206Z_55f097d`

说明：

- `20260522T081103Z_0a7ccb1` 覆盖 base 910B4/910C、ds3.2、gemma、longcat 和当时的 qwen inferred 910B4。
- `20260522T085206Z_55f097d` 只覆盖 qwen3-7b/qwen7b 在 `configs/ascend_910b4_1.json` 下的 MatMul/Attention 增量修复结果。
- 因此“当前 qwen 结论”以 `55f097d` 为准；其他模型仍以 `0a7ccb1` 全量基线为准。

## 过滤口径

- 排除 `duration_us < 10us` 的样本。这类样本按启动/调度开销主导处理，不作为 tiling 策略准确率的主要判断依据。
- 大 shape / 核较充分使用样本定义为：
  - `block_dim` 或 `mix_block_dim >= 0.8 * aic_num`，或
  - `cube_utilization_pct >= 70`。
- 误差指标：
  - `relative_error = abs(estimated_us - duration_us) / duration_us`。
  - `duration_over_estimate > 1` 表示低估耗时，`< 1` 表示高估耗时。
- 如果 `ideal_lower_bound_us > duration_us`，优先判为 parser、物理流量、dtype、shape 或硬件配置假设问题，不应通过经验系数拟合。
- `GroupedMatmul` 使用 routing-bound 区间误差，不使用普通 MatMul 单点 `estimated_us` 作为主精度指标。

## 全量基线汇总

以下结果来自 `20260522T081103Z_0a7ccb1`。表中 `large max/p95/median` 是大 shape/核较充分使用集合上的普通相对误差。GMM 单独看下一节的 routing-bound 口径。

| report | rows | large | max | p95 | median | LB violations | top op |
|---|---:|---:|---:|---:|---:|---:|---|
| base 910B4 Attention | 246 | 139 | 0.259 | 0.208 | 0.089 | 0 | FusedInferAttentionScore |
| base 910C Attention | 256 | 160 | 0.116 | 0.099 | 0.025 | 0 | FlashAttentionScore |
| base 910B4 MatMul | 1255 | 974 | 0.613 | 0.476 | 0.075 | 1 | MatMulV2 |
| base 910C MatMul | 672 | 588 | 0.542 | 0.240 | 0.055 | 0 | MatMulV2 |
| ds3.2 910C Attention | 90 | 90 | 0.200 | 0.155 | 0.034 | 0 | KvQuantSparseFlashAttention |
| ds3.2 910C MatMul | 940 | 300 | 0.689 | 0.657 | 0.318 | 0 | QuantBatchMatmulV3 |
| gemma 910B4 Attention | 90 | 75 | 0.416 | 0.236 | 0.152 | 0 | FusedInferAttentionScore |
| gemma 910B4 MatMul | 708 | 618 | 0.516 | 0.192 | 0.072 | 0 | MatMul |
| longcat 910B4 Attention | 28 | 0 | - | - | - | 0 | - |
| longcat 910B4 MatMul | 251 | 59 | 0.311 | 0.179 | 0.080 | 0 | MatMul |

历史 qwen inferred 910B4 行不再作为当前 qwen 结论：该结果使用旧 910B4 HBM 假设，会产生全量 lower-bound violation。当前 qwen 使用 `910B4-1` 专用配置，见后文。

## qwen3-7b/qwen7b 最新结果

以下结果来自 `20260522T085206Z_55f097d`，使用 `configs/ascend_910b4_1.json`：

| report | rows | large | max | p95 | median | LB violations | top op |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen7b MatMul 910B4-1 | 483 | 483 | 0.149 | 0.129 | 0.061 | 0 | MatMulV2 |
| qwen7b Attention 910B4-1 | 96 | 96 | 0.110 | 0.100 | 0.063 | 0 | FusedInferAttentionScore |

关键变化：

- qwen MatMulV2 旧配置下 `483/483` lower-bound violation 已清零。
- qwen MatMulV2 max 从旧 inferred 910B4 的 `0.811` 收敛到 `0.149`。
- qwen Attention max 从旧 inferred 910B4 的 `0.468` 收敛到 `0.110`。
- 当前残留主要是 small-M/N MatMulV2 fallback tiling 与 FIA decode source-strategy replay 的精度边界，不再是平台配置错误。

## GroupedMatmul 独立口径

`GroupedMatmul` 缺少真实 `group_list` 和 `tuningConfigOptional`，因此独立报告不把普通 MatMul 的 `estimated_us` 当作最终判断，而是输出两种 routing 场景边界。

源码依据来自 `ops-transformer-master/gmm/grouped_matmul`：

- host tiling 解析 `groupType`、`groupListType`、`tuningConfigOptional`、`singleN`、`usedCoreNum`。
- kernel 内逐 group 读取 `groupList`，空 group skip。
- group 间调度和 block 分配不能由 `Block Dim` 反推出活跃专家数。

GMM routing-bound 结果来自 `20260522T081103Z_0a7ccb1`：

| report | rows | large | bound max | bound p95 | bound median | positions |
|---|---:|---:|---:|---:|---:|---|
| base 910B4 GroupedMatmul | 208 | 187 | 0.180 | 0.074 | 0.000 | 171 within, 16 above |
| ds3.2 910C GroupedMatmul | 120 | 120 | 0.176 | 0.165 | 0.090 | 120 above |
| gemma 910B4 GroupedMatmul | 180 | 160 | 0.000 | 0.000 | 0.000 | 160 within |
| longcat 910B4 GroupedMatmul | 28 | 28 | 0.179 | 0.157 | 0.040 | 11 within, 17 above |

解释：

- Gemma 大样本全部落在 routing bounds 内。
- Base/Longcat 仍有少量 above-bound，最大约 18%。
- ds3.2 INT8 GMM 全部 above-bound，最大约 17.6%，需要 exact group/tuning 或 quant GMM 源码分支继续解释。
- 这些不是普通 MatMul lower-bound violation，应按 GMM 区间误差追踪。

## 当前主要差距分类

## 1. ds3.2 QuantBatchMatmulV3

已完成一轮 `QuantBatchMatmulV3` Weight-NZ epilogue 建模：

- 源码依据：arch35 `BlockMmadA8W8FixpipeQuant`，per-channel scale 按 N tile 进入 L1，fixpipe 输出阶段消费 scale/quant 参数，且该路径 `disableGemv=true`。
- 触发范围：`QuantBatchMatmulV3`、Weight-NZ、`per_channel_n`、`full_quant`、small-M、N tile 较多。
- 语义：只增加 current-kernel `template_overhead_us`，不修改 `ideal_lower_bound_us`。

本轮增量验证使用 `/tmp/ds32_matmul_quant_epilogue2.csv`：

- ds3.2 large MatMul max：从 `0.689` 降到 `0.572`
- ds3.2 `QuantBatchMatmulV3` large max：从 `0.689` 降到 `0.194`
- ds3.2 `QuantBatchMatmulV3` large p95：从 `0.661` 降到 `0.174`
- lower-bound violation：`0`

判断：

- `QuantBatchMatmulV3` 不再是 ds3.2 large MatMul 第一残差。
- 当前 ds3.2 large MatMul 第一残差转为 `TransposeBatchMatMul M=4,N=128,K=512,batch=128`，max 约 `0.572`。
- 已检查源码：未发现独立的 transpose 临时连续化 DataCopy kernel；transpose 由 `perm_x1/perm_x2` 模板、`SetTensorA/B(..., isTrans)`、`SetOrgShape` 和 GM offset 计算进入 MatmulImpl 装载路径。profiling 中 MTE2/fixpipe/scalar 时间明显高于 MAC 时间，符合 strided/transpose 访问与输出布局处理开销。该项当前记为遗留，有条件获取 perm attrs 或 exact tiling 后再处理。
- `full_quant_with_dequant` BF16 输出路径没有套用该项，因为大 K/N 路径原本已主要由 output/dequant/HBM 项覆盖，强行套用会导致高估。

## 2. base 910B4/910C MatMulV2 small-M

base 910B4/910C MatMul 的 top gap 都是 `MatMulV2 M=1` 类 small-M 路径：

- base 910B4 MatMul max：`0.613`
- base 910C MatMul max：`0.542`
- 910B4 仍有 1 个 lower-bound violation，需要单独检查该行 parser/traffic。

判断：

- qwen 专用 `910B4-1` small-M/N MatMulV2 已收敛，但该参数不应直接外推到 base 910B4/910C。
- base MatMulV2 small-M 仍需要从 MatMulV2/MatMulCommon 源码和 host tiling 路径继续建模。

## 3. gemma Attention FIA decode

gemma 910B4 Attention：

- large max：`0.416`
- p95：`0.236`
- median：`0.152`
- top op：`FusedInferAttentionScore`

判断：

- 这是 910B4 FIA decode/source-strategy replay 的残留。
- qwen 专用 floor 已收敛 qwen，但 gemma 不应直接套 qwen floor。
- 需要按 FIA decode 模板、head_dim、kv_seq、mask/aux 和平台 launch/latency floor 继续拆分。

## 4. ds3.2 KvQuantSparseFlashAttention

QSFA 已从旧基线的严重错误收敛到：

- large max：`0.200`
- p95：`0.155`
- median：`0.034`
- lower-bound violation：`0`

判断：

- 当前已经不是普通 attention parser 错误。
- 剩余 tail 属于 QSFA source-strategy replay residual，需要真实 block table、sparse indices、topK/PA workspace 访问次数或 exact tiling 才能继续收敛。

## 5. GMM above-bound residual

GMM 当前最大约 18% above-bound：

- base/longcat：少量 above-bound
- ds3.2：全部 above-bound

判断：

- 不能通过扩大经验区间或调高 scheduler_us 来“拟合”。
- 下一步需要真实 `group_list`、`groupListType`、`tuningConfigOptional`，或继续拆 arch35 quant/weight-quant/adaptive sliding window/tail split。

## 当前结论

- qwen3-7b/qwen7b 平台问题已解决：当前使用 `Ascend910B4-1`，HBM 1.6 TB/s，MatMul/Attention lower-bound violation 均为 0。
- Attention 当前最大已知残留不再是 QSFA parser，而是 gemma/base 910B4 FIA decode 与 QSFA exact replay 深度不足。
- MatMul 当前最大活动建模残留是 base `MatMulV2 M=1` small-M 路径；ds3.2 `TransposeBatchMatMul` transpose/strided 访问残差先作为遗留。
- GMM 应继续使用 routing-bound 口径；没有真实 group runtime 数据前，不应按普通 MatMul 单点误差做结论。

## 下一步优先级

1. base 910B4/910C `MatMulV2 M=1`：从 MatMulV2/MatMulCommon host tiling 与小 M/N policy 出发，区分平台和 kernel 模板，不复用 qwen 专用参数。
2. gemma/base FIA decode：继续拆 `FusedInferAttentionScore` decode 模板、head_dim/kv_seq/mask/aux 和平台 latency floor。
3. GMM：优先接入真实 `group_list` / `tuningConfigOptional`；缺失运行时数据时，只维护 routing bounds，不加无源码依据的校准项。
4. QSFA：若能拿到 block table/sparse indices 或 exact tiling，再继续收敛 20% tail；否则作为 source-strategy replay residual 跟踪。
5. ds3.2 `TransposeBatchMatMul M=4,N=128,K=512,batch=128`：当前按 transpose/strided 访问遗留跟踪；需要 perm attrs、exact tiling 或更细硬件计数器后再恢复建模。
