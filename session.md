# 会话记录

## 2026-05-22 当前仓库状态

本仓库是一个昇腾 kernel 评估工具，输入为导出的 profiling CSV，输出为按算子族拆分的 kernel 耗时估计与误差汇总。当前主要覆盖：

- `MatMul`
- `GroupedMatmul`
- Attention 家族，包括 `FlashAttentionScore`、`FusedInferAttentionScore`、`PromptFlashAttention`、`IncreFlashAttention`、`KvQuantSparseFlashAttention`

整体设计原则已经明确：

- 优先基于 `ops-nn` / `ops-transformer` 源码和 tiling 逻辑建模
- `estimated_us` 与 `ideal_lower_bound_us` 严格分离
- 不做脱离 kernel 机制的经验拟合

## 仓库结构认知

- `tools/op_eval/`：共享 API、profiling 解析、CSV 输出、CLI 注册
- `tools/matmul_eval/`：MatMul / GroupedMatmul 的 parser、tiling、成本模型与评估逻辑
- `tools/attention_eval/`：Attention 的 parser、source replay、成本模型与评估逻辑
- `configs/`：平台配置，包括 `910B4`、`910B4-1`、`910C`
- `example_profilings/`：已有 profiling 样本
- `eval_results/`：基线刷新汇总
- `docs/`：架构、设计和误差分析文档

## 平台识别结论

- 910B4 的典型 block dim 证据是 Cube 约 `20`、Vector 约 `40`
- 910C 的典型 block dim 证据是 Cube 约 `24`、Vector 约 `48`
- 不能拿全局最大 `Block Dim` 直接判定 MatMul / Attention 平台，因为很多最大值来自 Vector 算子
- qwen3-7b/qwen7b 按 block dim 原则仍归到 910B4 系列

## qwen3-7b / qwen7b 平台处理

已新增专用配置：

- `configs/ascend_910b4_1.json`

配置结论：

- `soc = Ascend910B4-1`
- `aic_num = 20`
- `aiv_num = 40`
- `hbm_bandwidth_tbps = 1.6`

该配置专用于 qwen3-7b/qwen7b。

## 近期建模与精度结论

### 1. GroupedMatmul

- `GroupedMatmul` 不能继续按普通 MatMul 的理想下界做精度判断
- profiling 中缺少真实 `group_list`，因此改为建模“均衡路由”和“极端不均衡路由”两个边界
- 已独立作为 `grouped_matmul` 算子族评估

### 2. KvQuantSparseFlashAttention

- QSFA 不能套普通 Attention parser 和流量模型
- 已基于 `ops-transformer-master/attention/kv_quant_sparse_flash_attention` 做专用 parser 和 current-kernel workspace 建模
- 修复后 ds3.2 上的 lower-bound violation 已清零，误差明显下降

### 3. qwen MatMulV2

- qwen 在旧 910B4 配置下出现全量 lower-bound violation，本质是平台带宽配置不匹配
- 切换到 `910B4-1` 后 lower-bound violation 归零
- 后续针对 `M=1/N=1` 的 small-M 路径，补充了源码可解释的 current-kernel overhead
- 当前 qwen MatMulV2 尾部已显著收敛

### 4. qwen Attention

- qwen Attention 的主要残留来自 `FusedInferAttentionScore` decode 小 kernel
- 已在 `910B4-1` 配置中加入平台专用 decode latency floor
- 当前 qwen Attention 尾部已明显收敛

## 最新基线状态

最新基线目录：

- `eval_results/LATEST`

当前已知结论：

- qwen MatMulV2：lower-bound violation 为 `0`
- qwen Attention：lower-bound violation 为 `0`
- 910C attention 基线整体优于 910B4
- QSFA 已从“模型错误”收敛为“仍需更深 replay 的残留”

## 已完成的关键提交

- `83a39c6 [feat] add 910B4-1 qwen config`
- `f36d64d [doc] refresh qwen 910B4-1 baseline`
- `55f097d [feat] model qwen small kernel floors`
- `b4435e8 [doc] refresh qwen residual baseline`

## 当前残留问题

- Attention 仍主要是 source strategy replay，而不是 exact host tiling replay
- 部分 decode/prefill 路径仍依赖平台专用 latency floor
- QSFA 仍缺 block table / sparse indices 等真实运行时信息
- MatMulV2 小 shape 路径若继续细化，仍需直接回到 MatMulCommon 和 host tiling 源码

## 本次文档整理

本次已完成：

- 删除仓库根目录 `hardware.log`
- 根 `README.md` 改为中文
- `docs/` 下英文入口文档删除并整理为中文说明
- `docs/attention_eval_iteration_plan.md` 调整为 `docs/attention_kernel_eval_design.md`
- `session.md` 与本地文档统一为中文

## 2026-05-22 文档方案重构

本次按当前实现重新组织 kernel 评估设计文档：

- 重写 `docs/attention_kernel_eval_design.md`，明确当前 attention 是 `ops-transformer` source strategy replay，不是 exact host tiling replay；补充 parser、QSFA、current-kernel cost equation、报告字段和限制。
- 重写 `docs/matmul_eval_design_zh.md`，聚焦普通 MatMul、BatchMatMul、TransposeBatchMatMul 和量化 MatMul；将 GMM 精度口径从 MatMul 文档中拆出。
- 新增 `docs/gmm_eval_design_zh.md`，说明 `GroupedMatmul` 基于 routing bounds 评估，使用 balanced/extreme 两个场景，并以 `gmm_duration_position` 和区间误差作为主口径。
- 重写 `docs/architecture.md`，按当前 CLI/API 注册、actual/fallback/optimal 语义、MatMul/GMM/Attention 内部链路、报告字段和基线流程更新。
- 更新 `README.md`，补充 GMM 设计文档入口。

当前文档与实现对应关系：

- `tools/op_eval/api.py` 当前只注册 `matmul`、`grouped_matmul`、`attention`。
- `tools/matmul_eval/gmm_model.py` 当前提供 GMM routing bounds，但没有 exact group_list replay。
- `tools/attention_eval/tiling_replay.py` 当前提供 source strategy replay，不声称获取二进制 tiling data。
- `tools/matmul_eval/kernel_model.py` 当前区分 `runtime_kb_exact`、`advanced_tiling_heuristic` 和 `analytic_search`。

## 2026-05-22 current gap 文档刷新

本次刷新 `docs/current_eval_gap_zh.md`：

- 将 gap 文档改为“两层基线”口径：全量基线使用 `eval_results/20260522T081103Z_0a7ccb1/eval_summary.csv`，qwen 最新增量使用 `eval_results/20260522T085206Z_55f097d/eval_summary.csv`。
- 明确 `eval_results/LATEST` 当前指向 `20260522T085206Z_55f097d`，但该目录只覆盖 qwen3-7b/qwen7b 增量，不覆盖全部模型。
- 删除旧结论中“qwen MatMul 全量 lower-bound violation”和“QSFA 严重 lower-bound violation”的当前问题表述，改为记录它们已经分别收敛到当前残留。
- 当前 gap 优先级更新为：ds3.2 `QuantBatchMatmulV3` small-M/Weight-NZ/dequant，base `MatMulV2 M=1`，gemma/base FIA decode，GMM routing above-bound，QSFA exact replay residual。

## 2026-05-22 profiling CSV 增列工具

本次新增 `tools/annotate_profiling.py`：

- 输入单个 profiling CSV 和硬件配置 JSON。
- 在原 CSV 基础上新增 `kernel_eval_value` 列并写出新 CSV。
- MatMul/Attention 行写入当前工具的 `estimated_us`。
- GroupedMatmul 行默认写入 routing bounds 的区间均值；也保留 `--gmm-value bounds` 可输出 `[low,high]`。
- 非 MatMul/GMM/Attention 行留空，作为后续新增算子族的扩展点。

## 后续要求

- 后续除非用户另行说明，每完成一个新增特性/功能都需要本地提交
- commit 信息格式使用 `[feat/doc/bugfix] xxxx`
- 每轮修改后同步刷新 `session.md`

## 2026-05-25 QuantBatchMatmulV3 Weight-NZ 建模

本轮解决 ds3.2 910C `QuantBatchMatmulV3` small-M / Weight-NZ / full-quant 低估问题：

- 新增 `quant_matmul.weight_nz_epilogue` 配置项，当前在 `configs/ascend_910c.json` 中开启。
- `tools/matmul_eval/api.py` 将该项计入 current-kernel `template_overhead_us`，不改变 `ideal_lower_bound_us`。
- 触发条件限制为 `QuantBatchMatmulV3`、B 为 `FRACTAL_NZ`、`per_channel_n`、`full_quant`、small-M、N tile 较多。
- 源码依据来自 arch35 `BlockMmadA8W8FixpipeQuant`：per-channel scale 按 N tile 进入 L1，fixpipe 输出阶段消费 scale/quant 参数，且该路径 `disableGemv=true`。
- 没有套用到 `full_quant_with_dequant` BF16 输出路径；验证显示该路径大 shape 原本主要由 output/dequant/HBM 覆盖，套用会高估。

验证：

- `python3 -m compileall tools`
- `python3 tools/eval_ops.py --op-kind matmul --profiling example_profilings/profiling_with_model_code/ds3.2/ASCEND_PROFILER_OUTPUT/kernel_details.csv --config configs/ascend_910c.json --output /tmp/ds32_matmul_quant_epilogue2.csv --unresolved-output /tmp/ds32_matmul_quant_epilogue2_unresolved.csv`
- `python3 tools/analyze_large_shape_gap.py /tmp/ds32_matmul_quant_epilogue2.csv /tmp/base910c_matmul_quant_epilogue.csv`

结果：

- ds3.2 `QuantBatchMatmulV3` large max：`0.689 -> 0.194`
- ds3.2 `QuantBatchMatmulV3` large p95：`0.661 -> 0.174`
- ds3.2 large MatMul 总体 max：`0.689 -> 0.572`
- lower-bound violation：`0`

新识别问题：

- ds3.2 large MatMul 第一残差转为 `TransposeBatchMatMul M=4,N=128,K=512,batch=128`，max 约 `0.572`。
- base 910C `MatMulV2 M=1` large max 仍约 `0.542`，需要下一轮按 MatMulV2 small-M 模板继续处理。

## 2026-05-25 TransposeBatchMatMul 残差遗留

根据用户判断，本轮将 ds3.2 `TransposeBatchMatMul M=4,N=128,K=512,batch=128` 记为遗留，有条件再处理。

已检查代码：

- arch35 `transpose_batch_mat_mul` 根据 `PERM_X1/PERM_X2/BATCH_SPLIT` 选择 `BMM_TRANS` / `TRANS_BMM_TRANS` 等模板。
- host tiling 解析 `perm_x1` 和 `perm_x2`，但当前 profiling CSV 不保留这些 attrs。
- kernel 内没有看到独立的 transpose 临时连续化 DataCopy kernel；transpose 通过 `SetTensorA/B(..., isTrans)`、`SetOrgShape` 和 `CalcGMOffset()` 进入 MatmulImpl 装载路径。
- ds3.2 样本硬件计数器显示 `aic_mte2_time`、`aic_fixpipe_time`、`aic_scalar_time` 明显高于 `aic_mac_time`，符合 transpose/strided 访问和输出布局处理开销。

处理结论：

- 不引入无关校准项或按单一 shape 拟合。
- 该项从活动 TODO 降级为遗留，后续需要 perm attrs、exact tiling 或更细硬件计数器后再建模。

## 2026-05-25 剩余 TODO 支撑信息审计

本轮按 review/iteration 口径检查剩余 TODO 是否具备继续完成的必要信息。

结论：

- base 910B4/910C `MatMulV2 M=1`：源码有 `MatmulToMul` policy、`disableGemv`、L1/L0 copy 与 sync 流水等机制线索；但当前 profiling 缺 MatMulV2 exact host tiling、模板 key、L1/L0 分块细节或 runtime KB 命中记录，且 910B4 仍有 1 个 lower-bound violation 未单独解释。若继续按 tail 反推有效吞吐，会变成样本拟合。降级为遗留。
- gemma/base FIA decode：当前 attention evaluator 是 source-strategy replay，不是 exact host tiling replay；profiling 缺 FIA tiling data、decode 模板 key、KV cache/block metadata 和 mask/aux 实际访问规模。继续调 floor 会变成样本拟合。降级为遗留。
- GMM above-bound：profiling 缺真实 `group_list`、`groupListType`、`tuningConfigOptional` 和 per-expert token 分布，无法把 routing bounds 收敛为单次执行估计，也不能为了 above-bound 样本扩大区间。降级为遗留。
- QSFA exact replay：当前缺 block table 实际值、sparse indices、runtime topK 行为和 exact host tiling。现有模型保持 source-strategy replay，不继续添加无运行时输入支撑的稀疏访问校准项。降级为遗留。
- ds3.2 `TransposeBatchMatMul` 已在上一节按 transpose/strided 访问残差降级为遗留。

当前状态：

- qwen3-7b/qwen7b 平台和 910B4-1 配置问题已解决。
- ds3.2 `QuantBatchMatmulV3` Weight-NZ/full-quant 已完成一轮可解释建模。
- 剩余残差在现有 profiling/source 信息下均不具备继续完成的充分支撑，全部转入遗留清单。
- 后续若补齐 exact tiling、runtime metadata 或平台级模板基准，再恢复对应项。
