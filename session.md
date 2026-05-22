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
