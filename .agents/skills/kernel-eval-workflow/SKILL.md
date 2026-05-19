---
name: kernel-eval-workflow
description: 用于执行完整 Ascend/CANN kernel 评估工作流，将方案设计、开发实现、检视 review 和误差迭代串联起来；适用于为某一类 kernel 从源码和硬件架构出发建立并收敛评估模型。
---

# Kernel 评估完整工作流

在用户要求从零启动、系统推进或端到端完成某类 kernel 评估器时使用本 skill。本 workflow 编排以下阶段：

- 方案设计：使用 `kernel-eval-design`。
- 开发实现：使用 `kernel-eval-development`。
- 检视 review：使用 `kernel-eval-review`。
- 误差迭代：使用 `kernel-eval-iteration`。

## 总体原则

完整流程必须围绕可解释建模展开：

- 先读 kernel 实现、tiling 实现和硬件配置，再写模型。
- 先定义当前 kernel、fallback 和物理下界的语义边界，再生成报告字段。
- 先看最大相对误差和 top tail，再谈中位数。
- 每轮修改都要能对应到源码路径、tiling 分支、硬件机制或明确的 parser 修复。
- 不能解释的个别 tail 可以保留，但必须经过检视和分类。

## 阶段 0：范围确认

明确以下输入：

- 算子族和 profiling `Type`。
- 目标 SoC，例如 910B4、910C 或其他 Ascend 目标。
- 对应源码仓：`ops-nn`、`ops-transformer`、`ops-math`、`ops-cv`。
- profiling 样例目录和期望输出报告。
- 验收指标，默认以最大相对误差为主。

输出：任务范围、源码路径、profiling 输入和报告文件名。

## 阶段 1：方案设计

触发 `kernel-eval-design`。

必须完成：

- 定位 op_host、tiling、device kernel、模板分支和平台分支。
- 记录关键 tile/block 常量、dtype/format 限制、workspace 和特殊路径。
- 明确硬件架构假设和配置来源。
- 定义 parser、成本模型、报告字段、diagnosis、confidence 和验证计划。
- 将方案写入 `docs/`，或更新已有方案文档。

设计完成门槛：

- 方案能解释 shape 到 tiling 到执行模板的主路径。
- 已说明哪些部分是 actual replay，哪些是 fallback，哪些是 physical lower bound。
- 已列出首轮实现任务和预期验证命令。

## 阶段 2：开发实现

触发 `kernel-eval-development`。

按方案实现或更新：

- `tools/<family>_eval/` 中的 spec/parser/cost/report 模块。
- `tools/op_eval/` 中的公共注册、CLI 和报告入口。
- `configs/` 中的硬件或校准配置。
- `docs/` 中的字段说明和已知限制。

开发完成门槛：

- 代码可以导入或编译。
- 目标 profiling 能生成 resolved/unresolved 报告。
- 报告字段能追踪 tiling 来源、瓶颈、估计分量和 residual。
- unresolved 行保留足够信息供 parser 继续修复。

## 阶段 3：自检验证

运行目标 SoC 报告生成和误差分析：

```bash
python3 tools/eval_ops.py --op-kind <family> \
  --profiling <profiling_dir> \
  --config <config.json> \
  --output <family>_eval_report_<soc>.csv \
  --unresolved-output <family>_eval_unresolved_<soc>.csv

python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py \
  <family>_eval_report_<soc>.csv
```

自检完成门槛：

- resolved/unresolved 数量明确。
- 最大相对误差、p95、p90、中位数和 top tail 已记录。
- 物理下界超过实测的样本已检查。
- top tail 已初步分类。

## 阶段 4：检视 review

触发 `kernel-eval-review`。

检视必须覆盖：

- 源码依据是否充分。
- tiling/template 和硬件机制是否被正确反映。
- parser 和报告字段是否支持定位 tail。
- 当前 kernel 估计、fallback 和物理下界是否混用。
- 最大相对误差 tail 是否有证据链和处理计划。

review gate：

- 有 `blocking` finding 时不能进入验收，只能回到设计或开发。
- 有 `high` finding 时必须修复或明确记录为受限残留。
- 无 `blocking/high` finding 后，才能进入正式误差迭代或结论整理。

## 阶段 5：误差迭代

触发 `kernel-eval-iteration`。

每轮迭代要求：

- 只针对已分类 tail 或明确 parser/model 缺口修改。
- 修改后重新生成报告和误差分析。
- 对比最大相对误差、p95 和 top tail 是否改善。
- 如果改善中位数但恶化最大误差，默认不接受。
- 如果 tail 无法继续解释，必须写入 docs/session 的残留说明。

迭代完成门槛：

- 最大相对误差达到当前可接受范围，或剩余 tail 已被 review 接受为残留。
- 文档、报告和 session 记录一致。

## 阶段 6：交付整理

最终输出应包含：

- 方案文档路径。
- 主要代码路径。
- 报告文件和 unresolved 文件路径。
- 最大相对误差、p95、中位数和 top tail 结论。
- 已接受残留和后续工作。
- 已运行的验证命令。

默认动作：

- 在完成设计、开发、检视、迭代并确认没有 blocking/high 未处理问题后，将本轮代码、文档、配置和 skill 变更提交到本地 git。
- 提交前先检查 `git status --short`，避免把外部源码快照、profiling 大输入、生成报告和缓存文件提交进去。
- commit message 应概括算子族、建模变化和验证结论。

## 回退规则

- 设计阶段发现源码路径不明：回到阶段 0，补齐源码或标注 `source_unavailable`。
- 开发阶段 parser 大量 unresolved：回到阶段 1 或 2，先修 parser。
- review 发现模型语义混淆：回到阶段 1，重写模型边界。
- 迭代阶段 tail 无法解释：回到阶段 4，先检视再决定是否作为残留。
