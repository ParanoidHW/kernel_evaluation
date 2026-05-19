---
name: kernel-eval-review
description: 用于检视本仓库内任意 Ascend/CANN kernel 评估方案、实现、报告和误差分析；重点检查是否正确结合 kernel 源码、tiling 实现和硬件架构，是否存在解析错误、模型语义混淆或最大相对误差尾部未解释。
---

# Kernel 评估检视

在需要 review 某类 kernel 评估方案、代码实现、报告结果或误差迭代结论时使用本 skill。检视对象可以是 MatMul、Attention、Elementwise、Reduction、CV、Quant 或其他 CANN 算子族。

本 skill 默认只做检视和提出 findings，不直接修改代码。若用户明确要求修复，再进入 `kernel-eval-iteration`。

## 检视目标

检视重点不是代码风格，而是评估模型是否可信：

- profiling 解析是否正确还原逻辑规格和物理存储。
- kernel 实现路径、host tiling 路径和模板分支是否有源码依据。
- 硬件架构假设是否明确，是否和配置文件一致。
- `actual_tiling`、`fallback_tiling`、`optimal_tiling` 是否语义分离。
- `estimated_us` 是否代表当前 kernel，`ideal_lower_bound_us` 是否只作为物理下界。
- 最大相对误差 tail 是否已分析、分类和解释。

## 输入材料

优先检查以下材料：

- 方案文档：`docs/` 下对应算子族设计或迭代文档。
- 实现代码：`tools/<family>_eval/`、`tools/op_eval/`、配置文件和报告写入逻辑。
- profiling 报告：`*_eval_report_*.csv` 和 `*_unresolved_*.csv`。
- 误差分析：`kernel-eval-iteration/scripts/analyze_report_errors.py` 输出。
- 源码依据：本地 `ops-nn`、`ops-transformer`、`ops-math`、`ops-cv` 或对应下载目录。
- 硬件配置：`configs/ascend_*.json` 和 `docs/info.md`。

## 检视流程

1. 明确检视范围：方案、实现、报告、误差迭代，或全量检视。
2. 检查源码链路：profiling `Type` 是否能映射到 op_host、tiling、device kernel 和模板分支。
3. 检查硬件假设：AI Core/AIV、HBM、cache/UB/L0、Cube/Vector 峰值、fixpipe、sync、launch 是否有来源。
4. 检查 parser：shape、dtype、format、layout、aux/mask/scale/bias/workspace、物理存储元素和 unresolved 输出是否合理。
5. 检查模型语义：当前 kernel 估计、fallback 估计和物理下界是否混用。
6. 检查报告字段：是否足以解释 tiling 来源、瓶颈、confidence、diagnosis、残差和 tail。
7. 检查误差结论：必须查看最大相对误差、p95、top tail；不能只用中位数证明模型有效。
8. 检查代码质量：算子族边界、公共层注册、配置覆盖、回归兼容和测试命令是否完整。
9. 输出 findings，按严重程度排序，并给出下一步建议。

## 严重程度

- `blocking`：会导致模型结论不可信，例如 parser 解析错、物理下界超过实测但未解释、把 fallback 当 actual。
- `high`：会显著影响最大相对误差或主要 tail，例如遗漏关键 kernel 模板、硬件参数错误、可选输入严重过计。
- `medium`：影响部分场景可信度，例如报告字段不足、diagnosis 不完整、某类 tail 未分类。
- `low`：文档、命名或维护性问题，不直接影响当前估计正确性。

## 输出格式

检视输出必须 findings 优先，按严重程度排序。每条 finding 应包含：

```text
[severity] 标题
位置：文件或报告行
证据：源码、报告字段、误差数据或配置假设
影响：为什么会影响评估可信度或最大相对误差
建议：下一步应修改、验证或记录为残留
```

如果没有发现问题，明确说明“未发现 blocking/high finding”，并列出仍存在的 residual risk 或测试缺口。

## 红线

- 不接受“中位数接近”作为充分结论。
- 不接受没有源码依据的具体 tiling/template 声明。
- 不接受把 per-shape 拟合参数伪装成 kernel 机制。
- 不接受未分类的极端 tail 直接作为残留。
- 不接受将硬件未知参数静默写死到模型代码中。
