---
name: kernel-eval-iteration
description: 用于本仓库内任意算子 kernel 评估模型的验证、误差分析和迭代收敛，尤其是在以最大相对误差而非中位数优化 estimated_us 与 duration_us 时使用；开发实现阶段优先使用 kernel-eval-development。
---

# Kernel 评估迭代

在 `kernel_evaluation` 仓库中处理任意算子族的评估模型验证和误差收敛时使用本 skill，包括但不限于 `matmul`、`attention` 以及后续新增算子。若任务仍处于方案设计阶段，先使用 `kernel-eval-design`；若仍处于代码落地阶段，先使用 `kernel-eval-development`。

## 目标

把当前 kernel 的 `estimated_us` 尽量贴近 profiling 中的 `duration_us`。主要优化指标是最大相对误差，而不是中位数。

```text
relative_error = abs(estimated_us - duration_us) / max(duration_us, eps)
```

`ideal_lower_bound_us` 只作为物理下界参考，不是当前 kernel 的耗时预测。如果物理下界超过实测时间，优先怀疑解析、最小流量或 dtype/shape 统计错误。

## 工作流

1. 先阅读对应算子文档，例如 `docs/architecture.md`、`docs/matmul_eval_design_zh.md`、`docs/attention_eval_iteration_plan.md`。
2. 明确本轮目标：降低最大相对误差、修复 unresolved、补齐 kernel 路径，或更新硬件/配置假设。
3. 重新生成目标 SoC 和目标算子族的报告。
4. 用 `scripts/analyze_report_errors.py` 分析最大误差、p95、p90、中位数和 top tail。
5. 修改模型前先分类尾部样本：解析问题、物理下界问题、可选输入过计、tiling/template 不匹配、硬件计数器矛盾、未支持 kernel 路径、测量噪声或可接受残留。
6. 只做能对应到 kernel 机制或硬件机制的最小修改，避免为了中位数引入 per-shape 拟合。
7. 重新生成报告并比较最大相对误差、p95 和 top tail 是否改善。
8. 把设计、开发、验证、残留问题写回 `docs/` 和 `session.md`。

## 昇腾平台匹配检查

对新增 profiling 网络做误差迭代前，先确认使用的硬件配置是否与 profiling 平台一致。该检查只适用于 Ascend/CANN 数据。

- 优先读取目录 README、profiling 元数据和用户说明中的平台信息。
- 若平台信息缺失或可疑，用 `Block Dim`/`Block Num` 做交叉验证：Cube 类算子匹配 `aic_num`，Vector 类算子匹配 `aiv_num`。
- 910B4 的典型证据是 Cube 最大 block dim 约 `20`、Vector 最大 block dim 约 `40`。
- 910C/A3 的典型证据是 Cube 最大 block dim 约 `24`、Vector 最大 block dim 约 `48`。
- matmul/FA 评估必须优先使用 Cube 侧证据选择 SoC；Vector 侧只用于交叉验证，不得把 Vector 最大 block dim 当作 Cube 核数。
- 如果发现此前报告使用了错误 SoC，应先重跑正确 SoC 的报告，再分析最大相对误差。

## 常用命令

MatMul 示例：

```bash
python3 tools/eval_ops.py --op-kind matmul \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report_910b4.csv \
  --unresolved-output matmul_eval_unresolved_910b4.csv
```

Attention 示例：

```bash
python3 tools/eval_ops.py --op-kind attention \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output attention_eval_report_910c.csv \
  --unresolved-output attention_eval_unresolved_910c.csv
```

通用误差分析：

```bash
python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py \
  matmul_eval_report_910b4.csv attention_eval_report_910c.csv
```

## 建模约束

- 不以中位数作为主要验收标准；必须查看最大相对误差和 top tail。
- 不用宽泛经验系数掩盖坏点；新增项应能解释为 launch、occupancy、traffic、workspace、sync、tiling、template、format 或 quant/dequant 等机制。
- 保持 `actual_tiling`、`fallback_tiling`、`optimal_tiling` 三类语义分离。
- 保持当前 kernel 估计与物理下界分离；不要把 `ideal_lower_bound_us` 当作 `estimated_us`。
- 910B4 和 910C 分开看。AI Core 数、HBM、模板路径和 launch/traffic 行为都可能不同。
- 个别尾部如果没有足够信息继续建模，可以作为残留，但必须先给出检查记录和分类原因。

## 当前已知残留

- Attention 910B4：`FusedInferAttentionScore q_seq=1 kv_seq=288 head_dim=8192` 的 custom-head-dim decode 路径仍有约 29.5% 最大相对误差。当前分类为 `unsupported_kernel_path`，需要更精确的 host tiling/template replay 才能继续收敛。
- Attention 910C：当前最大相对误差约 11.6%，主要来自长 prefill `FlashAttentionScore`，在现有 source-strategy replay 精度范围内暂可接受。
