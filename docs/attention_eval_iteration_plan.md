# Attention Kernel 评估迭代计划

## 目标

让 attention kernel 的 `estimated_us` 尽可能贴近 profiling 中的 `duration_us`。优化和验收以最大相对误差为主，不以中位数为主。

相对误差定义：

```text
relative_error = abs(estimated_us - duration_us) / max(duration_us, eps)
```

`ideal_lower_bound_us` 保留为物理下界参考，不是当前 kernel 的优化目标。如果物理下界超过实测时间，通常说明 shape/dtype/最小流量解析存在问题，除非有明确 profiling artifact 证据。

## 当前模型

Attention 报告中主要使用：

- `ideal_lower_bound_us`：最小 compute/vector/HBM 物理下界。
- `estimated_us`：当前 kernel 耗时估计。
- `current_kernel_bound_us`：加入 launch/sync/template floor 前的 kernel body 估计。

当前 kernel 估计包含：

- `ops-transformer` source strategy replay。
- fused-infer split-fuse 代码中的 `Q_TILE_CEIL=128` 和 `MAX_KV_STACK_LEN=512`。
- decode/prefill、mask/aux、MQA/GQA、FlashAttention/FusedInferAttention 策略标签。
- occupancy、traffic amplification、workspace score traffic、sync overhead、latency floor 和 template overhead factor。
- 按 SoC、kernel family、decode/short prefill/long prefill 拆分的 latency/template 参数。

## 迭代循环

1. 为每个 SoC 生成新报告。
2. 计算最大相对误差和 top tail。
3. 修改模型前先分类尾部样本。
4. 只修复有明确源码、硬件或 profiling 解释的 parser/model 问题。
5. 重新生成报告，对比最大相对误差、p95 相对误差和尾部样本。
6. 只有在检查后仍缺少可建模原因时，才把个别样本记录为残留限制。

## 尾部分类

使用以下分类：

- `parser_issue`：Q/K/V/output 维度、dtype 或 format 推断错误。
- `min_bound_violation`：`ideal_lower_bound_us > duration_us`，通常表示最小字节或 ops 统计过高。
- `optional_input_overcount`：FIA/FA 可选 tensor 被当作完整 HBM 流量，但实际可能是标量、metadata、空 tensor 或模板未使用输入。
- `template_factor_mismatch`：某个序列区间误用了另一个区间的 template factor。
- `hardware_counter_mismatch`：profiling counter 显示的瓶颈与模型不一致。
- `unsupported_kernel_path`：source strategy 已识别，但精确 tiling/template 行为尚未建模。
- `measurement_noise`：重复样本方差较高，或 duration 很小导致相对误差放大。
- `accepted_residual`：检查后仍没有明确可建模原因的保留项。

## 高优先级修复点

- Attention 可选输入不能全部按完整 HBM 流量累加。应按 dtype/shape/role 只计入可能高流量的 tensor，并限制 scalar/metadata 过计。
- 如果 `ideal_lower_bound_us > duration_us`，应标记该行并降低对最小流量统计的信任。
- FlashAttention 和 FusedInferAttention 的 template overhead 应按序列区间拆分，而不是只按算子 type。
- 910B4 和 910C 必须分开评估，因为 launch、traffic、template 和硬件吞吐行为不同。

## 最新验证快照

生成报告：

- `attention_eval_report_910b4.csv`
- `attention_eval_report_910c.csv`

最新分析结果：

```text
910B4: rows=246, max_relative_error=0.2946, p95=0.2380, median=0.0965
910C:  rows=256, max_relative_error=0.1155, p95=0.0968, median=0.0263
```

当前 910C 尾部是长 prefill `FlashAttentionScore` compute-bound 样本，最大相对误差约 11.6%，在当前 source replay 模型的不确定范围内。

当前 910B4 尾部是 `FusedInferAttentionScore` decode，规格为 `q_seq=1`、`kv_seq=288`、`head_dim=8192`。该项分类为 `unsupported_kernel_path`，不是未检查异常点：模型已经识别 incremental attention/source strategy 和源码可见 tile 常量，但还没有复现 custom-head-dim 模板的额外 vector/GM 行为。在加入更精确的 host tiling/template replay 前，将其作为残留。

## 验证命令

```bash
python3 tools/eval_ops.py --op-kind attention \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output attention_eval_report_910b4.csv \
  --unresolved-output attention_eval_unresolved_910b4.csv

python3 tools/eval_ops.py --op-kind attention \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output attention_eval_report_910c.csv \
  --unresolved-output attention_eval_unresolved_910c.csv

python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py \
  attention_eval_report_910b4.csv attention_eval_report_910c.csv
```
