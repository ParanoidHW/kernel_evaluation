# Attention Kernel 评估设计文档

## 目标

本文档描述本仓库 Attention 类 kernel 评估器的设计原则、建模边界、源码依据和当前残留问题。目标不是对历史 profiling 样本做经验拟合，而是结合 `ops-transformer` 的 kernel/tiling 实现和昇腾硬件约束，构建可解释、可迭代的耗时模型。

当前支持的 attention 范围包括：

- `FlashAttentionScore`
- `FusedInferAttentionScore`
- `PromptFlashAttention`
- `IncreFlashAttention`
- `KvQuantSparseFlashAttention`

## 设计原则

- `estimated_us` 表示当前 kernel 路径的运行时间估计。
- `ideal_lower_bound_us` 只表示物理下界，不包含模板固定开销、同步开销和小 kernel 启动地板。
- 优化目标是降低最大相对误差和 p95 相对误差，而不是只看中位数。
- 每个额外成本项都必须能回溯到源码路径、tiling 约束或硬件执行机制，禁止为了拟合个别样本引入无关旋钮。

相对误差定义：

```text
relative_error = abs(estimated_us - duration_us) / max(duration_us, eps)
```

## 输入事实与信息边界

profiling 输入主要来自 `kernel_details.csv`。当前可稳定获得的信息包括：

- 算子类型、名称、duration、Block Dim
- 输入/输出 shape、dtype、format
- 部分 profiling counter

但以下信息通常缺失或不完整：

- host tiling 的完整输出
- 真实模板选择结果
- runtime 中间状态和 workspace 访问次数
- 稀疏索引、block table、group/split 运行时内容

因此 Attention 评估器必须显式区分：

- `actual_tiling_source=ops_transformer_source_strategy_replay`：根据本地 `ops-transformer` 源码和 shape/类型命中的策略级 replay。
- `actual_tiling_source=unavailable_ops_transformer_replay`：本地源码不可用，只能退化为通用解析。
- `fallback_tiling_source=analytic_attention_bound`：仅用于缺失真实策略信息时的保守估算。

当前 replay 是“源码策略回放”，不是 host tiling 二进制级精确 replay。

## 模型分层

Attention 当前 kernel 估计由以下分量组成：

- `compute_us`：QK/PV 等 Cube 主计算成本。
- `vector_us`：softmax、逐元素、归约等 AIV 相关工作量。
- `hbm_us`：Q/K/V/output 及额外 workspace 的最小或放大后 GM/HBM 访问成本。
- `launch_overhead_us`：kernel 启动固定成本。
- `sync_overhead_us`：split-fuse、分块同步和阶段切换成本。
- `latency_floor_us`：decode、小序列或模板切换导致的最小时延地板。
- `template_overhead_us`：源码可见模板路径的固定额外成本。

最终：

```text
estimated_us = max(current_kernel_body_us, latency_floor_us) + launch_overhead_us + template_overhead_us
```

其中 `current_kernel_body_us` 由 compute、vector、HBM 和 sync 相关分量共同决定。

## 源码依据

当前 attention 设计主要依赖以下本地源码目录：

- `ops-transformer-master/attention/fused_infer_attention_score`
- `ops-transformer-master/attention/flash_attention_score`
- `ops-transformer-master/attention/incre_flash_attention`
- `ops-transformer-master/attention/prompt_flash_attention`
- `ops-transformer-master/attention/kv_quant_sparse_flash_attention`

已纳入模型的关键源码事实包括：

- fused-infer split-fuse 路径中的 `Q_TILE_CEIL=128`
- decode/prefill 路径的不同模板和同步特征
- `MAX_KV_STACK_LEN=512` 等策略级常量
- MQA/GQA、mask/aux、prefill/decode 的分支差异
- QSFA 的 PA cache 形状解释、`sInnerSize=512`、workspace/cache 公式和 K/V 低比特 dtype 约束

## 算子细分设计

## FIA / FlashAttention / Prompt / Incre

常规 attention 路径当前使用统一的解析框架：

1. 从 Q/K/V/output shape 推导 `q_seq`、`kv_seq`、`head_num`、`head_dim`。
2. 根据算子名和输入特征识别 decode/prefill、MQA/GQA、mask/aux 等路径。
3. 用 QK/PV FLOPs、softmax/vector 工作量和 HBM 最小流量建立物理下界。
4. 在当前 kernel 估计中叠加 occupancy、traffic amplification、sync、latency floor 和 template overhead。

当前已知结论：

- 910C attention 精度已相对稳定，长 prefill `FlashAttentionScore` 尾部在当前模型下约为 10% 量级。
- 910B4 attention 主要残留曾集中在 `FusedInferAttentionScore` decode 小 kernel，后续已通过平台专用 latency floor 缩小误差。

## KvQuantSparseFlashAttention

`KvQuantSparseFlashAttention` 不能套普通 FA/FIA 形状和流量模型，必须按专用 kernel 路径处理。

当前 QSFA 设计依据：

- query 支持 `BSND/TND`
- KV 支持 `BSND/TND/PA_BSND`
- PA 场景中 `s2Size = block_table.dim1 * block_size`
- `gSize = n1Size / n2Size`
- query/output 为 FP16/BF16，K/V 可为 INT8/FP8/HIFLOAT8
- workspace 受 `S2_BASE_SIZE=128`、`D_SIZE=576`、mm1/vec1/bmm2/vec2/topK 缓冲影响

因此当前 QSFA 模型包含：

- 专用 parser，正确解释 PA cache 下的 `kv_seq` 和 `kv_heads`
- K/V 低比特字节计算
- current-kernel 额外 workspace GM 流量

这部分修复后，ds3.2 上 QSFA 的 lower-bound violation 已清零，尾部误差显著下降。

## 平台与配置

Attention 成本对平台差异敏感，至少要区分：

- `Ascend910B4`
- `Ascend910B4-1`
- `Ascend910C`

当前配置中的差异主要体现在：

- `aic_num` / `aiv_num`
- HBM 带宽
- launch overhead
- decode latency floor
- template overhead

其中 `910B4-1` 是针对 qwen3-7b/qwen7b 新增的专用配置，使用 20 AIC / 40 AIV 和 1.6 TB/s HBM。

## 当前精度状态

截至当前仓库状态，Attention 相关主要结论如下：

- 基础 910C attention 精度优于 910B4，尾部主要来自长 prefill `FlashAttentionScore`
- qwen3-7b/qwen7b 在 `910B4-1` 下，FIA decode 小 kernel 残留已通过专用 floor 明显收敛
- QSFA 从“严重 lower-bound violation”收敛到“可解释 residual”

Attention 当前更适合继续做两类工作：

- 补足 exact host tiling / template replay 能力
- 按 kernel 家族继续细化 source-strategy replay

不建议继续做没有源码依据的统一经验系数调整。

## 残留问题

当前 Attention 仍有以下残留：

- source replay 仍是策略级，不是精确 host tiling 输出
- 部分 decode/prefill 路径仍依赖平台专用 latency floor
- 长 prefill `FlashAttentionScore` 的模板/并行细节仍可继续细化
- QSFA 仍缺少 block table / sparse indices 真实运行时数据，无法做到完全精确 replay

这些问题都属于“信息边界”或“源码 replay 深度不足”，不是继续加拟合项就能合理解决的问题。

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

python3 tools/eval_ops.py --op-kind attention \
  --profiling example_profilings/profiling_with_model_code/qwen7b \
  --config configs/ascend_910b4_1.json \
  --output attention_eval_report_qwen_910b4_1.csv \
  --unresolved-output attention_eval_unresolved_qwen_910b4_1.csv
```
