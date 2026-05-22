# Attention Kernel 评估设计文档

## 范围

本文档描述 `tools/attention_eval/` 当前实现的 Attention kernel 评估方案。评估对象来自 profiling CSV 中 `Type` 字段包含 attention 相关 token 的行，当前覆盖：

- `FlashAttentionScore`
- `FusedInferAttentionScore`
- `PromptFlashAttention`
- `IncreFlashAttention`
- `PagedAttention` 类名称
- `KvQuantSparseFlashAttention`

该评估器的目标是给出当前 kernel 路径的可解释耗时估计，而不是对 profiling 样本做经验拟合。所有 current-kernel 成本项都应能对应到 `ops-transformer` 源码策略、tiling 常量、硬件资源或 profiling 可见事实。

## 当前实现入口

- `tools/attention_eval/common.py`：行过滤、variant 识别、Q/K/V shape 解析、QSFA 专用解析。
- `tools/attention_eval/tiling_replay.py`：基于本地 `ops-transformer` 源码存在性和 shape/type 的策略级 replay。
- `tools/attention_eval/api.py`：物理下界、current-kernel 成本分量和最终 `AttentionCostEstimate`。
- `tools/attention_eval/evaluator.py`：逐行 profiling 评估、报告字段、diagnosis 和 summary。
- `tools/op_eval/api.py`：`op_kind=attention` 注册。

命令入口：

```bash
python3 tools/eval_ops.py --op-kind attention \
  --profiling <profiling_dir_or_csv> \
  --config <configs/ascend_*.json> \
  --output attention_eval_report.csv \
  --unresolved-output attention_eval_unresolved.csv
```

## profiling 解析规则

行过滤只看 `Type`，不看 `Name`。这样可以避免把 attention scope 内的 `Cast`、`Slice`、`Mul` 等辅助算子误判为 attention kernel。

当前 token 集合在 `ATTENTION_TYPE_TOKENS` 中定义，包含：

```text
attention
flashattention / flash_attention
fusedinferattentionscore / fusedinferattention
pagedattention / paged_attention
promptflashattention / prompt_flash_attention
increflashattention / incre_flash_attention
```

普通 attention parser 使用前三个输入作为 Q/K/V：

- 支持一维、二维、三维和四维常见布局。
- 对四维形状，如果最后一维等于 head dim，则按 `[B,H,S,D]` 类布局解释。
- 对三维形状按 `[B,S,D]` 类布局解释。
- 对 mask/aux 等输入，不把静态大 mask 全量计入 active HBM；若 aux tensor 元素数大于 active score window，则按 `score_elements = B * q_heads * q_seq * kv_seq` 截断。

输出的核心逻辑规格为 `AttentionSpec`：

- `batch`
- `q_heads`
- `kv_heads`
- `q_seq`
- `kv_seq`
- `head_dim`
- `value_dim`
- `q_elements / k_elements / v_elements / output_elements`
- `aux_elements / raw_aux_elements`
- `score_elements`
- `layout`
- `variant`
- `causal_or_masked`

## QSFA 专用解析

`KvQuantSparseFlashAttention` 不使用普通 Q/K/V parser。当前实现 `_infer_kv_quant_sparse_attention_spec(...)` 基于 `ops-transformer-master/attention/kv_quant_sparse_flash_attention` 的 PA cache 语义处理：

- query/output 常见为 `[T,N,D]`。
- KV cache 常见为 `[block_num,N,block_size,D]` 或 `[block_num,block_size,N,D]`。
- `block_num` 是 cache 总块数，不是 batch。
- `kv_seq` 优先按 block table 的第二维乘 `block_size` 推断。
- `kv_heads` 从 KV cache 的 head 轴推断。
- query/output 仍按 FP16/BF16 字节计，K/V 按 INT8/FP8/HIFLOAT8 类低比特存储计入。

这部分修复避免了把 PA cache 第一维误当 batch 导致的 QSFA lower-bound violation。

## 源码路径与 replay 语义

当前使用的本地源码路径：

- `ops-transformer-master/attention/flash_attention_score`
- `ops-transformer-master/attention/fused_infer_attention_score`
- `ops-transformer-master/attention/prompt_flash_attention`
- `ops-transformer-master/attention/incre_flash_attention`
- `ops-transformer-master/attention/kv_quant_sparse_flash_attention`

`tools/attention_eval/tiling_replay.py` 中的 replay 是策略级 source replay，不是 CANN host tiling 二进制 replay。它只记录当前 row 对应的源码文件和策略标签。

当本地源码存在时：

```text
actual_tiling_source = ops_transformer_source_strategy_replay
current_tiling_kind = source_strategy_replay
optimal_tiling_source = physical_lower_bound
```

当本地源码不存在时：

```text
actual_tiling_source = unavailable_ops_transformer_replay
fallback_tiling_source = analytic_attention_bound
current_tiling_kind = fallback_tiling
optimal_tiling_source = physical_lower_bound
```

当前策略标签包括：

- kernel variant：`flash_attention`、`fused_infer_attention`、`prompt_attention`、`incremental_attention`、`paged_attention`、`kv_quant_sparse_flash_attention`
- `decode` / `prefill`
- `varlen_or_cross_attention`
- `mask_or_aux`
- `mqa_gqa`
- `d64/d128/d192/d256` 或 `custom_head_dim`
- QSFA 专用：`kv_quant`、`sparse`、`mla_absorb_specialized`

## 物理下界模型

Attention 先计算 `ideal_lower_bound_us`，该字段只作为物理参考，不代表当前 kernel 必须达到。

Cube FLOPs：

```text
qk_flops = 2 * B * q_heads * q_seq * kv_seq * head_dim
pv_flops = 2 * B * q_heads * q_seq * kv_seq * value_dim
flops = qk_flops + pv_flops
compute_us = flops / (peak_tflops * 1e6 * pipeline_efficiency)
```

Vector 工作量：

```text
vector_ops = 4 * score_elements + output_elements
vector_us = vector_ops / (vector_tops * 1e6 * vector_efficiency)
```

最小 GM/HBM 字节：

```text
gm_bytes_min =
  Q_bytes + K_bytes + V_bytes +
  aux_elements * 4 +
  output_bytes +
  score_elements * 4 * score_spill_factor

hbm_us = gm_bytes_min / (hbm_bandwidth_tbps * 1e6)
```

物理下界：

```text
ideal_lower_bound_us = max(compute_us, vector_us, hbm_us)
```

QSFA 的 `Q_bytes/K_bytes/V_bytes` 单独处理：Q 使用 compute dtype，K/V 按低比特存储字节计入。

## Current-kernel 模型

当前 kernel 估计在物理下界之上加入以下实现相关项：

- occupancy：由 `work_tiles / aic_num` 估算，且不低于平台配置中的 `min_occupancy_efficiency`。
- traffic factor：decode/prefill、mask、GQA/MQA、QSFA 分别使用不同 GM 流量放大项。
- workspace bytes：普通 attention 可计入 score workspace；QSFA 额外计入 V-template GM workspace。
- vector multiplier：decode/prefill 分别放大 vector 路径成本。
- sync overhead：按 `kv_block_tiles * sync_us_per_kv_tile` 计入。
- latency floor：decode、小 prefill、长 prefill、QSFA 分别使用平台配置中的最小时延地板。
- template overhead factor：按 variant 和序列区间选择。
- launch overhead：来自 `attention_model.launch_overhead_us` 或 `calibration.launch_overhead_us_by_type`。

当前实现中的块大小：

```text
q_block = 128
kv_block = 512
QSFA kv_block = attention_model.kv_quant_sparse_s2_base_size，默认 128
```

work tile：

```text
q_block_tiles = ceil(q_seq / q_block)
kv_block_tiles = ceil(kv_seq / kv_block)
work_tiles = max(1, batch * q_heads * q_block_tiles * kv_block_tiles)
occupancy_efficiency = clamp(work_tiles / aic_num, min_occupancy_efficiency, 1)
```

current GM 字节：

```text
current_gm_bytes = gm_bytes_min * traffic_factor + workspace_bytes
current_hbm_us = current_gm_bytes / (hbm_bandwidth_tbps * 1e6)
```

current compute/vector：

```text
current_compute_us = compute_us / occupancy_efficiency
current_vector_us = vector_us * vector_multiplier / occupancy_efficiency
current_kernel_bound_us = max(current_compute_us, current_vector_us, current_hbm_us)
```

最终估计：

```text
kernel_with_sync_us =
  current_kernel_bound_us * template_overhead_factor +
  sync_overhead_us

estimated_us =
  launch_overhead_us +
  max(kernel_with_sync_us, latency_floor_us)
```

注意：当前代码没有单独输出 `template_overhead_us`，而是输出 `template_overhead_factor`。因此文档和报告都应使用 `template_overhead_factor` 表达 attention 模板开销。

## QSFA workspace 模型

QSFA current-kernel workspace 由 `_kv_quant_sparse_workspace_bytes(...)` 估算，只进入 `current_gm_bytes`，不进入 `gm_bytes_min`。依据来自 `QSFAMlaTiling::GetWorkspaceSize()` 可见的中间缓存：

- mm1 output
- vector1 output
- bmm2 output
- softmax sum
- vec2 output
- topK aggregation cache
- valid-mte2-size metadata

当前实现使用：

```text
act_core_num = aic_num
preload_num = 2
m_base_size = 128
s_inner_size_align = align(s2_base_size, 32)
head_dim_align = align(head_dim, 32)
cube_m_size = min(q_seq * (q_heads / kv_heads), 128)
```

该项是当前 kernel 路径的 workspace 流量估计，不是物理下界。

## 平台配置

Attention 参数位于 `configs/*.json::attention_model`。当前区分：

- `configs/ascend_910b4.json`
- `configs/ascend_910b4_1.json`
- `configs/ascend_910c.json`

关键配置项：

- `vector_tops`
- `vector_efficiency`
- `launch_overhead_us`
- `decode_latency_floor_us`
- `flash_decode_latency_floor_us`
- `fused_infer_decode_latency_floor_us`
- `short_prefill_latency_floor_us`
- `prefill_latency_floor_us`
- `sync_us_per_kv_tile`
- `decode_traffic_factor`
- `prefill_traffic_factor`
- `mask_traffic_factor`
- `gqa_traffic_factor`
- `workspace_score_factor`
- `kv_quant_sparse_*`
- `template_factor` 和各 variant/seq 区间专用 template factor

`910B4-1` 是 qwen3-7b/qwen7b 专用平台配置，使用 20 AIC、40 AIV 和 1.6 TB/s HBM，并包含 qwen FIA decode 小 kernel 的 latency floor。

## 报告字段

resolved 报告包含：

- profiling 来源：`file`、`line`、`name`、`type`
- 逻辑规格：`variant`、`batch`、`q_heads`、`kv_heads`、`q_seq`、`kv_seq`、`head_dim`、`value_dim`、`layout`
- 输入规模：`q_elements`、`k_elements`、`v_elements`、`aux_elements`、`raw_aux_elements`、`output_elements`、`score_elements`
- profiling counter：`block_dim`、`mix_block_dim`、`aicore_time_us`、`aic_mac_ratio`、`aiv_vec_ratio`、`cube_utilization_pct`
- 物理下界：`compute_us`、`vector_us`、`hbm_us`、`gm_bytes_min`、`ideal_lower_bound_us`
- current kernel：`current_compute_us`、`current_vector_us`、`current_hbm_us`、`current_gm_bytes`、`current_kernel_bound_us`
- current kernel 解释项：`occupancy_efficiency`、`traffic_factor`、`q_block_tiles`、`kv_block_tiles`、`work_tiles`、`sync_overhead_us`、`latency_floor_us`、`template_overhead_factor`
- replay 字段：`actual_tiling_source`、`fallback_tiling_source`、`optimal_tiling_source`、`current_tiling_kind`、`tiling_strategy`、`ops_transformer_source_file`、`tiling_notes`
- 结果字段：`estimated_us`、`residual_us`、`duration_over_estimate`、`bound_type`、`bottleneck`、`diagnosis`、`confidence`

unresolved 报告保留 file/line/type/name/input_shapes/output_shapes，用于 parser 缺口排查。

## diagnosis 与 confidence

当前 diagnosis 由 `tools/attention_eval/evaluator.py::classify(...)` 生成，主要标签：

- variant 名称
- `actual_tiling_unavailable`
- `ops_transformer_source_strategy_replay`
- `masked_or_aux_inputs`
- `decode_like`
- `prefill_like`
- `mqa_gqa`
- `specialized_kv_quant_sparse_path`
- `generic_attention_cost_low_confidence`
- `compute_bound` / `memory_access_bound` / `balanced_bound` / `launch_bound`
- `large_residual`

confidence 规则：

- 本地 source strategy replay 命中时通常为 `medium`
- source 不可用时为 `low`
- QSFA 当前固定为 `low`，因为仍缺 sparse indices、block table 实际值和 exact host tiling 输出

## 当前限制

- source replay 只到策略标签级别，不是 exact tiling replay。
- latency floor 和 template factor 是平台级 current-kernel 参数，不能扩展成 per-shape 拟合曲线。
- 普通 parser 对复杂 layout 的支持仍依赖 profiling shape 的可解释性。
- QSFA 没有真实 block table、sparse indices 和 runtime topK 行为，不能精确复现单次 sparse 访问。
- 如果 `ideal_lower_bound_us > duration_us`，优先检查 parser、dtype、aux 计数和最小流量，不应继续增加 current-kernel 参数。

## 后续任务

- 增加 exact host tiling replay 或导入 op_tiling 数据。
- 将 FIA/FA/PFA/IFA 的 arch22/arch35 模板分支进一步映射到 `tiling_strategy`。
- 为 QSFA 引入 block table / sparse indices 元数据接口，区分真实访问 KV blocks 与 cache 总容量。
- 在误差分析中按 `variant + decode/prefill + head_dim + q_seq/kv_seq` 分组输出 tail。
