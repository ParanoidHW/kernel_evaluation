# GroupedMatmul Kernel 评估设计文档

## 范围

本文档描述 `GroupedMatmul` 的独立评估方案。GMM 是 MoE 专家路由 kernel，不能直接按普通 MatMul 的单点 `estimated_us` 或 `ideal_lower_bound_us` 判断精度，因为 profiling CSV 通常不包含真实 `group_list` 值和每个专家的 token 分布。

当前独立入口：

```bash
python3 tools/eval_ops.py --op-kind grouped_matmul \
  --profiling <profiling_dir_or_csv> \
  --config <configs/ascend_*.json> \
  --output grouped_matmul_eval_report.csv \
  --unresolved-output grouped_matmul_eval_unresolved.csv
```

实现上 GMM 仍复用 `tools/matmul_eval/evaluator.py` 的 profiling 读取和通用字段输出，但通过 `tools/matmul_eval/gmm_model.py` 增加 routing-aware bounds 字段。

## 源码依据

GMM 的实现依据来自 `ops-transformer-master/gmm/grouped_matmul`，不是 `ops-nn/matmul`。

关键路径：

- `ops-transformer-master/gmm/grouped_matmul/op_api`
- `ops-transformer-master/gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp`
- `ops-transformer-master/gmm/grouped_matmul/op_host/op_tiling/arch35/*`
- `ops-transformer-master/gmm/grouped_matmul/op_kernel/grouped_matmul.h`
- `ops-transformer-master/gmm/grouped_matmul/op_kernel/arch35/*`
- `ops-transformer-master/gmm/common/cgmct/block/block_scheduler_grouped_matmul_aswt.h`
- `ops-transformer-master/gmm/common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h`

当前模型使用的源码事实：

- host tiling 会解析 `groupType`、`groupListType`、`tuningConfigOptional`、`singleN`、`usedCoreNum`。
- kernel 侧按 group 遍历 `groupList`，空 group 会被跳过。
- block 调度在 group 间轮转，不能从 profiling 的 `Block Dim` 反推出活跃专家数。
- quant 路径可按 `aicNum` 设置 block dim，因此 `Block Dim` 主要说明 launched Cube cores，不说明路由分布。
- arch35 下存在 no-quant、quant、weight-quant、adaptive sliding window、tail split 等多条路径，当前尚未 exact replay。

## profiling 解析规则

`op_kind=grouped_matmul` 会先按 MatMul 行规则读取 profiling，然后只保留：

```text
Type.lower() == groupedmatmul
```

GMM 专用 shape 解析位于 `infer_grouped_matmul_spec(...)`：

- A 输入必须按 ND 解释。
- B/weight 输入必须按 `FRACTAL_NZ` 解释。
- weight 的首维是专家数，不作为普通 batch 乘入 FLOPs。
- `batch` 固定为 1。
- M 来自 A 的 token 数。
- K 来自 A 与 weight 的 K 维匹配。
- N 来自 weight/output 的 N 维。
- storage elements 仍保留完整输入/权重/output 物理元素数。

如果 shape/format 无法满足上述条件，行进入 unresolved 报告。

## 为什么不能用普通 MatMul 口径

普通 MatMul 假设每次 kernel 访问完整 A/B/C 逻辑矩阵。GMM 不满足这个假设：

- 每个 token 只路由到部分专家。
- 单次执行中实际活跃专家数由 `group_list` 决定。
- weight 存储包含全部专家，但 kernel 可能只读取非空专家的 weight slice。
- 极端情况下所有 token 落在一个专家，权重流量小但并行度差。
- 均衡情况下多个专家活跃，权重流量大但并行度更好。

因此 GMM 不能用一个普通 MatMul lower bound 代表真实执行。当前报告保留普通字段用于兼容，但精度判断应使用 GMM routing bounds。

## Routing bounds 模型

`tools/matmul_eval/gmm_model.py` 输出两种场景：

### balanced

tokens 尽量均衡分布到可用专家：

```text
active_experts = min(expert_count, M)
tokens_per_active_expert = ceil(M / active_experts)
```

该场景通常代表更高权重流量和更高并行度。

### extreme_imbalance

所有 tokens 落到一个专家：

```text
active_experts = 1
tokens_per_active_expert = M
```

该场景通常代表最低权重流量和较差并行度。

两者共同构成当前 profiling 信息下可解释的成本区间。

## 成本模型

专家数：

```text
expert_count = weight_shape[0]  # len(weight_shape) >= 5
```

单专家权重：

```text
full_weight_elements = b_storage_elements or K * N * expert_count
weight_elements_per_expert = full_weight_elements / expert_count
weight_bytes_per_expert = quant_storage_bytes(weight_elements_per_expert, B_dtype)
```

场景 GM 字节：

```text
gm_bytes =
  A_bytes +
  active_experts * weight_bytes_per_expert +
  output_bytes +
  quant_aux_bytes
```

并行效率：

```text
token_tiles = ceil(tokens_per_active_expert / 16)
n_tiles = ceil(N / 128)
work_tiles = active_experts * token_tiles * n_tiles
rounds = ceil(work_tiles / aic_num)
core_efficiency = work_tiles / (rounds * aic_num)
```

compute：

```text
aligned_m = active_experts * align(tokens_per_active_expert, 16)
aligned_flops = 2 * aligned_m * align(N, 16) * align(K, 16)
compute_us =
  aligned_flops * operation_factor /
  (peak * 1e6 * core_efficiency * quant_or_dtype_efficiency)
```

HBM：

```text
hbm_us = gm_bytes / (hbm_bandwidth_tbps * 1e6)
```

调度项：

```text
scheduler_us = max(0, expert_count - aic_num) * 0.04
```

该项来自源码可见的 groupList 遍历和空 group skip 行为，是专家数超过 Cube cores 时的轻量调度成本，不代表 per-shape 拟合。

场景总成本：

```text
scenario_total_us = max(compute_us, hbm_us) + scheduler_us
```

## 报告字段

GMM 在普通 MatMul 字段之外增加：

- `gmm_model_kind = grouped_matmul_routing_bounds`
- `gmm_expert_count`
- `gmm_weight_elements_per_expert`
- `gmm_weight_bytes_per_expert`
- `gmm_balanced_active_experts`
- `gmm_balanced_tokens_per_active_expert`
- `gmm_balanced_gm_bytes`
- `gmm_balanced_compute_us`
- `gmm_balanced_hbm_us`
- `gmm_balanced_scheduler_us`
- `gmm_balanced_total_us`
- `gmm_balanced_core_efficiency`
- `gmm_balanced_work_tiles`
- `gmm_extreme_active_experts`
- `gmm_extreme_tokens_per_active_expert`
- `gmm_extreme_gm_bytes`
- `gmm_extreme_compute_us`
- `gmm_extreme_hbm_us`
- `gmm_extreme_scheduler_us`
- `gmm_extreme_total_us`
- `gmm_extreme_core_efficiency`
- `gmm_extreme_work_tiles`
- `gmm_bounds_min_us`
- `gmm_bounds_max_us`
- `gmm_duration_position`

`gmm_duration_position`：

- `within_gmm_bounds`：实测在两个 routing 场景边界内。
- `below_gmm_bounds`：实测低于当前可解释下界，优先检查解析、dtype、shape 或 cache 假设。
- `above_gmm_bounds`：实测高于当前可解释上界，说明还有调度、同步、atomic/merge、tail split 或 exact tiling 缺失。

## 精度判断口径

GMM 不应使用普通 MatMul 的 `duration_over_estimate` 作为主精度指标。大 shape gap 分析应使用区间误差：

```text
if duration in [gmm_bounds_min_us, gmm_bounds_max_us]:
    gmm_routing_bound_error = 0
elif duration < gmm_bounds_min_us:
    gmm_routing_bound_error = (gmm_bounds_min_us - duration) / duration
else:
    gmm_routing_bound_error = (duration - gmm_bounds_max_us) / duration
```

这表示在缺少真实 group_list 时，模型只判断实测是否落在可解释 routing 区间内。

## diagnosis 与 confidence

GMM 行会带有：

- `grouped_matmul_routing_bounds`
- `within_gmm_bounds` / `below_gmm_bounds` / `above_gmm_bounds`
- quant 相关标签
- `weight_nz`
- `low_tile_count`
- compute/memory/balanced 标签

confidence 固定为 low。原因是 profiling CSV 缺少：

- 真实 `group_list` 值
- `groupListType`
- `tuningConfigOptional`
- exact tiling 输出
- tail split / adaptive sliding window 的实际选择

## 当前限制

- 没有 exact host tiling replay。
- 没有真实 group_list，因此无法确定活跃专家数和 token 分布。
- quant adaptive sliding window、weight quant、tail split、finalize routing 等路径尚未拆成独立 current-kernel 分量。
- `scheduler_us` 是源码机制级轻量项，不足以解释所有 above-bound tail。
- 如果未来 profiling 导出 group_list 或 tiling data，应优先替换 routing bounds，而不是继续加宽区间。

## 后续任务

- 解析或接入真实 `group_list`，将 bounds 收敛为单次执行估计。
- 按 arch35 no-quant / quant / weight-quant / adaptive sliding window 拆分策略标签。
- 引入 exact host tiling replay，补齐 `usedCoreNum`、singleN、tail split 等字段。
- 对 `above_gmm_bounds` 样本回查源码中 sync、atomic/merge、finalize routing 和 quant epilogue。
