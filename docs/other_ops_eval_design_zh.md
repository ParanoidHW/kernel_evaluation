# Other Ops 评估设计

本文档描述 `tools/other_ops_eval/` 当前实现的其他算子评估方案。`other_ops` 覆盖 profiling 中尚未由 MatMul、GroupedMatmul、Attention 专用评估器处理的常规 AIC/AIV kernel，重点是建立可解释的 source-strategy/fallback 估计、统一报告字段和后续扩展入口。

当前实现不是 exact host tiling replay。只有能从 profiling shape/dtype/format 和源码路径稳定识别的策略会标为 `source_strategy_replay`；缺少 permutation、axis、stride、indices、routing、mask selected count 等运行时值时，必须标记 low confidence 或 unresolved。

## 当前实现入口

CLI：

```bash
python3 tools/eval_ops.py --op-kind other_ops \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output other_ops_eval_report_910c.csv \
  --unresolved-output other_ops_eval_unresolved_910c.csv
```

代码路径：

```text
tools/op_eval/api.py
  -> evaluate_profiling(..., op_kind="other_ops")

tools/other_ops_eval/common.py
  -> Type 过滤、分类、source map、shape/dtype/format 解析、缺失 attrs 标记

tools/other_ops_eval/api.py
  -> source-strategy / fallback 成本模型

tools/other_ops_eval/evaluator.py
  -> profiling CSV 读取、resolved/unresolved CSV 输出、summary 打印
```

配置入口在 `configs/*.json::other_ops_model`：

```json
{
  "vector_bandwidth_tbps": 1.6,
  "vector_gops": 4000.0,
  "launch_overhead_us": 2.0,
  "layout_strided_factor": 1.0,
  "reduction_passes": 2.0,
  "softmax_passes": 4.0,
  "norm_passes": 3.0,
  "activation_op_factor": 4.0,
  "transcendental_op_factor": 16.0,
  "index_random_access_factor": 1.5
}
```

这些参数是平台级机制参数：AIV 吞吐、HBM 带宽、kernel launch floor、多 pass reduction/norm/softmax、transcendental vector 指令复杂度和随机访问风险。它们不是 per-shape 拟合项。

## 范围和排除规则

`other_ops` 先读取 profiling `Type`，排除：

- MatMul 家族：`MatMul*`、`BatchMatMul*`、`TransposeBatchMatMul`、`QuantBatchMatmulV3`、`GroupedMatmul`
- Attention 家族：`FlashAttentionScore`、`FusedInferAttentionScore`、`PromptFlashAttention`、`IncreFlashAttention`、`PagedAttention`、`KvQuantSparseFlashAttention` 等
- 通信或通信辅助：`Hcom*`、`allreduce`、`allgather`、`alltoall`、`reducescatter`、`broadcastAicpuKernel`、`AllGatherMatmul*`

未命中的 Type 进入 unresolved，保留原始 name/type/shape/dtype/format，便于后续新增算子族。

## 算子族分类

当前分类由 `tools/other_ops_eval/common.py` 中的 Type set 决定。

### layout_memory

覆盖：

```text
Cast, TensorMove, TransData, Transpose, Slice, StridedSliceD, AsStrided,
ConcatD, ConcatV2D, SplitVD, Pack, Tile, MemSet
```

源码映射：

```text
Cast          -> ops-math/math/cast
TensorMove    -> ops-math/conversion/tensor_move
TransData     -> ops-math/conversion/trans_data
Transpose     -> ops-math/conversion/transpose
Slice         -> ops-math/conversion/slice
StridedSliceD -> ops-math/conversion/strided_slice
AsStrided     -> ops-math/conversion/as_strided
Concat*       -> ops-math/conversion/concat
SplitVD       -> ops-math/conversion/split
Pack          -> ops-math/conversion/pack
Tile          -> ops-math/math/tile
MemSet        -> ops-math/conversion/mem_set
```

策略标签：

```text
linear_ub_cast
linear_ub_copy
format_transform_nz_nd_simt
format_transform_5hd_simt
format_transform_simt
transpose_nddma_vconv_missing_perm
slice_move_align_or_nddma_missing_offsets
as_strided_gather_or_move_align_missing_stride
concat_axis_strategy_missing_axis
split_axis_strategy_missing_axis
pack_to_concat_missing_axis
tile_broadcast_copy_missing_multiples
memset_output_fill
```

`Cast/TensorMove` 源码中主要是 UB 分块 DataCopy/Cast。`TransData` 区分 NZ/ND、5HD/SIMT 等 format transform。`Transpose/Slice/AsStrided/Concat/Pack/Tile` 的真实路径强依赖 runtime attrs，当前 profiling 不保留这些值，因此只标记策略和缺失项。

### elementwise_vector

覆盖：

```text
Add, Mul, Sub, Neg, RealDiv, Pows, GreaterEqual, Greater, Less, Equal,
ZerosLike, OnesLike, Fill, Range, Muls, Sigmoid, BroadcastTo,
SelectV2, ClipByValueV2, Cos, Sin, RotaryPositionEmbedding
```

源码映射：

```text
BroadcastTo              -> ops-math/conversion/broadcast_to
ClipByValueV2            -> ops-math/conversion/clip_by_value_v2
Fill                     -> ops-math/conversion/fill
Range                    -> ops-math/math/range
ZerosLike                -> ops-math/conversion/zeros_like
OnesLike                 -> ops-math/math/ones_like
RealDiv                  -> ops-math/math/real_div
SelectV2                 -> ops-math/math/select_v2
Cos/Sin                  -> ops-math/math/cos | ops-math/math/sin
RotaryPositionEmbedding  -> ops-transformer-master/posembedding/rotary_position_embedding
其他普通 elementwise      -> ops-math/math
```

策略标签：

```text
elementwise_vector_pipeline
elementwise_scalar_broadcast_vector_pipeline
elementwise_broadcast_vector_pipeline
elementwise_fill_vector_pipeline
elementwise_expensive_math_vector_pipeline
elementwise_transcendental_vector_pipeline
range_output_generate
rotary_pos_embedding_vector_fusion
```

op factor：

```text
fill/zeros/ones       -> 0.5
普通 Add/Mul/Sub/Neg  -> 1.0
compare/select/clip   -> 2.0
RealDiv               -> 4.0
RotaryPositionEmbedding -> 6.0
Cos/Sin               -> transcendental_op_factor
Pows/Pow              -> max(transcendental_op_factor, 8.0)
Sigmoid               -> activation_op_factor
```

这些 factor 表示源码语义中的 vector 指令复杂度或融合计算步数，不针对具体样本调参。

### reduction

覆盖：

```text
ReduceSum, ReduceSumD, ReduceMean, ReduceAll, SoftmaxV2
```

源码映射：

```text
ReduceSum/ReduceSumD -> ops-math/math/reduce_sum
ReduceMean           -> ops-math/math/reduce_mean
ReduceAll            -> ops-math/math/reduce_all
SoftmaxV2            -> ops-nn/activation/softmax_v2
```

策略标签：

```text
reduce_tree
reduce_tree_with_scale
softmax_reduce_exp_sum_normalize
```

pass 规则：

```text
ReduceSum/ReduceAll -> reduction_passes
ReduceMean          -> reduction_passes + 1
SoftmaxV2           -> softmax_passes
```

`ReduceMean` 比普通 reduce 多 scale 语义；`SoftmaxV2` 用 max/exp/sum/normalize 的多 pass 语义建模。

### norm_activation

覆盖：

```text
RmsNorm, LayerNormV3, Add_RmsNorm, AddRmsNorm, InplaceAddRmsNorm,
AddRmsNormCast, Swish, Gelu, SwiGlu, GeGluV2, DequantSwigluQuant,
GroupNormSilu
```

源码映射：

```text
RmsNorm                 -> ops-nn/norm/rms_norm
LayerNormV3             -> ops-nn/norm/layer_norm_v3
AddRmsNorm              -> ops-nn/norm/add_rms_norm
InplaceAddRmsNorm       -> ops-nn/norm/inplace_add_rms_norm
AddRmsNormCast          -> ops-nn/norm/add_rms_norm_cast
Swish                   -> ops-nn/activation/swish
Gelu                    -> ops-nn/activation/gelu
SwiGlu/DequantSwigluQuant -> ops-nn/activation/swiglu
GeGluV2                 -> ops-nn/activation/geglu_v2
GroupNormSilu           -> ops-nn/norm/group_norm_silu
```

策略标签：

```text
activation_vector_pipeline
rmsnorm_reduce_scale
rmsnorm_residual_fusion
layernorm_mean_var_scale
groupnorm_reduce_silu
```

pass 规则：

```text
Swish/Gelu                         -> 1.0
SwiGlu/GeGluV2/DequantSwigluQuant  -> 1.5
RmsNorm                            -> norm_passes
AddRmsNorm/Inplace/AddRmsNormCast  -> max(norm_passes, 4.0)
LayerNormV3                        -> max(norm_passes, 4.0)
GroupNormSilu                      -> max(norm_passes, 4.0)
```

RMS/LayerNorm 类按 reduce + normalize/scale + optional residual/fusion 建模；activation-only 路径按 input/output HBM 和 vector op factor 建模。

### index_scatter_routing

覆盖：

```text
GatherV2, GatherV3, GatherElements, Scatter, ScatterUpdate,
ScatterNdUpdate, ScatterElementsV2, MaskedSelectV3, LinearIndex,
NonZero, TopKV2, ArgMaxV2, Index, Moe*
```

源码映射：

```text
GatherV2/GatherV3/GatherElements -> ops-nn/index/gather_*
Scatter/ScatterNd/ScatterElements -> ops-nn/index/scatter*
MaskedSelectV3                   -> ops-math/conversion/masked_select_v3
LinearIndex                      -> ops-nn/index/linear_index
NonZero                          -> ops-nn/index/non_zero
TopKV2                           -> ops-nn/index/apply_top_k_top_p_with_sorted
MoE routing                      -> ops-transformer-master/moe 或 ops-transformer-master/mc2
```

策略标签：

```text
gather_random_read_missing_indices
scatter_random_write_missing_indices
mask_compaction_missing_selected_count
linear_index_missing_indices
topk_sort_select_missing_k_distribution
moe_routing_missing_token_distribution
```

这类算子强依赖 indices、mask selected count、TopK 分布、routing/token 分布和 atomic/random write 行为。当前 profiling 不包含这些运行时值，因此模型保持 `analytic_fallback_missing_runtime_values` 和 low confidence，不通过随机访问因子拟合 tail。

### cv_regular

覆盖：

```text
Conv2D, Conv3DV2, ResizeBicubicV2, ResizeBilinearV2,
ResizeNearestNeighborV2, GridSample/GridSample2D/GridSample3D,
RoiAlign, NmsWithMask, NonMaxSuppressionV3/V6
```

当前 `cv_regular` 是低置信 fallback。`Conv2D` 已能分类，但还没有按 `ops-nn/conv/conv2d_v2` 的 host tiling、Cube FLOPs、NC1HWC0/FRACTAL_Z storage、L0/L1/BT、bias/scale/fixpipe 路径建模。Resize/GridSample/ROI/NMS 也需要后续按 `ops-cv` 各目录单独设计。

## profiling 解析规则

输入字段：

- `Type` / `Name`
- `Input Shapes`、`Input Data Types`、`Input Formats`
- `Output Shapes`、`Output Data Types`、`Output Formats`
- `Duration(us)`、`Block Dim`、`Mix Block Dim`
- `aicore_time(us)`、`aiv_time(us)` 等 profiling counter

解析结果：

```text
input_elements[] = product(input_shapes[])
output_elements[] = product(output_shapes[])
input_bytes[] = input_elements[] * dtype_size(input_dtype)
output_bytes[] = output_elements[] * dtype_size(output_dtype)
logical_elements = max(output_elements) or max(input_elements)
```

当 dtype 数量多于 shape 数量时，当前实现把缺失 shape 的输入当作 scalar，补 `input_elements=1`。这用于处理 scalar alpha、value、axis 等小输入。

缺少 shape/dtype/format 且没有 output 规模的行不能构造 spec，例如当前 910C `MemSet` 的 `N/A` 行，进入 unresolved。

## 缺失 attrs 和置信度

当前显式标记：

```text
Transpose            -> perm
Slice/StridedSliceD  -> begin|size_or_end|stride
Concat/Split/Pack    -> axis
Tile                 -> multiples
MemSet               -> fill_value
AsStrided            -> size|stride|storage_offset
MoE                  -> routing_values
TopK/MaskedSelect/NonZero -> selected_count_or_mask_values
LinearIndex          -> index_values
Gather/Scatter       -> indices_or_scatter_values
```

`tools/other_ops_eval/evaluator.py` 的诊断规则：

- 有 `missing_attrs`：`confidence=low`
- `tiling_source` 包含 `missing`：`confidence=low`
- `index_scatter_routing` / `cv_regular`：默认 `confidence=low`
- `duration_us / estimated_us > 5`：追加 `large_residual`

## 成本模型

共享变量：

```text
input_bytes = sum(input_bytes[])
output_bytes = sum(output_bytes[])
traffic_bytes = input_bytes + output_bytes
vector_ops = logical_elements * op_factor_or_passes
hbm_us = traffic_bytes / hbm_bandwidth
vector_compute_us = vector_ops / vector_gops
body_us = max(hbm_us, vector_compute_us) + layout_overhead_us + workspace_us + sync_overhead_us
estimated_us = body_us + launch_overhead_us
ideal_lower_bound_us = max((input_bytes + output_bytes) / hbm_bandwidth, vector_compute_us)
current_kernel_bound_us = body_us
```

### layout/memory

```text
traffic_bytes = input_bytes + output_bytes
vector_ops = 0
tiling_source = source_strategy_replay 或 source_strategy_replay_missing_attrs
layout_overhead_us = traffic_bytes * (layout_strided_factor - 1) / hbm_bandwidth
```

当前 `layout_strided_factor=1.0`，因此缺 attrs 不额外加 layout overhead。这样避免为了拟合 Slice/Pack tail 引入无依据旋钮。

### elementwise/vector

```text
traffic_bytes = input_bytes + output_bytes
vector_ops = logical_elements * op_factor
tiling_source = source_strategy_replay
estimated_us = max(hbm_us, vector_compute_us) + launch_overhead_us
```

Broadcast 输入按实际 storage 读流量，按输出 `logical_elements` 计 vector 工作量。

### reduction

```text
passes = reduction_passes / reduction_passes+1 / softmax_passes
traffic_bytes = input_bytes + output_bytes * passes
workspace_us = output_bytes / hbm_bandwidth
vector_ops = sum(input_elements) * passes
```

### norm/activation

Activation-only：

```text
traffic_bytes = input_bytes + output_bytes
vector_ops = logical_elements * activation_op_factor * activation_passes
```

Norm/fusion：

```text
traffic_bytes = (input_bytes + output_bytes) * norm_or_fusion_passes
vector_ops = logical_elements * activation_op_factor
```

### index/scatter/routing

```text
traffic_bytes = (input_bytes + output_bytes) * index_random_access_factor
vector_ops = logical_elements
tiling_source = analytic_fallback_missing_runtime_values
```

这里的 `index_random_access_factor` 只表达随机访问/atomic 风险的保守 fallback，不代表真实 indices 分布。

### cv_regular

```text
traffic_bytes = input_bytes + output_bytes
vector_ops = logical_elements * activation_op_factor
tiling_source = analytic_fallback_source_pending
```

`Conv2D` 等 Cube-heavy kernel 当前只是分类占位，后续需要独立成本模型。

## 报告字段

resolved CSV 包含：

```text
file,line,name,type,accelerator_core
op_family,source_repo,source_path,source_strategy,layout_pattern,tiling_source,missing_attrs
input_shapes,output_shapes,input_dtypes,output_dtypes,input_formats,output_formats
input_elements,output_elements,input_bytes,output_bytes,logical_elements
block_dim,mix_block_dim,aicore_time_us,aiv_time_us
vector_compute_us,hbm_us,layout_overhead_us,workspace_us,sync_overhead_us,launch_overhead_us
current_kernel_bound_us,ideal_lower_bound_us,estimated_us,total_us
duration_us,residual_us,duration_over_estimate
bottleneck,diagnosis,confidence
```

unresolved CSV 包含：

```text
file,line,type,name,reason,input_shapes,output_shapes,input_dtypes,output_dtypes,input_formats,output_formats
```

## 当前基线

最新 other_ops 快照：

- `eval_results/20260525T110812Z_9969381/other_ops_eval_summary.csv`
- commit：`9969381`

关键结果：

| 报告 | rows | unresolved | rel max | p95 | median | 主要 tail |
|---|---:|---:|---:|---:|---:|---|
| `other_ops_eval_report_910b4` | 9285 | 923 | 639.020 | 2.234 | 0.533 | `GatherV2` 缺 indices |
| `other_ops_eval_report_910c` | 72364 | 34 | 5.710 | 0.554 | 0.299 | `Pack/Slice/Gather` 缺 attrs/indices |
| `profiling_with_model_code_ds32_other_ops_eval_910c` | 2980 | 870 | 16.635 | 1.476 | 0.614 | `GatherV2` 缺 indices |
| `profiling_with_model_code_gemma_other_ops_eval_910b4` | 2820 | 555 | 47.205 | 3.729 | 0.521 | `GatherV2/Scatter` 缺 indices |
| `profiling_with_model_code_longcat_other_ops_eval_910b4` | 640 | 101 | 666.847 | 0.961 | 0.465 | `GatherV2` 缺 indices |
| `profiling_with_model_code_qwen7b_other_ops_eval_910b4_1` | 3882 | 195 | 148.856 | 0.857 | 0.329 | `GatherV2` 缺 indices |

这些最大误差主要由 low-confidence index/scatter fallback 触发。因为 profiling 没有 indices 实际访问范围，不能通过调小随机访问流量来拟合。

## 已知限制和残留

- `MemSet`：部分 profiling 行 shape/dtype/format 全为 `N/A`，无法估计 output bytes。需要 profiling 补规格，或从相邻 `TransData/Conv` 上下文推导 buffer 大小。
- `Gather/Scatter/TopK/MoE`：缺 indices、selected count、routing/token 分布，当前只能 low-confidence fallback。
- `Pack/Slice/Transpose/AsStrided/Concat/Tile`：缺 axis、offset、stride、multiples 等 attrs，不能 exact replay。
- `Conv2D`：已分类为 `cv_regular`，但当前没有 Cube tiling/BT/fixpipe 模型。
- 模型样本 unresolved：`RotaryMul`、`DynamicQuant`、`Rsqrt`、`MlaPrologV3`、`LightningIndexerQuant`、`InterleaveRope`、`KvRmsNormRopeCache`、`MoeComputeExpertTokens` 等需要下一轮 transformer/vector fusion 设计。`AutomaticBufferFusionOp` 是非固定 pattern 的融合包装算子，无法仅从 `Type/shape` 直接评估，后续按忽略项处理。
- 910C longcat stage2 已补充 `FloorDiv`、`FloorMod`、`ReduceMax`、`GatherElementsV2`、`Maximum`、`Cumsum`、`Tril`、`LogicalNot`、`Unpack`。其中 `GatherElementsV2/Cumsum/Tril/Unpack` 仍因缺 indices、axis 或 diagonal 等 attrs 标为低置信。

## 后续任务

1. 为 transformer/vector fusion 建独立设计：`RotaryMul`、`InterleaveRope`、`KvRmsNormRopeCache`、`MlaPrologV3`、`DynamicQuant`、`Rsqrt`。
2. 为 `Conv2D` 建立 `ops-nn/conv/conv2d_v2` 评估模型，按 Cube FLOPs、NC1HWC0/FRACTAL_Z storage、tiling、L0/L1/BT、bias/scale/fixpipe 拆分。
3. 为 index/scatter 增加 bounds 语义：在缺 indices 时输出 min/max，而不是单点 overestimate。
4. 若 profiling 能补 attrs，恢复 `Transpose/Slice/Pack/Tile/AsStrided/Concat` 的更精确 source replay。
5. 把 `tools/annotate_profiling.py` 扩展为调用 `other_ops`，让非 MatMul/GMM/Attention 的已支持 Type 也能写入 `kernel_eval_value`。
