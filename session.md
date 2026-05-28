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

## 2026-05-25 其他算子评估方案计划

本轮确认新增源码目录：

- `ops-math`
- `ops-cv`

已将其他算子评估方案写入 `docs/architecture.md` 的“其他算子评估方案”章节。

首轮处理顺序：

1. layout/memory 类：`Cast`、`Transpose`、`TransData`、`TensorMove`、`Slice`、`StridedSliceD`、`AsStrided`、`ConcatD/ConcatV2D`、`SplitVD`。
2. elementwise/vector 类：`Add`、`Mul`、`Sub`、`Neg`、`RealDiv`、`Pows`、比较和填充类。
3. reduction/norm/activation 类：`ReduceSum`、`ReduceMean`、`RmsNorm`、`LayerNormV3`、`AddRmsNorm`、`InplaceAddRmsNorm`、`Swish`、`Gelu`、`SwiGlu`。
4. index/scatter/routing 类：`GatherV2/V3`、`ScatterUpdate`、`ScatterNdUpdate`、`TopKV2`、`Moe*`。
5. CV/常规大 kernel：`Conv3DV2`、`GroupNormSilu`、`Resize*`、`GridSample*`、ROI/NMS 类。

原则：

- 通信类 `Hcom*`、`hcom_*`、AICPU 通信辅助和 `AllGatherMatmul*` 首轮排除。
- 缺少运行时 attrs 的复杂 layout/index/routing 算子只能做 fallback、bounds 或 unresolved，不能声称 exact replay。
- 成本模型先建立 vector/HBM/layout/reduction/workspace/launch 分量，不做 per-shape 拟合。

## 2026-05-25 other_ops 基础框架

本轮完成 `other_ops` 第一阶段基础框架：

- 新增 `tools/other_ops_eval/`：
  - `common.py`：Type 分类、shape/dtype/format 解析、source map、缺失 attrs 标记。
  - `api.py`：layout/memory、elementwise/vector、reduction、norm/activation、index/scatter/routing、CV 的首轮 analytic fallback 成本模型。
  - `evaluator.py`：profiling CSV 读取、resolved/unresolved 报告和 summary。
- `tools/op_eval/api.py` / `tools/op_eval/cli.py` 注册 `op_kind=other_ops`。
- `configs/ascend_910b4*.json`、`configs/ascend_910c.json` 新增 `other_ops_model` 平台级参数。
- `docs/architecture.md` 更新当前实现状态、验证命令和后续任务。

验证：

- `python3 -m compileall tools`
- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C --config configs/ascend_910c.json --output /tmp/other_ops_910c_stage1.csv --unresolved-output /tmp/other_ops_910c_stage1_unresolved.csv`

910C 结果：

- resolved：`72016`
- unresolved：`382`
- family 概况：
  - elementwise_vector：`67264`
  - layout_memory：`3012`
  - norm_activation：`828`
  - index_scatter_routing：`656`
  - reduction：`256`
- 主要 unresolved：`RotaryPositionEmbedding`、`MemSet`、`Conv2D`、`Tile`、`Cos`、`Sin`。

当前限制：

- 模型仍是 analytic fallback / source-strategy 级别，不是 exact tiling replay。
- `Transpose/Slice/Concat/Split/AsStrided/Gather/Scatter/MoE` 等缺 runtime attrs 或 indices/routing 值的行保留 `missing_runtime_attrs` 和 low confidence。
- 后续按优先级继续补 layout/memory 源码策略。

## 2026-05-25 other_ops layout/memory 源码策略分类

本轮完成其他算子第一优先级中的 layout/memory 源码策略增强：

- `tools/other_ops_eval/common.py` 将 layout/memory source map 从族级目录细化到具体源码目录：
  - `Cast -> ops-math/math/cast`
  - `TensorMove -> ops-math/conversion/tensor_move`
  - `TransData -> ops-math/conversion/trans_data`
  - `Transpose -> ops-math/conversion/transpose`
  - `Slice/StridedSliceD -> ops-math/conversion/slice|strided_slice`
  - `AsStrided -> ops-math/conversion/as_strided`
  - `ConcatD/ConcatV2D/SplitVD/Pack -> ops-math/conversion/concat|split|pack`
- resolved 报告新增 `source_strategy` 和 `layout_pattern`：
  - 线性路径：`linear_ub_cast`、`linear_ub_copy`
  - 格式转换：`format_transform_*_simt`
  - 运行时 attrs 缺失路径：`transpose_nddma_vconv_missing_perm`、`slice_move_align_or_nddma_missing_offsets`、`as_strided_gather_or_move_align_missing_stride`、`concat_axis_strategy_missing_axis`、`pack_to_concat_missing_axis`
- `tools/other_ops_eval/api.py` 对 layout/memory 使用 `source_strategy_replay` / `source_strategy_replay_missing_attrs` 语义，不把缺 attrs 的路径伪装成 exact tiling。
- `docs/architecture.md` 同步更新当前实现状态和报告字段。

验证：

- `python3 -m compileall tools`
- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C --config configs/ascend_910c.json --output /tmp/other_ops_910c_layout_strategy.csv --unresolved-output /tmp/other_ops_910c_layout_strategy_unresolved.csv`

910C 结果：

- resolved：`72016`
- unresolved：`382`
- family 中位 `duration_over_estimate`：
  - elementwise_vector：`0.77`
  - layout_memory：`2.48`
  - norm_activation：`1.10`
  - index_scatter_routing：`10.89`
  - reduction：`5.53`
- top tail 仍集中在 `Pack/Slice/GatherV2/GatherV3`，原因分别是缺 `axis`、`begin/size/stride` 或 indices 实际值；当前保持 low confidence，不做拟合参数。

## 2026-05-25 other_ops elementwise/vector 分类增强

本轮继续处理第二优先级 elementwise/vector：

- `tools/other_ops_eval/common.py` 新增或细化：
  - `Cos/Sin/Equal/Greater` 纳入 elementwise/vector。
  - `Tile/MemSet` 纳入 layout/memory，其中 `Tile` 按 broadcast-copy，`MemSet` 按 output fill；profiling 中 `MemSet` 为 `N/A` shape 时仍 unresolved。
  - elementwise source map 补到 `ops-math/math/cos`、`sin`、`real_div`、`select_v2`、`ops-math/conversion/fill`、`zeros_like`、`clip_by_value_v2` 等具体目录。
  - `source_strategy` 区分普通 vector、scalar broadcast、broadcast、fill、expensive math、transcendental vector pipeline。
- `tools/other_ops_eval/api.py` 将 elementwise 从纯 analytic fallback 升为 source-strategy 级别，并按算子语义计 vector op factor：
  - 普通 Add/Mul/Sub/Neg：`1`
  - compare/select/clip：`2`
  - RealDiv：`4`
  - Cos/Sin：`transcendental_op_factor`
  - Pows/Pow：不低于 transcendental factor
  - fill/zeros/ones：`0.5`
- `configs/ascend_910b4*.json`、`configs/ascend_910c.json` 新增平台级 `transcendental_op_factor=16.0`，对应 transcendental vector 指令/近似多步计算，不针对单个 shape 拟合。
- `docs/architecture.md` 同步更新当前实现状态。

验证：

- `python3 -m compileall tools`
- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C --config configs/ascend_910c.json --output /tmp/other_ops_910c_elementwise.csv --unresolved-output /tmp/other_ops_910c_elementwise_unresolved.csv`

910C 结果：

- resolved：`72054`，较上一轮 `72016` 增加 `38`
- unresolved：`344`，较上一轮 `382` 减少 `38`
- `Cos/Sin/Tile/Equal` 已被分类；`MemSet` 因 profiling shape 为 `N/A` 仍 unresolved。
- 主要 unresolved 变为 `RotaryPositionEmbedding`、`MemSet`、`Conv2D`、`MaskedSelectV3`、`Range`、`LinearIndex`、`ScatterElementsV2`、`NonZero`、`GatherElements`。
- family 中位 `duration_over_estimate` 基本保持：
  - elementwise_vector：`0.77`
  - layout_memory：`2.48`
  - norm_activation：`1.10`
  - index_scatter_routing：`10.89`
  - reduction：`5.53`

## 2026-05-25 other_ops reduction/norm/activation 策略增强

本轮处理第三优先级 reduction/norm/activation：

- `tools/other_ops_eval/common.py` 新增源码映射：
  - `ReduceSum/ReduceSumD -> ops-math/math/reduce_sum`
  - `ReduceMean -> ops-math/math/reduce_mean`
  - `ReduceAll -> ops-math/math/reduce_all`
  - `SoftmaxV2 -> ops-nn/activation/softmax_v2`
  - `RmsNorm/LayerNormV3/AddRmsNorm/InplaceAddRmsNorm/AddRmsNormCast -> ops-nn/norm/*`
  - `Swish/Gelu -> ops-nn/activation/*`
  - `GroupNormSilu -> ops-nn/norm/group_norm_silu`
- `source_strategy` 新增：
  - `reduce_tree`
  - `reduce_tree_with_scale`
  - `softmax_reduce_exp_sum_normalize`
  - `rmsnorm_reduce_scale`
  - `rmsnorm_residual_fusion`
  - `layernorm_mean_var_scale`
  - `activation_vector_pipeline`
  - `groupnorm_reduce_silu`
- `tools/other_ops_eval/api.py` 将 reduction/norm 从 analytic fallback 升为 source-strategy 级别：
  - `ReduceMean` 在 reduce pass 后额外计 scale pass。
  - `SoftmaxV2` 使用 `softmax_passes` 表达 max/exp/sum/normalize 等源码语义。
  - RMS/LayerNorm/AddRmsNorm/GroupNormSilu 按 reduce + normalize/fusion pass 计入 traffic。
- `configs/ascend_910b4*.json`、`configs/ascend_910c.json` 新增 `softmax_passes=4.0`。
- `docs/architecture.md` 同步更新当前实现状态。

验证：

- `python3 -m compileall tools`
- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C --config configs/ascend_910c.json --output /tmp/other_ops_910c_norm_reduce.csv --unresolved-output /tmp/other_ops_910c_norm_reduce_unresolved.csv`

910C 结果：

- resolved：`72054`
- unresolved：`344`
- reduction 中位 `duration_over_estimate`：`5.53 -> 4.92`
- norm_activation 中位 `duration_over_estimate`：`1.10`
- top tail 仍不在 reduction/norm，而是缺 runtime attrs 的 `Pack/Slice/Gather`。

## 2026-05-25 other_ops index/scatter/routing 分类增强

本轮处理第四优先级 index/scatter/routing：

- `tools/other_ops_eval/common.py` 新增分类：
  - `GatherElements`
  - `ScatterElementsV2`
  - `MaskedSelectV3`
  - `LinearIndex`
  - `NonZero`
- source map 补充到：
  - `ops-nn/index/gather_v2|gather_v3|gather_elements`
  - `ops-nn/index/scatter|scatter_nd|scatter_elements_v2`
  - `ops-math/conversion/masked_select_v3`
  - `ops-nn/index/linear_index|non_zero`
  - `ops-transformer-master/moe/*` / `mc2/*` MoE routing 目录
- `missing_attrs` 从统一 `indices_or_routing_values` 细化为：
  - `indices_or_scatter_values`
  - `selected_count_or_mask_values`
  - `index_values`
  - `routing_values`
- `source_strategy` 细化为：
  - `gather_random_read_missing_indices`
  - `scatter_random_write_missing_indices`
  - `mask_compaction_missing_selected_count`
  - `linear_index_missing_indices`
  - `topk_sort_select_missing_k_distribution`
  - `moe_routing_missing_token_distribution`
- 成本模型仍保持 low confidence 的 random-access fallback，不引入无 indices/routing 支撑的校准。
- `docs/architecture.md` 同步更新当前实现状态。

验证：

- `python3 -m compileall tools`
- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C --config configs/ascend_910c.json --output /tmp/other_ops_910c_index.csv --unresolved-output /tmp/other_ops_910c_index_unresolved.csv`

910C 结果：

- resolved：`72072`，较上一轮 `72054` 增加 `18`
- unresolved：`326`，较上一轮 `344` 减少 `18`
- 主要 unresolved 缩减为 `RotaryPositionEmbedding`、`MemSet`、`Conv2D`、`Range`。
- index_scatter_routing 中位 `duration_over_estimate`：`10.83`，仍为低置信；主因是 profiling 缺 indices、mask selected count、routing/token 分布。

## 2026-05-25 other_ops unresolved tail 分类

本轮处理第五优先级 unresolved tail：

- `RotaryPositionEmbedding` 纳入 elementwise/vector：
  - source path：`ops-transformer-master/posembedding/rotary_position_embedding`
  - source strategy：`rotary_pos_embedding_vector_fusion`
  - 成本按输入 Q + cos/sin + 输出 HBM，vector op factor 表达 rotate/mul/add 融合逻辑。
- `Range` 纳入 elementwise/vector：
  - source path：`ops-math/math/range`
  - source strategy：`range_output_generate`
- `Conv2D` 纳入 `cv_regular`：
  - 当前作为常规大 kernel/Cube-heavy CV 类 fallback，source 族对齐 `ops-nn/conv/conv2d_v2` 及 `ops-cv` 后续设计口径。
  - 本轮只完成分类和低置信 fallback，不做 Conv exact tiling。
- `MemSet` 已有 source path `ops-math/conversion/mem_set`，但当前 910C profiling 中 34 行 `Input/Output Shapes`、dtype、format 全为 `N/A`，无法推导 output bytes；保持 unresolved，作为需要 profiling 补 shape 的遗留。
- `docs/architecture.md` 同步更新当前实现状态。

验证：

- `python3 -m compileall tools`
- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C --config configs/ascend_910c.json --output /tmp/other_ops_910c_unresolved_tail.csv --unresolved-output /tmp/other_ops_910c_unresolved_tail_unresolved.csv`

910C 结果：

- resolved：`72364`，较上一轮 `72072` 增加 `292`
- unresolved：`34`，较上一轮 `326` 减少 `292`
- 唯一 unresolved type：`MemSet`，共 `34` 行，原因是 profiling 规格全为 `N/A`。
- 新增 `cv_regular`：`32` 行 `Conv2D`，中位 `duration_over_estimate=10.21`，当前低置信，后续需要按 `ops-nn/conv2d_v2` host tiling/Cube 逻辑单独设计。

## 2026-05-25 other_ops 当前基线汇总和新问题

本轮完成优先级内的其他算子分类和首轮建模后，刷新了 910C 子集与全量探索基线。

910C 子集命令：

- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C --config configs/ascend_910c.json --output /tmp/other_ops_910c_unresolved_tail.csv --unresolved-output /tmp/other_ops_910c_unresolved_tail_unresolved.csv`
- `python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py /tmp/other_ops_910c_unresolved_tail.csv`

910C 子集结果：

- resolved：`72364`
- unresolved：`34`
- unresolved 仅剩 `MemSet`，且 profiling shape/dtype/format 全为 `N/A`。
- relative error：
  - max：`5.7103`
  - p95：`0.5539`
  - p90：`0.3889`
  - median：`0.2987`
- `duration_over_estimate` median：`0.7700`
- family 中位 `duration_over_estimate`：
  - elementwise_vector：`0.77`
  - layout_memory：`2.48`
  - norm_activation：`1.10`
  - index_scatter_routing：`10.83`
  - reduction：`4.92`
  - cv_regular：`10.21`

910C top tail：

- `Pack`：缺 axis，当前按 output storage 做 conservative HBM 估计，出现 overestimate。
- `GatherV2/GatherV3`：缺 indices 实际访问范围，按随机访问 fallback 导致 overestimate。
- `Slice`：缺 begin/size/stride，无法判断是否只是连续小片段，当前保守估计。

全量探索命令：

- `python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings --config configs/ascend_910c.json --output /tmp/other_ops_all_910c_config.csv --unresolved-output /tmp/other_ops_all_910c_config_unresolved.csv`
- `python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py /tmp/other_ops_all_910c_config.csv`

全量探索结果：

- resolved：`91971`
- unresolved：`2678`
- relative error：
  - max：`333.4668`
  - p95：`0.7867`
  - p90：`0.6534`
  - median：`0.3158`
- 该全量结果混合 910B4、910C、longcat 等不同平台，却统一使用 `ascend_910c.json`，只能作为 Type 覆盖和 tail 发现，不作为严格精度基线。
- 全量主要 unresolved：`AutomaticBufferFusionOp`、`RotaryMul`、`DynamicQuant`、`Rsqrt`、`MoeComputeExpertTokens`、`MemSet`、`MlaPrologV3`、`LightningIndexerQuant`、`PadV3`、`Sort`、`InterleaveRope`、`KvRmsNormRopeCache`。其中 `AutomaticBufferFusionOp` 后续确认应作为非固定 pattern 融合包装忽略项。

新问题和建议措施：

- `MemSet N/A`：profiling 缺 shape/dtype/format，当前无法估计 output bytes。需要 profiling 导出补齐规格，或从相邻 `TransData/Conv` fusion 上下文解析 memset buffer 大小。
- `Pack/Slice/Gather` overestimate：都属于缺 axis/offset/indices 的低置信路径。后续需要 profiling attrs、host tiling data 或运行时输入摘要；不应通过降低 HBM 带宽或随机访问因子拟合。
- `Conv2D`：已分类为 `cv_regular`，但当前只是 fallback。后续应按 `ops-nn/conv/conv2d_v2` tiling、Cube FLOPs、NC1HWC0/FRACTAL_Z storage、L0/L1/BT 和 bias/scale/fixpipe 路径单独建模。
- 全量 unresolved 的 `DynamicQuant/Rsqrt/RotaryMul/InterleaveRope/KvRmsNormRopeCache/MlaPrologV3` 需要新一轮按 transformer/vector fusion 类设计，不应混入本轮 basic other_ops fallback。`AutomaticBufferFusionOp` 不再列为待建模项。

## 2026-05-25 eval_results other_ops 快照刷新

本轮按当前 commit `9969381` 刷新 `eval_results`，新增快照：

- `eval_results/20260525T110812Z_9969381`
- `eval_results/LATEST` 已指向该目录。

产物：

- `eval_summary.csv`：沿用历史大 shape / Cube occupied 过滤口径，便于和旧 MatMul/Attention/GMM summary 共存。
- `other_ops_eval_summary.csv`：新增 other_ops 专用汇总，按所有 resolved other_ops 行统计，不使用 Cube 过滤。
- `metadata.txt`：记录 commit、报告数量、配置和 summary 口径。
- 详细 resolved/unresolved CSV 已在本地同名目录生成，但按 `eval_results/.gitignore` 约定不提交。

覆盖报告：

- `other_ops_eval_report_910b4.csv`：`example_profilings/910B4` + `configs/ascend_910b4.json`
- `other_ops_eval_report_910c.csv`：`example_profilings/910C` + `configs/ascend_910c.json`
- `profiling_with_model_code_ds32_other_ops_eval_910c.csv`
- `profiling_with_model_code_gemma_other_ops_eval_910b4.csv`
- `profiling_with_model_code_longcat_other_ops_eval_910b4.csv`
- `profiling_with_model_code_qwen7b_other_ops_eval_910b4_1.csv`

other_ops 专用 summary 结果：

- 910B4：rows `9285`，unresolved `923`，rel max `639.020`，p95 `2.234`，median `0.533`。top tail 为缺 indices 的 `GatherV2`。
- 910C：rows `72364`，unresolved `34`，rel max `5.710`，p95 `0.554`，median `0.299`。top tail 为缺 axis 的 `Pack`，unresolved 仅 `MemSet`。
- ds3.2：rows `2980`，unresolved `870`，rel max `16.635`，p95 `1.476`，median `0.614`。主要 unresolved 为 `DynamicQuant/RotaryMul/MlaPrologV3/LightningIndexerQuant`。
- gemma：rows `2820`，unresolved `555`，rel max `47.205`，p95 `3.729`，median `0.521`。主要 unresolved 为 `AutomaticBufferFusionOp/RotaryMul/MoeComputeExpertTokens`，其中 `AutomaticBufferFusionOp` 后续按忽略项处理。
- longcat：rows `640`，unresolved `101`，rel max `666.847`，p95 `0.961`，median `0.465`。主要 unresolved 为 `InterleaveRope/KvRmsNormRopeCache`。
- qwen7b 910B4-1：rows `3882`，unresolved `195`，rel max `148.856`，p95 `0.857`，median `0.329`。主要 unresolved 为 `Rsqrt`。

结论：

- 基础 910C other_ops 覆盖已经较完整；剩余 `MemSet` 需要 profiling 补规格或上下文解析。
- 910B4/模型样本最大误差主要来自 `GatherV2/GatherV3` 缺 indices，当前为低置信 fallback，不能用随机访问因子拟合。
- 下一轮优先处理 transformer/vector fusion 类 unresolved：`RotaryMul`、`DynamicQuant`、`Rsqrt`、`MlaPrologV3`、`InterleaveRope`、`KvRmsNormRopeCache`、`MoeComputeExpertTokens`。`AutomaticBufferFusionOp` 不进入建模 TODO。

## 2026-05-25 other_ops 设计文档补齐

用户指出其他算子的评估方案没有形成独立 docs 文档。本轮补齐：

- 新增 `docs/other_ops_eval_design_zh.md`，按当前实现详细记录：
  - `other_ops` CLI/API 入口和配置项。
  - layout/memory、elementwise/vector、reduction、norm/activation、index/scatter/routing、cv_regular 的 Type 覆盖。
  - `ops-math`、`ops-nn`、`ops-transformer-master`、`ops-cv` source map。
  - `source_strategy`、`layout_pattern`、`missing_attrs` 语义。
  - AIV/HBM/pass/launch/source-strategy fallback 成本模型。
  - resolved/unresolved 报告字段。
  - `eval_results/20260525T110812Z_9969381` other_ops 当前基线。
  - `MemSet N/A`、`Gather/Scatter` 缺 indices、`Conv2D` 低置信 fallback、transformer/vector fusion unresolved 等限制。
- `docs/architecture.md` 的算子族入口新增 Other Ops 设计文档链接，并将旧 TODO 改成当前已完成/剩余任务状态。
- `README.md` 新增 Other Ops 设计文档入口和当前能力摘要。

## 2026-05-25 架构术语与信息来源补充

本轮按用户要求补充 `docs/architecture.md`：

- 新增“信息来源”章节，明确 profiling CSV、CANN 开源仓库、平台配置和基线统计的优先级。
- 补充 CANN 开源仓库链接：`ops-nn`、`ops-transformer`、`ops-math`、`ops-cv`，并记录本地对应目录和使用范围。
- 新增“术语说明”章节，解释 `replay`、`runtime_kb_exact`、`source_tiling_replay`、`source_strategy_replay`、`ops_transformer_source_strategy_replay`、`analytic_fallback`、`physical_lower_bound`、`current_kernel_bound_us`、`estimated_us`、`missing_attrs`、`unresolved`、`routing bounds` 等概念。
- 重写 `actual / fallback / optimal` 语义入口，明确 `actual_tiling_source` 不等于都具备 exact 二进制 replay，`fallback_tiling_source` 不能伪装成 actual tiling，`optimal_tiling_source` 只表示下界参考。

## 2026-05-26 架构术语表格重组

本轮继续整理 `docs/architecture.md` 的“术语说明”：

- 将原先单一术语大表拆为四组：`核心概念`、`来源与 Replay`、`结果与误差字段`、`诊断与限制`。
- 保留 `xxx_replay` 后缀约定为独立表格，避免和正文术语混排。
- 术语语义保持不变，仅优化阅读结构，便于按问题类型快速查找。

## 2026-05-27 910C longcat 验证集

用户新增 profiling：

- `example_profilings/910C/longcat_kernel_details.csv`

本轮只作为验证集检查当前建模精度，不刷新正式 `eval_results/LATEST`。使用配置：

- `configs/ascend_910c.json`

生成报告：

- `/tmp/longcat_910c_matmul.csv`
- `/tmp/longcat_910c_gmm.csv`
- `/tmp/longcat_910c_attention.csv`
- `/tmp/longcat_910c_other_ops.csv`

已新增轻量验证快照：

- `eval_results/20260527T075806Z_e632e87_longcat910c_validation/eval_summary.csv`
- `eval_results/20260527T075806Z_e632e87_longcat910c_validation/unsupported_types.csv`
- `eval_results/20260527T075806Z_e632e87_longcat910c_validation/metadata.txt`

验证结果：

- MatMul：resolved `695`，unresolved `0`，relative error max `3.797`，p95 `3.547`，median `0.749`。主要 tail 是 `QuantBatchMatmulV3` Weight-NZ/small-M 高估；`TransposeBatchMatMul` 仍属于已知 transpose/strided 残留。
- GMM：resolved `140`，unresolved `0`，普通单点 relative error max `14.072`，p95 `5.481`，median `2.250`。按现有单点均值明显高估，且有 below-bound/min-bound 问题；仍应优先按 routing bounds 和真实 group/tuning 信息解释。
- Attention：resolved `140`，unresolved `140`。resolved 行均为 `FusedInferAttentionScore B=161,Sq=8,Sk=128,D=512`，source-strategy 模型显著高估，relative error max `14.075`，p95 `13.686`，median `13.237`。另有 140 行 `FusedInferAttentionScore` shape 为 `N/A`，属于解析缺口，不是新 Type。
- Other Ops：resolved `1740`，unresolved `630`，relative error max `374.256`，p95 `1.128`，median `0.563`。top tail 仍是 `GatherV2` 缺 indices 的低置信 fallback。

已知不支持/遗留：

- 通信类 `HcomAllReduce`、`HcomAllGather` 仍按规则排除。
- transformer/vector fusion 类：`MlaPrologV3`、`DynamicQuant`、`AddRmsNormDynamicQuant`。
- 忽略项：`AutomaticBufferFusionOp` 是非固定 pattern 的融合包装算子，无法仅从 `Type/shape` 直接评估。
- Attention shape `N/A` 行：`FusedInferAttentionScore` 140 行，需要 profiling 补 shape 或从上下文恢复规格。

本验证集 stage1 新增 unsupported Type：

- `FloorDiv`
- `FloorMod`
- `ReduceMax`
- `GatherElementsV2`
- `Maximum`
- `Cumsum`
- `Tril`
- `LogicalNot`
- `Data`
- `Unpack`

stage2 已完成覆盖：

- `FloorDiv`
- `FloorMod`
- `ReduceMax`
- `GatherElementsV2`
- `Maximum`
- `Cumsum`
- `Tril`
- `LogicalNot`
- `Unpack`

stage2 后 `other_ops` resolved 从 `1740` 增加到 `1860`，unresolved 从 `630` 降到 `510`，median relative error 从 `0.563` 降到 `0.515`。剩余新增 unsupported 仅 `Data` 5 行；`GatherV2` 缺 indices 仍是最大误差 tail。

## 2026-05-28 FIA PA-NZ cache parser 修复

本轮继续处理 910C longcat 验证集中的 `FusedInferAttentionScore` 高估问题。

源码依据：

- `ops-transformer-master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md` 说明 PagedAttention 使用 block table 和 blockSize 管理 KV cache。
- PA 场景下 KV cache 可为 `BnNBsD` 或 NZ 形式；本样本 K/V shape 为 `[161,1,32,128,16]`，应解释为 `blockNum=161, KV_N=1, D/16=32, blockSize=128, D0=16`，不是 `B=161`。

实现：

- `tools/attention_eval/common.py` 新增 FIA PA cache parser。
- parser 从 query 得到逻辑 batch/q_seq/head_dim，从 block table `[4,40]` 得到 `B=4` 和 `kv_seq=40*128=5120`。
- 5D NZ cache 标记为 `layout=pa_nz_cache`。
- `tools/attention_eval/api.py` 新增 `fused_infer_pa_short_prefill_template_factor`，避免把普通 FIA short-prefill 的高 template factor 套到 PA cache 路径。

验证：

- 命令：`python3 tools/eval_ops.py --op-kind attention --profiling example_profilings/910C/longcat_kernel_details.csv --config configs/ascend_910c.json --output /tmp/longcat_910c_attention_stage3.csv --unresolved-output /tmp/longcat_910c_attention_stage3_unresolved.csv`
- relative error max：`14.075 -> 3.362`
- p95：`13.686 -> 3.249`
- median：`13.237 -> 3.119`
- duration/estimate median：`0.070 -> 0.243`

剩余问题：

- profiling 缺 block table 实际值和 `actualSeqLengthsKv` 数组，当前只能按 `maxBlockNumPerSeq * blockSize` 估计 active KV 范围。
- 继续降低 PA cache 流量会变成对单个样本反推 active block 数；因此剩余 tail 记录为需要 runtime block table/actualSeqLengths 后再继续。

## 2026-05-27 AutomaticBufferFusionOp 口径调整

用户确认 `AutomaticBufferFusionOp` 是融合算子，且不是固定 pattern 融合，无法从 `Type/shape` 直接评估。

本轮处理：

- `AutomaticBufferFusionOp` 从后续 transformer/vector fusion 建模 TODO 中移除。
- 在 `eval_results/20260527T075806Z_e632e87_longcat910c_validation/unsupported_types.csv` 中改为 `ignored_unestimable_fusion`。
- 文档中保留其 unresolved 统计背景，但不再把它视为需要补模型的 unsupported Type。

## 2026-05-28 other_ops transformer/vector fusion 增量

本轮继续按 source/tiling 逻辑处理剩余 other_ops TODO，不引入无关拟合项。

新增代码覆盖：

- `DynamicQuant` / `DynamicQuantV*`：对齐 `ops-nn/quant/dynamic_quant`，按逐行 reduce max、scale、quant 输出和可选 smooth pass 估计。
- `AddRmsNormDynamicQuant` / `AddRmsNormDynamicQuantV2`：对齐 `ops-nn/norm/add_rms_norm_dynamic_quant`，按 add、RMS reduce/rstd、gamma/beta/smooth、quant 输出和 scale 输出估计。
- `RotaryMul`：对齐 `ops-transformer-master/posembedding/apply_rotary_pos_emb`，归入 elementwise/vector rope fusion。
- `MlaPrologV3`：对齐 `ops-transformer-master/attention/mla_prolog_v3`，按四个内部 matmul、norm/rope vector pass 和 active cache 写入估计；报告新增 `cube_compute_us` 字段承载融合 kernel 内部 Cube 子图成本。
- `ReverseV2`、`ReduceAny`、`ArgMaxWithValue`：补入 layout/reduction 分类和源码路径。

验证命令：

```bash
python3 -m compileall tools
python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/910C/longcat_kernel_details.csv --config configs/ascend_910c.json --output /tmp/longcat_910c_other_ops_final.csv --unresolved-output /tmp/longcat_910c_other_ops_final_unresolved.csv
python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/profiling_with_model_code/ds3.2/ASCEND_PROFILER_OUTPUT/kernel_details.csv --config configs/ascend_910c.json --output /tmp/ds32_other_ops_stage6.csv --unresolved-output /tmp/ds32_other_ops_stage6_unresolved.csv
python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py /tmp/longcat_910c_other_ops_final.csv
python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py /tmp/ds32_other_ops_stage6.csv
```

验证结果：

- longcat 910C other_ops：resolved `2210`，unresolved `160`；relative error max `374.2560`，p95 `1.0833`，p90 `0.9232`，median `0.5095`，duration/estimate median `1.4044`。unresolved 只剩 `AutomaticBufferFusionOp` 155 行和 `Data` 5 行。
- ds3.2 other_ops：resolved `3680`，unresolved `170`；relative error max `16.6354`，p95 `1.4422`，p90 `0.8877`，median `0.6322`，duration/estimate median `1.4200`。unresolved 为 `LightningIndexerQuant` 90 行、`Data` 40 行、`AutomaticBufferFusionOp` 40 行。

新识别问题和处理结论：

- `GatherV2` 仍是两组最大 tail。profiling 缺 indices 实际访问范围，当前 low-confidence 单点 fallback 会高估；后续应输出 bounds，而不是调小随机访问因子拟合。
- `LightningIndexerQuant` 当前源码快照只找到 experimental 示例，未找到 op_host/op_kernel 主实现，降级为 `source_unavailable` 遗留。
- `DynamicQuant`、`RotaryMul` 小 shape 样本仍存在固定开销残留，但缺少 tiling/runtime 证据，不加 per-op 校准项。
- `MlaPrologV3` 已能估计内部 matmul 和 active cache 写入，但缺 `tiling_key`、`cache_index`、`actual_seq_len`，置信度保持 low。

## 2026-05-28 QuantBatchMatmulV3 longcat 残留审计

针对 longcat 910C `QuantBatchMatmulV3 M=4,N=12288,K=3072` 高估问题，已对照 ds3.2 full_quant 样本和 `ops-nn/matmul/quant_batch_matmul_v3` 源码检查。

源码观察：

- BF16 dequant 路径通过 `blockIdx` 映射 `mDim/nDim`，每 core 循环 `nLoops`。
- INT32/full_quant 路径使用 `usedCoreNum`、`mTileCntL2/nTileCntL2`、`realRound_` 等 L2 tile 和 block 循环。
- ds3.2 `M=4,N=4096/36864,K=7168` full_quant 样本在现有量化 epilogue 模型下表现合理；直接降低 longcat epilogue 项会破坏这些样本。

结论：

- 当前 profiling 缺 exact `baseN/singleCoreN/tileL2cache/template key`，无法解释 longcat 与 ds3.2 full_quant 的分支差异。
- 不修改模型，不引入针对 longcat 形状的校准项；该问题记录为遗留，后续需要 exact quant tiling 或更细 MTE/fixpipe 计数器后再恢复。

## 2026-05-28 LightningIndexerQuant alias 修正

用户指出 profiling 中的 `LightningIndexerQuant` 可能对应源码 `QuantLightningIndexer`。复查源码后确认：

- 源码主路径：`ops-transformer-master/attention/quant_lightning_indexer`
- 注册 Type：`OP_ADD(QuantLightningIndexer)`
- 文档：`ops-transformer-master/attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md`
- profiling 输入与源码原型匹配：INT8 query/key、weights、query/key dequant scale、actual seq length、block table，输出 `sparse_indices`。

实现调整：

- 新增通用 Type alias 机制，默认包含 `LightningIndexerQuant -> QuantLightningIndexer`。
- alias 是评估入口行为，不写入 `configs/ascend_*.json`，避免把命名兼容误绑定到硬件平台。
- CLI 新增 `--disable-type-aliases`，可关闭默认 alias；新增 `--type-alias FROM=TO`，可为单次运行追加其他映射。
- 报告新增 `canonical_type` 和 `type_alias_applied`，保留 profiling 原始 `type`。
- 新增 `lightning_indexer_fusion` 模型，对齐 `QuantLightningIndexer` 的 INT8 QK、PA block、workspace score、topK sparse index 输出。由于 profiling 缺 `sparse_count/sparse_mode/actual_seq_lengths/block_table` 实际值，当前为 `source_strategy_replay_missing_runtime_values`，置信度 low。

验证：

```bash
python3 -m json.tool configs/ascend_910c.json
python3 -m json.tool configs/ascend_910b4.json
python3 -m json.tool configs/ascend_910b4_1.json
python3 -m compileall tools
python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/profiling_with_model_code/ds3.2/ASCEND_PROFILER_OUTPUT/kernel_details.csv --config configs/ascend_910c.json --output /tmp/ds32_other_ops_alias2.csv --unresolved-output /tmp/ds32_other_ops_alias2_unresolved.csv
python3 tools/eval_ops.py --op-kind other_ops --profiling example_profilings/profiling_with_model_code/ds3.2/ASCEND_PROFILER_OUTPUT/kernel_details.csv --config configs/ascend_910c.json --disable-type-aliases --output /tmp/ds32_other_ops_no_alias2.csv --unresolved-output /tmp/ds32_other_ops_no_alias2_unresolved.csv
```

结果：

- 默认 alias 开启：ds3.2 other_ops resolved `3770`，unresolved `80`；新增 `lightning_indexer_fusion` 90 行，median duration/estimate `1.84`。
- 关闭 alias：resolved `3680`，unresolved `170`；`LightningIndexerQuant` 90 行回到 unresolved。
- 当前 ds3.2 top tail 仍是 `GatherV2` 缺 indices；`QuantLightningIndexer` 不是最大误差来源。

## 2026-05-28 待校准算子正向评估方案

用户要求新增面向“待加强、低置信算子”的正向 kernel 评估方案：不能依赖实测 profiling 或 actual tiling 来校准硬件利用率，只能从 kernel 实现、host tiling 逻辑和硬件配置出发。已新增 `to_be_calib/README.md` 记录设计。

本轮重点以 `MlaPrologV3` 为 P0 方案对象，源码依据如下：

- `ops-transformer-master/attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp`：V3 device kernel 入口，模板参数包含 `CacheMode`、`Scenario`、`QuantMode`、`SplitMMode`、`CvMode`，默认 `KERNEL_TYPE_MIX_AIC_1_2`；arch35 下包含 splitN 和 splitM 实现。
- `ops-transformer-master/attention/mla_prolog_v3/op_host/mla_prolog_v3_tiling_register.cpp`：V3 直接注册 `TilingMlaProlog`。
- `ops-transformer-master/attention/mla_prolog/op_host/mla_prolog_tiling.cpp`：共享 host tiling，包含 shape 解析、四个 Matmul 切分、workspace 计算、tiling key 生成和 blockDim 设置。
- `ops-transformer-master/attention/mla_prolog_v3/docs/MlaPrologV3算子设计介绍.md`：说明四个 Matmul、RMSNorm、Rope、cache 写入，以及 AIC/AIV 同步点。

方案要点：

- 新增评估语义 `source_tiling_search_no_calib`：在没有 actual tiling 时，只枚举源码中存在的 tile/split/template 候选，并用硬件配置做容量过滤和成本计算。
- `MlaPrologV3` 解析 `B/S/T/He/Hcq/Hckv/N/D/Dr/blockNum/blockSize`，并根据 attr、dtype、optional scale 推断 `CACHE_MODE`、`SCENARIO`、`QUANT_MODE`、`ACTUAL_SEQ_LEN_MODE`、`SPLIT_M_MODE`、`CV_MODE`。
- 四个 Cube 子图按源码维度建模：`MatmulCq(M=stepBatch,N=Hcq,K=He)`、`MatmulCkvKr(M=stepBatch,N=Hckv+Dr,K=He)`、`MatmulQcQr(M=stepBatch,N=N*(D+Dr),K=Hcq)`、`MatmulQn(M=stepBatch,N=Hckv,K=D)`。
- Vector 子图按 `RmsNormCq`、`RmsNormCkv`、`RopeQr/RopeKr`、DynamicQuant/Dequant、cache scatter 拆分；workspace 复用 `CalcWorkSpace` 公式。
- 调度估计区分 `lower_bound_us`、`source_schedule_bound_us`、`upper_bound_us`，`estimated_us` 取源码 DAG 约束下的 bound；缺 `actualSeqLen/cacheIndex/quant attr` 时输出区间均值和诊断标签。
- 正向评估策略建议放独立配置，不写入 `configs/ascend_*.json`，避免把算子策略误绑定到硬件平台。

后续实施 TODO：

1. 新增 `source_tiling_search_no_calib` 入口和报告字段。
2. 实现 `MlaPrologV3` parser，能从 profiling shape/attr 推断 `MlaPrologBaseShapeInfo`。
3. 重放 `TilingMlaProlog` 的 tiling key、四个 Matmul 切分和 workspace 公式。
4. 增加候选 tiling 容量过滤、Cube/Vector/GM/workspace/sync 分项成本。
5. validation-only 对比 profiling 残差，不回写经验校准项。
6. 将 `QuantLightningIndexer` 纳入同一正向评估入口，缺 sparse runtime 值时输出区间均值。
