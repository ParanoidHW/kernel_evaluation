# 待校准算子正向评估方案

## 目标

`to_be_calib` 用于沉淀“待加强、低置信算子”的正向评估方案。这类算子暂时没有足够 profiling 样本、actual tiling 或 runtime 输入值，不能通过实测结果反推硬件利用率、效率系数或 per-shape 校准项；评估只能从 CANN kernel 实现、host tiling 逻辑和硬件配置出发，给出可解释的估计值、区间和诊断标签。

该目录中的方案不是拟合记录。后续即使拿到实测数据，也只能用于验证残差来源和发现源码分支缺口，不能用来增加与 kernel/tiling 无关的旋钮、超参或校准项。

## 适用范围

优先处理条件如下：

| 优先级 | 算子 | 当前状态 | 正向评估依据 | 主要缺口 |
| --- | --- | --- | --- | --- |
| P0 | `MlaPrologV3` | 已有低置信 fallback，需增强为源码 tiling 搜索 | `ops-transformer-master/attention/mla_prolog_v3` 复用 `mla_prolog` host tiling，device kernel 拆为四个 Matmul、RMSNorm、Rope、cache 写入和多处 AIC/AIV 同步 | profiling 通常缺 `tiling_key`、`actual_seq_len`、`cache_index` 实际值 |
| P1 | `QuantLightningIndexer` / `LightningIndexerQuant` | 已支持可配置 type alias 和低置信估计 | `ops-transformer-master/attention/quant_lightning_indexer` | 缺 `sparse_count`、`sparse_mode`、`actual_seq_lengths`、`block_table` 实际值 |
| P2 | Transformer/vector fusion | 已有源码分类和粗模型 | `ops-transformer-master/posembedding`、`ops-nn/quant`、`ops-nn/norm`、`ops-math/math` | 缺 actual tiling、axis/mode、cache/rope 运行时值或小 kernel 固定开销来源 |
| P3 | Index/scatter/routing | 已有 analytic fallback，top tail 明显 | `ops-nn/index`、`ops-transformer-master/moe` | indices、mask selected count、routing/token 分布缺失 |
| P4 | Layout/memory fallback | 已有 source strategy replay，但缺 attrs 时低置信 | `ops-math/conversion`、`ops-nn/index` | perm、axis、begin/size/stride、multiples、diagonal 等 attrs 缺失 |
| P5 | CV/图像算子 | 已有评估计划，尚未逐个落地 | `ops-cv`、`ops-nn/conv` | profiling 样本和 layout/ROI 运行时元数据不足 |
| P6 | MatMul/Attention/GMM fallback tiling | 已有专用评估器和遗留分类 | `ops-nn/matmul`、`ops-transformer-master/attention`、`ops-transformer-master/gmm` | exact tiling、runtime KB、模板 key、routing/block table 缺失 |

显式排除：

| 类型 | 处理方式 | 原因 |
| --- | --- | --- |
| `AutomaticBufferFusionOp` | 忽略 | 非固定 pattern 融合，无法从单个 Type 直接恢复内部算子图 |
| `Data` / host 行 / 通信行 | 暂不评估 | 不是当前 kernel 计算模型目标 |
| 缺 shape、dtype、format 的行 | 输出 unresolved | 无法构造物理工作量 |

## 通用正向评估流程

1. 解析 profiling CSV 或用户输入规格，得到 canonical op type、shape、dtype、format、attr 和硬件配置名。
2. 根据 type 映射定位源码路径，优先使用本仓库内 `ops-nn-master`、`ops-transformer-master`、`ops-math`、`ops-cv`；源码缺失时标记 `source_unavailable`。
3. 从 op_host 提取 tiling key、blockDim、workspace、template 分支、数据格式和硬件平台约束。
4. 从 op_kernel 提取实际计算 DAG、Cube/Vector 子任务、GM/UB/L1/L0 搬运、fixpipe、atomic、sync 和特殊分支。
5. 在没有 actual tiling 时，根据源码规则枚举 candidate tiling；只允许枚举源码中存在的 tile、split、template 和硬件约束。
6. 对 candidate 做容量过滤：L0A/L0B/L0C、L1、UB、workspace、对齐、core 数、CV ratio、dtype 支持。
7. 分别计算 Cube compute、Vector compute、GM/HBM、workspace spill、sync、launch 和 occupancy/tail imbalance。
8. 选择“当前 kernel 可执行候选”中的最低 source-schedule bound 作为 `estimated_us`；同时输出 lower/upper bound 和所有未知项。
9. 报告 `confidence=low`，直到能拿到 actual tiling 或足够的 runtime 元数据验证分支选择。

## 建模约束

| 约束 | 要求 |
| --- | --- |
| 不使用实测校准 | 不从 `duration_us` 拟合硬件利用率、pipeline efficiency、launch 常数或 per-op scale |
| 不新增无关旋钮 | 参数必须来自源码常量、tiling 规则、硬件配置或输入规格 |
| 区分估计语义 | `actual_tiling`、`source_tiling_search`、`ideal_lower_bound` 必须在报告中分开 |
| 保留未知项 | 缺 runtime 值时输出诊断标签和区间，不用单点假设掩盖 |
| 单调性检查 | M/N/K、token 数、cache 写入量、量化附加输出增加时，估计不应反向下降，除非源码分支明确改变 |

## 报告字段建议

| 字段 | 含义 |
| --- | --- |
| `source_eval_mode` | `source_tiling_search_no_calib` / `source_fallback_no_runtime` / `source_unavailable` |
| `canonical_type` | alias 后的源码算子 Type |
| `source_paths` | op_host/op_kernel/文档路径 |
| `tiling_key_guess` | 根据 attr/shape 推断的 tiling key 字段，不等同 actual tiling |
| `candidate_count` | 枚举并通过容量过滤的 tiling 候选数 |
| `selected_candidate_id` | 被选中的候选 ID |
| `cube_compute_us` | Cube 子图计算耗时估计 |
| `vector_compute_us` | Vector 子图计算耗时估计 |
| `gm_us` | 直接 GM/HBM 读写耗时估计 |
| `workspace_us` | workspace 中间结果写读耗时估计 |
| `sync_us` | 源码可见 barrier/fence 的估计或未知标签 |
| `launch_us` | 硬件配置中明确给出的 launch 开销；没有配置时不反推 |
| `lower_bound_us` | 理想资源下界，不能当作当前 kernel 耗时 |
| `source_schedule_bound_us` | 按源码 DAG 和资源约束得到的当前 kernel bound |
| `upper_bound_us` | 保守串行 bound |
| `diagnosis` | 缺失 runtime、分支不确定、容量边界、低 occupancy 等标签 |

## MlaPrologV3 正向评估设计

### 源码依据

| 类型 | 路径 | 结论 |
| --- | --- | --- |
| 算子文档 | `ops-transformer-master/attention/mla_prolog_v3/docs/MlaPrologV3算子设计介绍.md` | 说明四个 Matmul、RMSNorm、Rope、cache 写入和 AIC/AIV 同步关系 |
| V3 kernel 入口 | `ops-transformer-master/attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp` | 根据 `CacheMode`、`Scenario`、`QuantMode`、`SplitMMode`、`CvMode` 等模板参数选择 splitN/splitM 实现；默认 `KERNEL_TYPE_MIX_AIC_1_2` |
| V3 host tiling | `ops-transformer-master/attention/mla_prolog_v3/op_host/mla_prolog_v3_tiling_register.cpp` | V3 直接注册到 `TilingMlaProlog` |
| 共享 tiling | `ops-transformer-master/attention/mla_prolog/op_host/mla_prolog_tiling.cpp` | 给出 shape 解析、四个 Matmul 切分、workspace 公式、tiling key 和 blockDim |
| 共享 tiling 数据 | `ops-transformer-master/attention/mla_prolog/op_host/mla_prolog_tiling.h` | 给出 input/output/attr index、`CACHE_MODE`、`QUANT_MODE`、`ACTUAL_SEQ_MODE`、`MlaPrologBaseParams` |

### 输入解析

| 逻辑量 | 推断来源 | 用途 |
| --- | --- | --- |
| `B`、`S1`、`T` | `tokenX`、`query`、`cacheIndex` shape；TND/BSND/PA 模式按 `cacheMode` 分支解析 | token 数、stepBatch、active row |
| `He` | `tokenX` 或 `weightDq` K 维 | `MatmulCq`、`MatmulCkvKr` 的 K |
| `Hcq` | `rmsnormGammaCq` 或 `weightDq` N 维 | `MatmulCq` 输出和 `MatmulQcQr` K |
| `Hckv` | `rmsnormGammaCkv`、KV cache 或输出 | `MatmulQn` N、KV cache 写入 |
| `N`、`D`、`Dr` | `query`、`queryRope`、`weightUqQr`、`ropeSin/ropeCos` | head 数、nope head dim、rope dim |
| `blockNum`、`blockSize` | PA cache shape、cache mode | cache 写入和 page/block metadata |
| `weightQuantMode`、`kvQuantMode`、`queryQuantMode` | V3 attrs 或 dtype/optional scale 组合 | `QUANT_MODE`、附加 dequant/quant/vector 成本 |
| `actualSeqLen` | optional input 是否存在；若缺实际值，只能用 max shape 上界或区间 | active token 和 cache 写入区间 |

### tiling key 与模板分支

| 字段 | 源码规则 | 正向评估处理 |
| --- | --- | --- |
| `CACHE_MODE` | `BSND/TND/PA_BSND/PA_NZ/PA_BLK_*` 由 attr 字符串解析；V3 cacheIndex 为 input 11 | 优先读 attr；profiling 缺 attr 时从 cache shape 和 format 推断，失败标记 `cache_mode_unknown` |
| `SCENARIO` | `NO_QUANT` 为 1，其他量化为 2 | 由 weight dtype、scale tensor 和 attr 推断 |
| `QUANT_MODE` | tiling 中枚举 0 到 15，覆盖 partial/full/MXFP8/FP8/HIF8 与 KV cache quant | 无 attr 时只生成可能集合，不强行选择单一模式 |
| `ENABLE_DEQUANT_OPTIONAL` | INT8 全量化且 `N>=8`，或 FP8/HIF8/MXFP8 场景触发 | 按源码条件推断 |
| `ENABLE_GROUP_COMPUTE_OPTIONAL` | partial quant、`T=1`、`Nkv=8`、AIV/AIC 达阈值且 CV 非 1:1 | 按源码条件推断，同时把 AIC/AIV 改为 16/32 |
| `EMPTY_TENSOR_MODE` | empty query/cache 特判 | shape 为 0 时走 empty 路径，blockDim=1 或跳过对应子图 |
| `ACTUAL_SEQ_LEN_MODE` | optional `actualSeqLen` 存在时使能 | 缺值时输出 active token 区间 |
| `SPLIT_M_MODE` | MXFP8 full quant 且非 TND 时启用 splitM | 按源码条件推断 |
| `CV_MODE` | 默认 1:2；AIV=AIC 时 1:1，且 1:1 只支持 no quant + PA_BSND/PA_NZ | 从硬件 `aic_num/aiv_num` 推断并校验限制 |

### 四个 Matmul 候选 tiling

| 子图 | 源码维度 | 源码切分 | 评估公式 |
| --- | --- | --- | --- |
| `MatmulCq` | `M=stepBatchSize`，`N=Hcq`，`K=He` | splitM 时 `singlecoreHeadSizeCq=Hcq, mm1BlockNum=aicNum`；否则 `CalcSingleCoreN(Hcq,aicNum,32/dtype_size)` 且至少 64 | FLOPs=`2*M*N*K`，按 dtype Cube peak、block 数、tail imbalance 估计 |
| `MatmulCkvKr` | `M=stepBatchSize`，`N=Hckv+Dr`，`K=He` | splitM 时全 N；否则 AIC>=9 时 baseN=64，其他用 `CalcSingleCoreN` | FLOPs=`2*M*N*K`；empty cache 时跳过 |
| `MatmulQcQr` | `M=stepBatchSize`，`N=N*(D+Dr)`，`K=Hcq` | group/dequant/splitM 分支分别按源码计算 `singlecoreHeadSizeQcQr` | FLOPs=`2*M*N*K`；量化场景额外计 dequant/dynamic quant |
| `MatmulQn` | `M=stepBatchSize`，`N=Hckv`，`K=D`，K stride 为 `D+Dr` | splitM 时 `mm4BlockNum=aicNum`；否则按 head 数分核 | FLOPs=`2*M*N*K`；部分全量化 per-tensor 场景会写 BF16 workspace |

`stepBatchSize=min(128,T)`，`vectorBlockNum=min(stepBatchSize,aivNum)`，`stepNumHeadDequant` 在 `D=128` 时取 `min(64,N)`，否则取 `min(16,N)`。这些参数直接来自 host tiling，不允许根据实测样本调整。

### Vector、内存和同步成本

| 成本项 | 计数依据 | 说明 |
| --- | --- | --- |
| `RmsNormCq` | `stepBatchSize * Hcq` 的 sumsq、rsqrt、scale、读写 workspace | `queryNormFlag` 使能时增加 query norm 输出写 |
| `RmsNormCkv` | `stepBatchSize * Hckv` | 与 KR rope/cache scatter 共享 vector pipeline |
| `RopeQr/RopeKr` | `stepBatchSize * N * Dr` 或 cache KR 写入维度 | 读取 sin/cos，输出 queryRope 和 KR cache |
| Dynamic quant/dequant | `QUANT_MODE`、scale/smooth 输入和输出 | partial/full/per-tile 分支分别计 scale 读写和 int8/fp8 输出 |
| GM/HBM | tokenX、weights、gamma、sin/cos、scale、cacheIndex、cache read/write、query/queryRope 输出 | 权重按当前 kernel 单次执行读入，不做跨 kernel cache 复用假设 |
| Workspace | 直接使用 `CalcWorkSpace` 中的公式重放 | splitM 时 `mm*_Mult` 和 `dequantScaleMult` 按源码放大 |
| Sync | 文档列出的 `SYNC_MMCQ_NOMRCQ`、`SYNC_MMCKVKR_NORMCKV`、`SYNC_NOMRCQ_MMQCQR`、`SYNC_MMQCQR_ROPEQR`、`SYNC_ALL_CUBE`、`SYNC_ALL_VECTOR` | 若硬件配置没有 sync latency，只报告 `sync_unknown` 并进入区间 |

### 调度 bound

MlaPrologV3 不是四个 Matmul 的简单求和。source-schedule bound 使用源码 DAG：

```text
AIC: MatmulCq -> MatmulCkvKr -> wait RmsNormCq -> MatmulQcQr -> all-cube sync -> MatmulQn
AIV: GetSinCos -> wait MatmulCq -> RmsNormCq -> signal MatmulQcQr
     -> wait MatmulCkvKr -> RmsNormCkv/RopeKr/cache -> wait MatmulQcQr -> RopeQr/query output
```

第一版实现建议输出三个值：

| 值 | 公式语义 |
| --- | --- |
| `lower_bound_us` | `max(total_cube_compute, total_vector_compute, total_hbm)`，只作为物理下界 |
| `source_schedule_bound_us` | 按上述 DAG 做双资源 list scheduling，并加入 workspace、sync、launch 中可确定的部分 |
| `upper_bound_us` | Cube、Vector、HBM、workspace、sync 的保守串行和 |

`estimated_us` 取 `source_schedule_bound_us`。当 `actualSeqLen/cacheIndex/quant attr` 缺失导致 candidate 集合不唯一时，输出区间均值作为默认单点，同时保留 `lower_bound_us/upper_bound_us` 和 `diagnosis`。

## 其他低置信算子设计

### QuantLightningIndexer / LightningIndexerQuant

`LightningIndexerQuant` 已通过可配置 alias 映射到源码 Type `QuantLightningIndexer`。该映射属于评估入口策略，不写入硬件配置。

| 项 | 设计 |
| --- | --- |
| 源码路径 | `ops-transformer-master/attention/quant_lightning_indexer` |
| 输入解析 | query/key INT8 shape 推断 `q_tokens`、`q_heads`、`head_dim`、KV block/page 数；输出 sparse index shape 推断 `sparse_count` 上界 |
| 计算子图 | INT8 QK score、PA block 遍历、score workspace、TopK/sparse index 选择、index 输出 |
| tiling 搜索 | 按 Q token、KV block、head 维和 sparse_count 枚举 block 粒度；过滤 UB/workspace 和 AIC/AIV 分配 |
| 成本项 | Cube INT8 QK、Vector TopK/排序选择、workspace score 写读、sparse index 写出、block table/actual seq 读取 |
| 缺失运行时值 | `sparse_count`、`sparse_mode`、`actual_seq_lengths_query`、`actual_seq_lengths_key`、`block_table_values` |
| 输出语义 | `source_tiling_search_no_calib`；缺运行时值时输出 `lower_bound_us/source_schedule_bound_us/upper_bound_us`，单点取区间均值 |

不能用实测 duration 调整 TopK factor。若 `block_table` 或 actual seq 缺失，评估必须保留 `missing_sparse_runtime_values`，并给出 dense-all-block 与 active-block 两个边界。

### Rope 与 KV cache 融合类

覆盖 `RotaryMul`、`RotaryPositionEmbedding`、`InterleaveRope`、`KvRmsNormRopeCache`、`QkvRmsNormRopeCache` 等。

| 算子/族 | 源码路径 | 正向建模方案 | 缺口与输出 |
| --- | --- | --- | --- |
| `RotaryMul` | `ops-transformer-master/posembedding/apply_rotary_pos_emb` | 按 query/key head 维拆成 sin/cos 读、even/odd rotate、mul/add、输出写；按 UB tile 枚举行块 | 缺 rotary mode、interleave flag 时输出 `rope_mode_unknown` |
| `RotaryPositionEmbedding` | `ops-transformer-master/posembedding/rotary_position_embedding` | 同上，额外处理 position ids/gather sincos 的随机读边界 | 缺 position ids 值时输出 gather 边界 |
| `InterleaveRope` | `ops-transformer-master/posembedding/interleave_rope` | 将 head dim 拆成 pair/interleave lane，估计重排、sin/cos、mul/add、GM 写回 | 缺 interleave layout/position 值时使用区间 |
| `KvRmsNormRopeCache` | `ops-transformer-master/posembedding/kv_rms_norm_rope_cache` | DAG：KV RMSNorm reduce -> scale -> rope -> PA/BSND cache scatter；按 token/head/cache block 枚举 vector tile | 缺 cache_index/actual_seq 时输出 active cache 写入上下界 |
| `QkvRmsNormRopeCache` | `ops-transformer-master/posembedding/qkv_rms_norm_rope_cache` | 在 KV 分支基础上增加 Q branch norm/rope 输出；共享 sin/cos 读和 cache scatter | 缺 q/kv 分支 attrs 时保持 low confidence |

这类算子是 AIV/vector-heavy，不应通过新增“rope_efficiency”拟合。可解释参数只能来自：元素个数、head dim、pair/interleave 布局、sin/cos 读取方式、cache 写入格式、UB tile 容量、AIV 数和 HBM 带宽。

### DynamicQuant / AddRmsNormDynamicQuant / Rsqrt

| 算子/族 | 源码路径 | 正向建模方案 | 缺口与输出 |
| --- | --- | --- | --- |
| `DynamicQuant*` | `ops-nn/quant/dynamic_quant`、`ops-nn/quant/dynamic_quant_v2` | 每行 reduce absmax/max、计算 scale、可选 smooth、quant cast、scale 输出；按 row/hidden 枚举 UB tile | 缺 `dst_type/quant_mode/symmetry` 时输出候选集合 |
| `AddRmsNormDynamicQuant*` | `ops-nn/norm/add_rms_norm_dynamic_quant` | add residual、RMS reduce、rsqrt、gamma/beta/smooth、quant、scale 输出；多输出按 output shape 计流量 | 缺 optional smooth/bias 语义时区间 |
| `MultiAddRmsNormDynamicQuant` | `ops-nn/norm/multi_add_rms_norm_dynamic_quant` | 多 residual 输入累加后走 RMSNorm+quant；输入数决定读流量和 add pass | 缺融合输入实际个数时 unresolved 或区间 |
| `Rsqrt` | `ops-math/math/rsqrt` | 单输入单输出 transcendental vector pipeline；按 dtype 和元素数估算 vector ops 与 HBM | 小 shape 主要受 launch/sync floor 影响，硬件配置缺 floor 时标记 `launch_unknown` |

`Rsqrt` 在 qwen7b other_ops 中大量出现，但它本身不是 tiling 难点；主要问题是小 kernel 固定开销和调度开销。若硬件配置没有 launch/sync 参数，不能从 qwen 样本反推一个专用 floor。

### Index / Scatter / Mask / Routing

这类算子不能在缺运行时值时给出 exact 单点，因为 indices、mask 和 routing 分布本身决定实际访问量和 cache locality。正向方案应改为 bounds-first。

| 算子/族 | 源码路径 | bounds 设计 | 缺口 |
| --- | --- | --- | --- |
| `GatherV2/GatherV3/GatherElements` | `ops-nn/index/gather_v2`、`ops-nn/index/gather_v3`、`ops-nn/index/gather_elements` | 下界：线性/重复 indices 命中；上界：随机 GM gather；区间按输出元素、index dtype、axis 维度和 cacheline 放大 | indices 值、axis |
| `Scatter*` | `ops-nn/index/scatter`、`ops-nn/index/scatter_nd`、`ops-nn/index/scatter_elements_v2` | 下界：无冲突连续写；上界：随机写+atomic/冲突；按 update 元素和 index fanout 建区间 | indices、冲突率、reduction mode |
| `MaskedSelectV3/NonZero` | `ops-math/conversion/masked_select_v3`、`ops-nn/index/non_zero` | 下界：mask 全 false 或低 selected；上界：mask 全 true；估计 mask scan、prefix/compaction、输出写 | selected count/mask 值 |
| `TopKV2/ArgMaxV2` | `ops-nn/index/apply_top_k_top_p_with_sorted`、`ops-math/math/arg_max_with_value` | 按 K、axis、row 数估计局部排序/归约；缺 K 分布时输出小 K/全排序边界 | K、axis、sorted flag |
| MoE routing | `ops-transformer-master/moe/moe_*` | 按 token 数、expert 数、topK、routing table 建 balanced/extreme bounds；空 expert skip 和 tail imbalance 显式输出 | routing values、per-expert token 分布 |
| `MoeComputeExpertTokens` | `ops-transformer-master/moe/moe_compute_expert_tokens` | 计 routing list scan、expert token count reduce、prefix/offset 输出；balanced/extreme 两端 | expert 分布和 topK |

后续实现时，`estimated_us` 建议默认写区间均值，但报告必须包含 `bounds_min_us/bounds_max_us` 和 `bounds_reason`。`GatherV2` 当前 top tail 不应通过调低 `index_random_access_factor` 修正。

### Layout / Memory 缺 attrs

已有 `source_strategy_replay_missing_attrs`，但缺 attrs 的行不应升级为 actual tiling。

| 算子/族 | 源码路径 | 正向建模方案 | 缺口 |
| --- | --- | --- | --- |
| `Transpose` | `ops-math/conversion/transpose` | 枚举常见 perm：last2 swap、NHWC/NCHW、rank reverse；分别估计 NDDMA/vconv/UB tile 搬运 | `perm` |
| `Slice/StridedSlice/AsStrided` | `ops-math/conversion/slice`、`strided_slice`、`as_strided` | 根据 contiguous slice、strided gather、NDDMA 三类路径给 bounds | begin/size/end/stride/storage_offset |
| `Concat/Split/Pack/Unpack` | `ops-math/conversion/concat`、`split`、`pack` | 枚举 axis 候选，计算每段线性搬运和小 tensor 调度开销 | axis |
| `Tile` | `ops-math/math/tile` | 根据 multiples 枚举广播复制层级；缺 multiples 时用 output/input ratio 边界 | multiples |
| `PadV3` | `ops-math/conversion/pad_v3` | 输入 copy + padding fill；按 padding mode 区分 constant/reflect/edge | pads、mode、constant value |
| `Sort/SortV2` | `ops-math/math/sort`、`ops-math/experimental/math/sort_v2` | 按 row length、axis 和 dtype 建 bitonic/merge/local sort 近似；小 K 与全 axis 排序分开 | axis、descending、stable |
| `RepeatInterleaveV2` | 需按源码定位到 repeat/interleave 实现 | 下界连续复制，上界按 indices/repeats 展开随机/变长复制 | repeats 值、axis |
| `MemSet` | `ops-math/conversion/mem_set` | 有 output shape/dtype 时按 fill GM 写和 vector fill 估计 | profiling 为 `N/A` 时仍 unresolved |

### Conv / CV / 图像类

`Conv2D` 当前在 other_ops 中只是 `cv_regular` fallback，应独立成 Cube-heavy 正向方案，不与 elementwise/vector 混用。

| 算子/族 | 源码路径 | 正向建模方案 | 缺口 |
| --- | --- | --- | --- |
| `Conv2D` | `ops-nn/conv` 或 `ops-nn/conv2d_v2`，本地缺目录时降级 `source_path_pending` | 解析 N/C/H/W、Cout、KH/KW、stride/pad/dilation/group；估计 im2col/Load3D、Cube FLOPs、L0/L1/BT、bias/activation/fixpipe、NC1HWC0/FRACTAL_Z layout | exact tiling、format attrs、group/pad/stride |
| `Conv3DV2` | `ops-nn/conv3d` | 在 Conv2D 基础上增加 D 维 tile 和 z 方向重复加载 | 3D tiling/format |
| `Resize*` | `ops-cv` 或 `ops-nn` resize 路径 | 按 output pixel、插值核、坐标计算、边界处理、GM 读写估计 | align_corners、half_pixel、ROI |
| `GridSample*` | `ops-cv` | 每 output 采样 4/8 邻域，估计 gather、插值、边界 mode | grid 值、padding mode |
| ROI/NMS | `ops-cv` | ROI Align 按 bin/sample ratio；NMS 按 box 数排序和 IoU 矩阵/筛选 bounds | boxes 分布、threshold、selected count |

CV 类优先级低于 transformer/vector fusion 和 index/routing，因为当前 profiling 覆盖较少，且许多输入 attr 不在 CSV 中。

### MatMul / Attention / GMM 缺 actual tiling

这些已有专用评估器，不迁入 other_ops，但也适用“无校准正向方案”的约束。

| 算子/族 | 当前问题 | 正向增强方案 | 缺口 |
| --- | --- | --- | --- |
| base `MatMulV2 M=1` | small-M fallback tiling 残留 | 从 `ops-nn/matmul` 重放 MatmulToMul、disableGemv、L1/L0 copy、sync 流水候选；只在解释 lower-bound violation 后启用 | exact tiling/template key、runtime KB |
| `TransposeBatchMatMul` | transpose/strided 装载开销 | 按 perm、stride、SetTensorA/B trans flag 和 GM offset 估计 MTE2/L1/L0 额外搬运 | perm attrs、exact tiling、MTE/fixpipe counter |
| `QuantBatchMatmulV3` longcat tail | full quant small-M 分支差异 | 继续按 arch35 Weight-NZ、fixpipe、L2 tile、usedCoreNum 重放候选，不新增 shape 校准 | baseN/singleCoreN/tileL2/template key |
| FIA decode | source-strategy replay residual | 重放 decode/paged cache tiling、KV block 元数据、mask/aux 访问；缺失时只给 bounds | FIA exact tiling、cache/block metadata |
| QSFA | 稀疏/TopK/PA 残留 | 按 block table、sparse indices、TopK workspace 建 bounds-first 模型 | block table、sparse indices、runtime TopK |
| `GroupedMatmul` | routing bounds above-bound | 保持 balanced/extreme routing bounds；有 group_list 后才能转单点 | group_list、groupListType、tuningConfig、per-expert tokens |

这些项不得通过扩大经验区间或增加全局 latency floor 来拟合。若缺口未补齐，保持遗留或 bounds。

## 配置建议

正向评估开关不写入硬件平台配置，避免把算子策略误绑定到某个平台。建议后续新增独立策略配置，例如：

```json
{
  "forward_eval": {
    "enabled_ops": ["MlaPrologV3"],
    "allow_empirical_calibration": false,
    "tiling_search": {
      "max_candidates": 4096,
      "select": "min_source_schedule_bound"
    },
    "unknown_runtime_policy": "interval_mean",
    "report_candidate_details": true
  }
}
```

硬件参数仍从 `configs/ascend_*.json` 读取，包括 `aic_num`、`aiv_num`、HBM 带宽、dtype 峰值、L0/L1/UB 容量等。若硬件配置缺 sync/launch 项，则不通过 profiling 反推，只在报告中标记未知。

## 实施 TODO

| 顺序 | 任务 | 验收标准 |
| --- | --- | --- |
| 1 | 新增 `source_tiling_search_no_calib` 评估入口和报告字段 | CLI 能对指定 op 启用正向评估，默认不影响现有 matmul/fa/gmm/other_ops |
| 2 | 实现 `MlaPrologV3` parser | 能从 profiling shape/attr 推断 `MlaPrologBaseShapeInfo`，缺 attr 时给出明确 unresolved/diagnosis |
| 3 | 重放 `TilingMlaProlog` 中的 tiling key、四个 Matmul 切分和 workspace 公式 | 单元测试覆盖 no quant、partial quant、full quant、MXFP8 splitM、empty tensor |
| 4 | 实现候选 tiling 容量过滤和成本模型 | 报告 Cube/Vector/GM/workspace/sync 分项和 candidate 选择结果 |
| 5 | 增加 validation-only 对比 | 使用 profiling 只看残差和分类，不回写经验校准项 |
| 6 | 将 `QuantLightningIndexer` 纳入同一入口 | 缺 runtime 值时输出区间均值和 `missing_sparse_runtime_values` |
| 7 | 将 index/scatter/routing 改为 bounds-first 报告 | `Gather/Scatter/Mask/MoE` 输出 `bounds_min_us/bounds_max_us`，默认单点为区间均值 |
| 8 | 将 layout 缺 attrs 算子枚举常见源码策略候选 | `Transpose/Slice/Pack/Tile/Pad/Sort` 报告候选策略和缺失 attrs |
| 9 | 为 Conv/CV 类建立独立 Cube/vector/memory 模型 | `Conv2D` 不再只走 `cv_regular` fallback；源码缺失时明确 `source_path_pending` |

## 当前风险

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| profiling CSV 缺 attr | 无法唯一确定 `QUANT_MODE`、`cacheMode`、`queryNormFlag` | 输出候选集合和区间，不做隐式校准 |
| 缺 `actualSeqLen/cacheIndex` 实际值 | active token/cache 写入量只能取上界或区间 | 默认 `interval_mean`，报告上下界 |
| 硬件配置缺 sync/launch | 小 shape/fusion kernel latency floor 可能低估 | 标记 `sync_unknown/launch_unknown`，不从实测拟合 |
| 权重 cache 复用未知 | HBM 读权重可能高估或低估 | 默认单 kernel 冷读，后续只有拿到 runtime cache 证据才调整 |
