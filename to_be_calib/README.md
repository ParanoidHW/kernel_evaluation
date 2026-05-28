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
| P2 | 常规向量融合算子 | 已有源码分类和粗模型 | `ops-math`、`ops-nn`、`ops-transformer` 中的 elementwise/reduction/norm kernel | 部分算子缺具体 tiling 常量或动态 mask |
| P3 | CV/图像算子 | 已有评估计划，尚未逐个落地 | `ops-cv` | profiling 样本和 layout/ROI 运行时元数据不足 |

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

## 当前风险

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| profiling CSV 缺 attr | 无法唯一确定 `QUANT_MODE`、`cacheMode`、`queryNormFlag` | 输出候选集合和区间，不做隐式校准 |
| 缺 `actualSeqLen/cacheIndex` 实际值 | active token/cache 写入量只能取上界或区间 | 默认 `interval_mean`，报告上下界 |
| 硬件配置缺 sync/launch | 小 shape/fusion kernel latency floor 可能低估 | 标记 `sync_unknown/launch_unknown`，不从实测拟合 |
| 权重 cache 复用未知 | HBM 读权重可能高估或低估 | 默认单 kernel 冷读，后续只有拿到 runtime cache 证据才调整 |
