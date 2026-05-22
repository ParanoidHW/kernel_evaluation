# Profiling 样本说明

本目录保存当前已有的实测 profiling。评估工具主要读取各目录下的 `kernel_details*.csv`，并按 `Type` 字段过滤 MatMul、GroupedMatmul 和 Attention 类 kernel。

## 目录结构

- `910B4/`：按 910B4 归档的基础 profiling 样本。
- `910C/`：按 910C 归档的基础 profiling 样本。
- `profiling_with_model_code/`：带模型代码和 profiler 输出的模型级样本，包括 `ds3.2`、`gemma`、`longcat` 和 `qwen7b`。

不要用全文件最大 `Block Dim` 判断 Cube 核数。部分文件包含 Vector、通信或异常导出的统计项，全局最大值会远高于 AIC 数；评估时应看目标 kernel 行的 `Block Dim`、`Mix Block Dim` 和 `cube_utilization(%)`。

## 样本与报告对应关系

| profiling 输入 | 当前归属/平台 | 主要评估报告 |
|---|---|---|
| `910B4/kernel_details.csv` | 910B4 基础样本，少量 BatchMatMulV2 | `eval_results/matmul_eval_report_910b4.csv` |
| `910B4/kernel_details3.csv` | longcat/910B4 基础样本 | `eval_results/matmul_eval_report_910b4.csv`、`eval_results/attention_eval_report_910b4.csv`、`eval_results/grouped_matmul_eval_report_910b4.csv` |
| `910B4/kernel_details_2.csv` | 910B4 量化 MatMul 样本 | `eval_results/matmul_eval_report_910b4.csv` |
| `910B4/kernel_details_gemma4.csv` | gemma/910B4 基础样本 | `eval_results/matmul_eval_report_910b4.csv`、`eval_results/attention_eval_report_910b4.csv`、`eval_results/grouped_matmul_eval_report_910b4.csv` |
| `910B4/kernel_details_hyimage.csv` | hyimage/910B4 基础样本 | `eval_results/matmul_eval_report_910b4.csv`、`eval_results/attention_eval_report_910b4.csv` |
| `910B4/kernel_details_pangu.csv` | pangu/910B4 MatMulV3 样本 | `eval_results/matmul_eval_report_910b4.csv` |
| `910C/kernel_details4.csv` | 910C FusedInferAttention + MatMul/GMM 样本 | `eval_results/matmul_eval_report_910c.csv`、`eval_results/attention_eval_report_910c.csv` |
| `910C/kernel_details5.csv` | 910C FlashAttention + MatMul/GMM 样本 | `eval_results/matmul_eval_report_910c.csv`、`eval_results/attention_eval_report_910c.csv` |
| `profiling_with_model_code/ds3.2/ASCEND_PROFILER_OUTPUT/kernel_details.csv` | ds3.2/910C 模型级样本 | `eval_results/profiling_with_model_code_ds32_*_910c.csv` |
| `profiling_with_model_code/gemma/ASCEND_PROFILER_OUTPUT/kernel_details.csv` | gemma/910B4 模型级样本 | `eval_results/profiling_with_model_code_gemma_*_910b4.csv` |
| `profiling_with_model_code/longcat/.../ASCEND_PROFILER_OUTPUT/kernel_details.csv` | longcat/910B4 模型级样本 | `eval_results/profiling_with_model_code_longcat_*_910b4.csv` |
| `profiling_with_model_code/qwen7b/ASCEND_PROFILER_OUTPUT/kernel_details.csv` | qwen3-7b/qwen7b，profiling 缺少 Block Dim 字段；Block Num 匹配 910B4，HBM 使用 910B4-1 | `eval_results/profiling_with_model_code_qwen7b_*_910b4_1.csv`、历史 `*_910b4_inferred.csv` |

## Kernel 类型覆盖

- 910B4 基础样本包含 `MatMul`、`MatMulV2`、`MatMulV3`、`BatchMatMul`、`BatchMatMulV2`、`QuantBatchMatmulV3`、`GroupedMatmul` 和 `FusedInferAttentionScore`。
- 910C 基础样本包含 `MatMulV2`、`MatMulV3`、`GroupedMatmul`、`FusedInferAttentionScore` 和 `FlashAttentionScore`。
- ds3.2 模型级样本包含 `MatMul`、`QuantBatchMatmulV3`、`GroupedMatmul`、`TransposeBatchMatMul` 和 `KvQuantSparseFlashAttention`。
- qwen7b 模型级样本包含 `MatMulV2` 和 `FusedInferAttentionScore`，但原始 `kernel_details.csv` 没有可直接使用的 `Block Dim`/`Mix Block Dim` 字段。按 `Block Num` 判断，Cube 侧匹配 910B4；qwen3-7b 专用基线使用 `configs/ascend_910b4_1.json`，HBM 带宽为用户确认的 1.6 TB/s。

## 当前注意事项

- `profiling_with_model_code_matmul_eval_910b4.csv` 和 `profiling_with_model_code_attention_eval_910b4.csv` 是混合报告，包含 ds3.2 的 910C 样本；评估 ds3.2 时应优先看 `*_ds32_*_910c.csv`。
- `GroupedMatmul` 缺少真实 `groupList` 和 `tuningConfigOptional`，不能按普通 MatMul 单点估计判断精度，应优先看 GMM routing-bound 报告字段。
- `KvQuantSparseFlashAttention` 是当前 attention 最大误差来源，需要专门模型；不能继续套普通 FlashAttention/FusedInferAttention 口径。
- `qwen7b` 历史 `910b4_inferred` MatMulV2 报告在 0.8 TB/s HBM 下存在全量物理下界违反；该问题已作为平台配置差异处理，使用 `910B4-1` 后 lower-bound violation 降为 0。剩余误差集中在 `MatMulV2 M=1` small-M memory-bound 路径，应继续从 kernel/tiling 逻辑完善。
