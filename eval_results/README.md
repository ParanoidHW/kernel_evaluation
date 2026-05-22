# 评估结果快照

本目录保存本地评估汇总，便于按时间和 commit 回溯。每次评估放在独立子目录中，避免多个 CSV 混在一起。

- `<UTC时间>_<commit>/eval_summary.csv`：该次评估的汇总表。
- `<UTC时间>_<commit>/metadata.txt`：该次评估的时间、commit 和报告数量。
- `LATEST`：最近一次评估子目录名。

生成方式：

```bash
python3 tools/summarize_eval_results.py <report.csv> [...]
```

汇总默认使用与大 shape 分析一致的过滤口径：`duration_us >= 10`，且 `block_dim/mix_block_dim >= 0.8 * aic_num` 或 `cube_utilization_pct >= 70`。对 GMM 报告，误差字段使用 routing-bound 区间误差。

## 当前报告组织

- 当前只保留带 commit 的快照子目录，例如 `<UTC时间>_<commit>/eval_summary.csv`。
- 根目录下旧的裸 CSV 报告已经清理，因为它们缺少稳定 commit 对应关系，容易和当前代码或 `LATEST` 产生歧义。
- 后续新增评估结果应通过 `tools/summarize_eval_results.py` 写入新的 `<UTC时间>_<commit>/` 子目录。
- 如果需要保留详细 resolved/unresolved CSV，应放入同一个带时间和 commit 的子目录，而不是直接散放在 `eval_results/` 根目录。

注意：当前 `LATEST` 指向的子目录只汇总了 4 个 GMM 报告，且记录的 commit 与当前工作区 HEAD 不一致。查看当前精度时，应重新生成完整快照，不要把历史裸 CSV 或旧 `LATEST` 当作当前结论。

## 当前精度快照

以下结果来自清理旧裸 CSV 前对历史报告的重新汇总，过滤口径同上，仅保留为误差背景。`large max/p95/median` 是 large occupied 集合上的相对误差；GMM 报告使用 routing-bound 区间误差。若代码或报告重新生成，应以新的 `<UTC时间>_<commit>/eval_summary.csv` 为准。

| 报告 | rows | large | large max | p95 | median | 主要 tail |
|---|---:|---:|---:|---:|---:|---|
| `matmul_eval_report_910b4` | 1463 | 1161 | 19.088 | 6.215 | 0.100 | 普通单点口径混入 `GroupedMatmul`，不应作为 GMM 最终精度 |
| `matmul_eval_report_910c` | 672 | 588 | 0.542 | 0.240 | 0.055 | `MatMulV2` 小 M、长 K/N decode-like 路径 |
| `attention_eval_report_910b4` | 246 | 139 | 0.259 | 0.208 | 0.089 | `FusedInferAttentionScore` decode |
| `attention_eval_report_910c` | 256 | 160 | 0.116 | 0.099 | 0.025 | 长 prefill `FlashAttentionScore` |
| `profiling_with_model_code_ds32_matmul_eval_910c` | 1060 | 420 | 0.689 | 0.651 | 0.166 | `QuantBatchMatmulV3` 小 M、`TransposeBatchMatMul` |
| `profiling_with_model_code_ds32_attention_eval_910c` | 90 | 90 | 1.840 | 1.821 | 1.719 | `KvQuantSparseFlashAttention`，大量物理下界违反 |
| `profiling_with_model_code_gemma_matmul_eval_910b4` | 888 | 778 | 0.516 | 0.188 | 0.052 | `MatMul` 小 M decode-like 路径 |
| `profiling_with_model_code_gemma_attention_eval_910b4` | 90 | 75 | 0.416 | 0.236 | 0.152 | FIA decode，`head_dim=4096` |
| `profiling_with_model_code_qwen7b_matmul_eval_910b4_inferred` | 483 | 483 | 0.811 | 0.584 | 0.390 | 全量物理下界违反 |
| `profiling_with_model_code_qwen7b_attention_eval_910b4_inferred` | 96 | 96 | 0.468 | 0.459 | 0.416 | decode launch/latency floor 低估 |
| `profiling_with_model_code_longcat_matmul_eval_910b4` | 279 | 87 | 0.542 | 0.507 | 0.127 | GMM above-bound 样本主导 |
| `profiling_with_model_code_longcat_attention_eval_910b4` | 28 | 0 | - | - | - | 没有进入 large occupied 集合 |
| `grouped_matmul_eval_report_910b4` | 208 | 187 | 0.543 | 0.473 | 0.000 | longcat GMM above-bound，gemma 多数 within-bound |
| `profiling_with_model_code_gemma_grouped_matmul_eval_910b4` | 180 | 160 | 0.000 | 0.000 | 0.000 | large 样本全部 within GMM bounds |
| `profiling_with_model_code_longcat_grouped_matmul_eval_910b4` | 28 | 28 | 0.542 | 0.513 | 0.391 | 28/28 above GMM bounds |
| `profiling_with_model_code_ds32_grouped_matmul_eval_910c` | 120 | 120 | 0.176 | 0.165 | 0.090 | 120/120 above GMM bounds |

## 主要结论

- 910C attention 当前最好：large max 约 11.6%，p95 约 9.9%，主要 tail 是长 prefill `FlashAttentionScore`。
- 910B4 attention 的普通 FIA decode 基本可解释，但 gemma/qwen7b decode 小规格仍有 40% 级别误差，主要是模板/latency floor 和 source replay 不完整。
- MatMulV3/普通大形状整体可用；`MatMulV2` 小 M、长 K/N 路径仍是主要 tail，当前多为 `fallback_tiling`。
- `GroupedMatmul` 必须看 routing-bound 区间误差。gemma GMM 当前 large 样本落在区间内；longcat 和 ds3.2 仍有 above-bound，说明缺少真实 groupList、专家调度、同步、merge/atomic 或更具体 kernel 分支信息。
- `KvQuantSparseFlashAttention` 是当前 attention 最大问题，且存在大量 `ideal_lower_bound_us > duration_us`，需要基于专用源码路径建立模型。
- `qwen7b` inferred MatMulV2 报告不能直接用于校准。它的 483 行全部出现物理下界违反，应先核对平台推断、HBM 配置、profiling 字段缺失和 storage/shape 解释。

## 当前 unresolved

- `matmul_eval_unresolved_910b4.csv` / `grouped_matmul_eval_unresolved_910b4.csv`：仍有 128 条 `GroupedMatmul`，典型输入为 `ND;FRACTAL_NZ` 多输入形态。
- `matmul_eval_unresolved_910c.csv`：仍有 256 条 `GroupedMatmul`，典型输入 format 为 `ND;NCL;...`。
- `profiling_with_model_code_matmul_unresolved_910b4.csv`：仍有 90 条 `TransposeBatchMatMul`，输入形态类似 `128,4,512;128,512,128 -> 4,128,128`。

这些 unresolved 应作为 parser/model 覆盖缺口处理，不能计入已解释精度。
