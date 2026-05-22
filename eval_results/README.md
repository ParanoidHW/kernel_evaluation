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
