# CASE-A01：DREChanNet seed42

**HUP，Outer Fold 4，49 个通道：10 EZ、39 NEZ。**

以下均为冻结原验证阈值下的 WO_BOTH_PR → FULL。原完整 DREChanNet 含 early FiLM / late Gate（11,425 参数）。没有重新训练、推理或调整阈值。

|指标|No PR|Full PR|
|---|---:|---:|
|TP：检出的 EZ|2|5|
|FP：误报的 NEZ|0|1|
|FN：漏检的 EZ|8|5|
|TN：正确排除的 NEZ|39|38|
|AUROC|0.7385|0.7974|
|Macro-F1|0.6202|0.7759|
|Top-1|100.00%|100.00%|
|Recall@10%|50.00%|40.00%|
|Recall@20%|70.00%|70.00%|

精确率：100% → 83.33%；敏感度：20% → 50%。前 10% 审阅预算实际为 5 个通道，前 20% 为 10 个通道。

## 筛选背景

病例按三个 seed（42/43/44）的平均 TP 增益、有限 FP 增量及平均 AUROC 提高筛选；用户在查看三 seed 结果后指定导出 seed42。本包仅展示 seed42。原三 seed 数据完整保留于可信服务器及本机。原病例平均 AUROC 0.7521 → 0.7692；两个 seed 提高、seed44 下降。该病例及 seed 的展示选择是回顾性的，不能作为独立确认或替代总体结果。

## 文件

- [全部 49 通道的标签、实际分数、预测与排名](DRECHANNET_SEED42_CHANNELS_WIDE.csv)。
- [10 个 EZ 通道](DRECHANNET_SEED42_EZ_CHANNELS_ONLY.csv)。
- [TP/FP/FN/TN、精确率等](CASE_CONFUSION.csv)。
- [全部指标与原筛选分位](CASE_METRICS.csv)。
- [逐个整数 K 的 Recall 曲线](RECALL_AT_K.csv)。

真实患者身份与原始通道名称未公开。通道别名 CH0001 等保留原同分排序顺序，不代表解剖位置。独立核验覆盖全部指标、混淆矩阵、分数排序、每个 K 和宽表一致性。
