# PR 排序、患者稳健性与有限通道审阅：主表一致版 A–C

本版 A、B、C 已完成，覆盖全部 **80 人、7,635 通道、256 次发作、四个来源**。TimeConv-CNN 使用与主表相同的 **PR Linear Residual**：原 ictal 编码器和分类器冻结，患者相对表征通过已训练的线性残差头修正 EZ logit。SVM 与 DREChanNet 的 No PR、Full PR、Score-Z，以及 TimeConv 的 No PR、Score-Z，均沿用原封存版本。

本轮新增训练为 **0**。五个 TimeConv 残差头在 80 名 outer 患者的已有表征上重放，概率最大绝对误差 **0**；TimeConv Original 和 PR Linear Residual 的全部 11 项患者等权指标与主表原结果一致。

## A：患者内 EZ 排序

|Model / condition|AUROC|AP|MRR|Hit@1|Hit@3|Hit@5|Macro-F1|EZ-F1|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Linear SVM / No PR|68.65|48.16|64.83|53.75|71.25|80.00|46.06|28.73|
|Linear SVM / Full PR|74.36|54.32|73.88|65.00|80.00|85.00|62.47|45.21|
|Linear SVM / Score-Z|68.65|48.16|64.83|53.75|71.25|80.00|57.93|35.05|
|TimeConv-CNN / No PR|62.06|43.15|57.01|43.75|65.00|72.50|49.11|26.70|
|TimeConv-CNN / PR Linear Residual|64.68|44.58|60.69|45.00|71.25|77.50|54.87|35.91|
|TimeConv-CNN / Score-Z|62.06|43.15|57.01|43.75|65.00|72.50|54.12|31.97|
|DREChanNet / No PR|76.09|55.94|74.83|65.00|80.00|86.25|50.23|37.44|
|DREChanNet / Full PR|76.60|56.86|77.93|67.50|87.50|90.00|64.41|44.85|
|DREChanNet / Score-Z|76.09|55.94|74.83|65.00|80.00|86.25|63.08|42.99|

TimeConv 的 Macro-F1 从 **49.11% 升至 54.87%**，AUROC 从 **62.06% 升至 64.68%**，AP 从 **43.15% 升至 44.58%**。PR 相对 No PR 的 AUROC、AP 平均增益为正，但其配对 95% 区间均跨零。MRR、Hit@3 和 Hit@5 的均值提高，Hit@1 从 43.75% 升至 45.00%。

三种模型的 Score-Z 与各自 No PR 的 AUROC、AP、MRR、Hit@1/3/5 在每名患者上完全相同。Score-Z 可改变分类阈值下的结果，不能改变通道排序。

## B：患者与来源稳健性

### 患者配对增益

正、零、负列为对应患者比例。区间单位为百分点；12 项家族是三模型 × AUROC/AP/Macro-F1/EZ-F1，独立于下方六项 Score-Z 家族。

|Model / metric|Gain [95% CI]|Gain [12-test CI]|Positive|Zero|Negative|
|---|---:|---:|---:|---:|---:|
|Linear SVM / AUROC|+5.70 [+1.84, +9.67]|+5.70 [-0.19, +11.70]|66.25|2.50|31.25|
|Linear SVM / AP|+6.16 [+3.20, +9.22]|+6.16 [+1.84, +10.63]|67.50|1.25|31.25|
|Linear SVM / Macro-F1|+16.40 [+12.29, +20.50]|+16.40 [+10.14, +22.14]|83.75|0.00|16.25|
|Linear SVM / EZ-F1|+16.48 [+11.10, +22.26]|+16.48 [+8.72, +24.69]|70.00|2.50|27.50|
|TimeConv-CNN / AUROC|+2.63 [-0.73, +6.28]|+2.63 [-2.00, +8.15]|53.75|2.50|43.75|
|TimeConv-CNN / AP|+1.42 [-0.79, +3.70]|+1.42 [-1.77, +4.76]|51.25|2.50|46.25|
|TimeConv-CNN / Macro-F1|+5.76 [+3.55, +8.08]|+5.76 [+2.66, +9.23]|63.75|8.75|27.50|
|TimeConv-CNN / EZ-F1|+9.21 [+5.62, +13.00]|+9.21 [+4.07, +14.83]|63.75|13.75|22.50|
|DREChanNet / AUROC|+0.51 [-1.88, +2.80]|+0.51 [-2.89, +3.85]|45.00|1.25|53.75|
|DREChanNet / AP|+0.92 [-1.56, +3.36]|+0.92 [-2.63, +4.46]|51.25|0.00|48.75|
|DREChanNet / Macro-F1|+14.17 [+10.29, +18.27]|+14.17 [+8.67, +20.15]|80.00|1.25|18.75|
|DREChanNet / EZ-F1|+7.42 [+3.17, +11.81]|+7.42 [+1.50, +14.23]|65.00|3.75|31.25|

TimeConv 的 Macro-F1 和 EZ-F1 增益通过该 12 项家族调整；AUROC 提高的患者占 **53.75%**，AP 提高的患者占 **51.25%**。SVM 的 AP 和分类增益有调整后区间支持；其 AUROC 的 12 项调整区间跨零。DREChanNet 的分类增益有支持，AUROC/AP 的增益区间跨零。

### 与输出 Score-Z 的比较

|Model / metric|Gain [95% CI]|Gain [6-test CI]|
|---|---:|---:|
|Linear SVM / AUROC|+5.70 [+1.84, +9.67]|+5.70 [+0.26, +11.21]|
|Linear SVM / Macro-F1|+4.54 [+1.66, +7.41]|+4.54 [+0.71, +8.49]|
|TimeConv-CNN / AUROC|+2.63 [-0.73, +6.28]|+2.63 [-1.72, +7.64]|
|TimeConv-CNN / Macro-F1|+0.75 [-1.98, +3.44]|+0.75 [-2.98, +4.28]|
|DREChanNet / AUROC|+0.51 [-1.88, +2.80]|+0.51 [-2.68, +3.61]|
|DREChanNet / Macro-F1|+1.33 [-0.22, +2.94]|+1.33 [-0.72, +3.49]|

TimeConv 相对 Score-Z 的 Macro-F1 差为 **+0.75 pp**，95% CI **[−1.98, 3.44]**；AUROC 差为 **+2.63 pp**，95% CI **[−0.73, 6.28]**。现有结果未支持它在这两项上优于 Score-Z。SVM 的两项差在六项家族调整后仍为正；DREChanNet 的两项差区间跨零。

### 四来源结果

这是同一五折 **patient-held-out outer** 预测按来源分组，并非 LOSO。CM 为四来源等权平均；Worst 为该指标四来源均值的最小值，各指标可来自不同最差来源。HUP/LZU/Multicenter/Pediatric 分别有 36/21/15/8 人。

Macro-F1（%）：

|Model / condition|HUP|LZU|Multi.|Pediatric|CM|Worst|
|---|---:|---:|---:|---:|---:|---:|
|Linear SVM / No PR|49.27|40.22|46.43|46.29|45.55|40.22|
|Linear SVM / Full PR|65.43|56.72|64.62|60.18|61.74|56.72|
|Linear SVM / Score-Z|61.62|49.87|62.72|53.47|56.92|49.87|
|TimeConv-CNN / No PR|53.40|47.71|48.99|33.73|45.96|33.73|
|TimeConv-CNN / PR Linear Residual|59.24|51.74|57.92|37.73|51.66|37.73|
|TimeConv-CNN / Score-Z|58.38|49.28|54.42|47.08|52.29|47.08|
|DREChanNet / No PR|51.25|45.02|55.18|50.06|50.38|45.02|
|DREChanNet / Full PR|67.81|56.49|67.64|63.83|63.94|56.49|
|DREChanNet / Score-Z|66.29|54.26|68.68|61.25|62.62|54.26|

AUROC（%）：

|Model / condition|HUP|LZU|Multi.|Pediatric|CM|Worst|
|---|---:|---:|---:|---:|---:|---:|
|Linear SVM / No PR|72.58|60.00|76.43|59.11|67.03|59.11|
|Linear SVM / Full PR|79.64|64.10|81.42|64.28|72.36|64.10|
|Linear SVM / Score-Z|72.58|60.00|76.43|59.11|67.03|59.11|
|TimeConv-CNN / No PR|65.81|59.76|63.58|48.36|59.38|48.36|
|TimeConv-CNN / PR Linear Residual|68.91|59.18|68.41|53.12|62.40|53.12|
|TimeConv-CNN / Score-Z|65.81|59.76|63.58|48.36|59.38|48.36|
|DREChanNet / No PR|80.86|64.13|84.32|70.56|74.97|64.13|
|DREChanNet / Full PR|80.95|66.13|80.07|78.04|76.30|66.13|
|DREChanNet / Score-Z|80.86|64.13|84.32|70.56|74.97|64.13|

AP（%）：

|Model / condition|HUP|LZU|Multi.|Pediatric|CM|Worst|
|---|---:|---:|---:|---:|---:|---:|
|Linear SVM / No PR|48.49|50.16|49.33|39.22|46.80|39.22|
|Linear SVM / Full PR|55.70|54.07|56.12|45.37|52.82|45.37|
|Linear SVM / Score-Z|48.49|50.16|49.33|39.22|46.80|39.22|
|TimeConv-CNN / No PR|43.87|49.62|42.07|25.00|40.14|25.00|
|TimeConv-CNN / PR Linear Residual|45.32|48.86|47.54|24.42|41.54|24.42|
|TimeConv-CNN / Score-Z|43.87|49.62|42.07|25.00|40.14|25.00|
|DREChanNet / No PR|57.89|53.26|58.27|49.86|54.82|49.86|
|DREChanNet / Full PR|58.98|55.51|55.43|53.53|55.86|53.53|
|DREChanNet / Score-Z|57.89|53.26|58.27|49.86|54.82|49.86|

TimeConv 的 PR 相对 No PR 在四来源上的 Macro-F1 均值均提高；AUROC 在 LZU 由 59.76% 降至 59.18%。PR 的 Macro-F1 CM/Worst 为 **51.66%/37.73%**，低于 Score-Z 的 **52.29%/47.08%**。总体平均提高不代表每个来源或每项指标都提高。

## C：有限通道审阅

每名患者按预测 EZ 分数排序，取前 `max(1, ceil(q × 通道数))` 个通道，再对患者等权平均。EZ recall 衡量已标注 EZ 的覆盖比例；precision 衡量审阅通道中的 EZ 比例。Lift 是该 precision 相对本患者 EZ 比例的比值。

|Model / condition|Budget|EZ recall (%)|Precision (%)|NEZ count|Lift|Channels|
|---|---:|---:|---:|---:|---:|---:|
|Linear SVM / No PR|5%|15.87|49.60|2.73|2.82|5.28|
|Linear SVM / No PR|10%|26.46|47.21|5.54|2.51|10.03|
|Linear SVM / No PR|20%|43.48|41.39|11.80|2.12|19.51|
|Linear SVM / Full PR|5%|18.92|56.63|2.40|3.34|5.28|
|Linear SVM / Full PR|10%|30.07|52.11|5.01|2.85|10.03|
|Linear SVM / Full PR|20%|49.86|46.39|10.85|2.43|19.51|
|TimeConv-CNN / No PR|5%|13.07|43.00|3.15|2.29|5.28|
|TimeConv-CNN / No PR|10%|22.74|41.47|6.19|2.15|10.03|
|TimeConv-CNN / No PR|20%|36.52|36.74|12.84|1.78|19.51|
|TimeConv-CNN / PR Linear Residual|5%|14.37|46.67|2.96|2.55|5.28|
|TimeConv-CNN / PR Linear Residual|10%|25.89|44.50|5.81|2.46|10.03|
|TimeConv-CNN / PR Linear Residual|20%|39.61|37.87|12.55|1.94|19.51|
|DREChanNet / No PR|5%|19.39|59.19|2.33|3.44|5.28|
|DREChanNet / No PR|10%|31.89|53.97|4.89|3.02|10.03|
|DREChanNet / No PR|20%|50.52|46.83|10.88|2.47|19.51|
|DREChanNet / Full PR|5%|21.03|62.60|2.10|3.75|5.28|
|DREChanNet / Full PR|10%|33.55|56.88|4.55|3.18|10.03|
|DREChanNet / Full PR|20%|52.04|49.32|10.41|2.54|19.51|

配对 EZ recall 增益（pp）：

|Model|Budget|Gain [95% CI]|Gain [9-test CI]|
|---|---:|---:|---:|
|Linear SVM|5%|+3.05 [+1.06, +5.22]|+3.05 [+0.34, +6.13]|
|Linear SVM|10%|+3.61 [+0.60, +6.62]|+3.61 [-0.76, +8.03]|
|Linear SVM|20%|+6.38 [+1.80, +11.16]|+6.38 [-0.18, +13.36]|
|TimeConv-CNN|5%|+1.30 [-0.75, +3.27]|+1.30 [-1.74, +4.06]|
|TimeConv-CNN|10%|+3.14 [+0.97, +5.47]|+3.14 [+0.17, +6.40]|
|TimeConv-CNN|20%|+3.09 [+0.09, +6.35]|+3.09 [-0.89, +7.77]|
|DREChanNet|5%|+1.64 [-0.61, +3.94]|+1.64 [-1.42, +5.12]|
|DREChanNet|10%|+1.65 [-1.39, +4.69]|+1.65 [-2.59, +5.88]|
|DREChanNet|20%|+1.52 [-1.96, +5.04]|+1.52 [-3.44, +6.46]|

TimeConv 在审阅前 10% 通道时，EZ recall 从 **22.74% 升至 25.89%**，配对增益 **+3.14 pp**；95% CI **[0.97, 5.47]**，九项家族调整区间 **[0.17, 6.40]**。其 5% 和 20% 预算差的调整区间均跨零。SVM 的 5% 预算差通过九项调整；DREChanNet 三个预算的差区间均跨零。

固定 1/3/5 个通道（小于预算时取患者实际通道数）：

|Model / condition|Budget|EZ recall (%)|Precision (%)|Hit (%)|Actual channels|
|---|---:|---:|---:|---:|---:|
|Linear SVM / No PR|1|3.83|53.75|53.75|1.00|
|Linear SVM / No PR|3|10.69|51.67|71.25|3.00|
|Linear SVM / No PR|5|16.83|49.75|80.00|5.00|
|Linear SVM / Full PR|1|4.84|65.00|65.00|1.00|
|Linear SVM / Full PR|3|12.96|59.58|80.00|3.00|
|Linear SVM / Full PR|5|19.70|56.50|85.00|5.00|
|TimeConv-CNN / No PR|1|3.51|43.75|43.75|1.00|
|TimeConv-CNN / No PR|3|9.83|46.25|65.00|3.00|
|TimeConv-CNN / No PR|5|14.06|43.50|72.50|5.00|
|TimeConv-CNN / PR Linear Residual|1|3.41|45.00|45.00|1.00|
|TimeConv-CNN / PR Linear Residual|3|9.79|47.50|71.25|3.00|
|TimeConv-CNN / PR Linear Residual|5|15.18|46.75|77.50|5.00|
|DREChanNet / No PR|1|4.88|65.00|65.00|1.00|
|DREChanNet / No PR|3|13.66|62.08|80.00|3.00|
|DREChanNet / No PR|5|20.47|58.50|86.25|5.00|
|DREChanNet / Full PR|1|5.14|67.50|67.50|1.00|
|DREChanNet / Full PR|3|14.25|64.58|87.50|3.00|
|DREChanNet / Full PR|5|21.98|63.00|90.00|5.00|

这些结果评价对既有临床 EZ 标签的覆盖效率，不等同于手术结局预测。

## 图和完整数据

- [患者 AUROC/AP 增益分布](figures/Figure_A_patient_gain_distributions.pdf)：按来源着色，轮廓由匿名 5 pp 分箱平滑；点和误差线是未分箱的实际来源均值及名义 95% 患者 bootstrap CI。不是逐患者散点重建。
- [EZ 覆盖曲线](figures/Figure_B_EZ_coverage.pdf)：三模型各 No PR/PR，实线为 PR、虚线为 No PR；TimeConv 的 PR 为 Linear Residual，其他两模型为原 Full PR。阴影为 PR 的名义 95% CI，右图放大 0–20% 审阅预算。两图同时提供可编辑 SVG 和 600 dpi PNG。
- [匿名全量 CSV](results/public/)：所有来源、11 项患者指标、所有固定/比例预算、分箱、名义与调整区间完整保留。患者指标与 480 行配对差值仅保存在可信服务器。

## 统计、版本与复核

沿用原五折 FIT/VAL/outer 角色、已选择的检查点与验证阈值。每名患者先计算指标，再对 80 人等权平均；没有新增拟合、预测集成或有利患者子集。使用固定 10,000 次来源内配对患者 bootstrap；校正家族分别为 6、12、9 项，均在重算前锁定。全部分析属于已查看队列上的探索性补充。

冻结源码提交：`e631d60f801bfda5ca46e59f8fd85d3fd7c39f55`。服务器完成状态为 `ABC_PR_LINEAR_RESIDUAL_COMPLETE_VERIFIED`；独立复核 720 行患者指标、4,320 行患者预算、480 行患者覆盖曲线，并以另一种直接索引实现重算 **1,203 个区间**。40 个旧单元对应的 640 行患者指标完全一致。

匿名包 SHA256：`f26fab5a7b04cc5e680abb52ef4ac43dc4eff5618043ea6748b32f62e4d62aa5`。本地先核验全包 SHA 和逐文件清单，再白名单提取；原始数据、身份、个体预测、权重和缓存未导出。

此前端到端 TimeConv 的 [A–D 实验](../drechan_pr_ranking_robustness_v1/report.md) 保留。该旧版的少发作实验 D 没有改称 PR Linear Residual，也没有被本版 A–C 替换。JBHI 现有主表和图表未修改。
