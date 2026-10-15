# Eight Raw iEEG Baselines ± Patient-Relative Normalization — Seed42

## 结论与完成状态

八组配置、40 个训练单元及全部五折 Outer **实际完成**；每组均覆盖 80 人、7,635 个患者通道、256 次发作及 1,471,965 个有效 Raw channel-windows。全部 640 条患者级指标、冻结选择和 10,000 次配对患者 bootstrap 已独立复核 PASS。训练、预测、阈值和原始科学代码均未因结果改变。

本轮没有证据支持“PR 普遍提高 Raw 排序能力”：AUROC/AP 的点估计仅在 SEEG-Net 和 SEEGformer 上提高，TimeConv-CNN、EEGNet 均下降。只有 SEEG-Net 的 AUROC/AP 配对区间同时排除零。八组均未超过同队列 D1/M8 参考。

完成日期：2026-10-16，Asia/Shanghai。科学状态：**EXPLORATORY_REPEATED_TEST**。这是此前研究已查看队列上的预先冻结实验，不是全新盲测。所有模型均为适配版本，不能将本结果冒充原论文完整官方协议的复现。

## 1. Patient-macro 主结果

每位患者先独立计算再等权平均；不是 pooled channel AUROC。EZ/SOZ=1；AUPRC 指 Average Precision。EZ-Frac. Bias 为患者内预测 EZ 比例减真实 EZ 比例，再对患者等权平均；负值表示少报 EZ。各组真实 EZ 比例的患者均值均为 0.238261。

| 模型 | PR | ACC | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | MCC | EZ-Frac. Bias |
|---|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN-adapted | No | 0.667590 | 0.620571 | 0.431539 | 0.491133 | 0.266967 | 0.715298 | 0.194159 | 0.019725 |
| TimeConv-CNN-adapted | Yes | 0.690643 | 0.600367 | 0.386075 | 0.506914 | 0.225040 | 0.788789 | 0.085624 | -0.059929 |
| EEGNet-adapted | No | 0.679370 | 0.560085 | 0.345239 | 0.441691 | 0.125678 | 0.757705 | 0.031206 | -0.061747 |
| EEGNet-adapted | Yes | 0.678999 | 0.547871 | 0.326085 | 0.498119 | 0.211265 | 0.784973 | 0.054484 | -0.054967 |
| SEEG-Net-adapted | No | 0.728502 | 0.450626 | 0.263783 | 0.422239 | 0.030769 | 0.813710 | -0.075213 | -0.176567 |
| SEEG-Net-adapted | Yes | 0.675389 | 0.561613 | 0.341576 | 0.496440 | 0.214483 | 0.778396 | 0.065922 | -0.050137 |
| SEEGformer-adapted | No | 0.625904 | 0.534515 | 0.318833 | 0.424519 | 0.159583 | 0.689454 | 0.015576 | 0.025714 |
| SEEGformer-adapted | Yes | 0.653136 | 0.583621 | 0.337734 | 0.503621 | 0.253645 | 0.753598 | 0.065453 | 0.013436 |

MCC 的可估计患者数随模型变化，不能忽略第 7 节覆盖率而直接比较其均值。

## 2. Pooled channel-level 结果

每组 7,635 个通道合并计算。阈值仍来自各折 inner validation，不能把这个表与上表混为一个评价单位。

| 模型 | PR | ACC | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN-adapted | No | 0.663654 | 0.527132 | 0.256007 | 0.547280 | 0.317747 | 0.776812 | 0.096636 |
| TimeConv-CNN-adapted | Yes | 0.692207 | 0.539151 | 0.276907 | 0.525785 | 0.244859 | 0.806712 | 0.055697 |
| EEGNet-adapted | No | 0.697315 | 0.541994 | 0.250080 | 0.524833 | 0.238550 | 0.811116 | 0.055801 |
| EEGNet-adapted | Yes | 0.681336 | 0.517977 | 0.245570 | 0.513781 | 0.228354 | 0.799208 | 0.030647 |
| SEEG-Net-adapted | No | 0.742764 | 0.478290 | 0.227562 | 0.473313 | 0.096596 | 0.850031 | 0.008933 |
| SEEG-Net-adapted | Yes | 0.674656 | 0.541960 | 0.254610 | 0.508750 | 0.223265 | 0.794235 | 0.019733 |
| SEEGformer-adapted | No | 0.631041 | 0.536804 | 0.245160 | 0.502224 | 0.249000 | 0.755448 | 0.006090 |
| SEEGformer-adapted | Yes | 0.653307 | 0.549726 | 0.258742 | 0.527123 | 0.282850 | 0.771396 | 0.055324 |

## 3. PR 配对增量与不确定性

Δ=PR−Original。固定已冻结的模型与每折阈值，不在 bootstrap 内重新训练、选择模型或优化阈值。80 位患者按 ID 配对重采样 10,000 次，seed42，百分位 95% CI；区间未作多重比较校正，也不包含训练 seed 不确定性。未定义指标沿用配对有效患者，详见 CSV。

| Backbone | ΔACC | ΔAUROC | ΔAP | ΔMacro-F1 | ΔEZ-F1 | ΔMCC |
|---|---|---|---|---|---|---|
| TimeConv-CNN-adapted | 0.023053 | -0.020204 | -0.045464 | 0.015782 | -0.041927 | -0.108536 |
| EEGNet-adapted | -0.000371 | -0.012214 | -0.019153 | 0.056428 | 0.085588 | 0.023277 |
| SEEG-Net-adapted | -0.053112 | 0.110988 | 0.077793 | 0.074200 | 0.183714 | 0.141134 |
| SEEGformer-adapted | 0.027232 | 0.049106 | 0.018901 | 0.079103 | 0.094062 | 0.049877 |

| Backbone | 指标 | Δ | 95% CI 下界 | 95% CI 上界 |
|---|---|---|---|---|
| TimeConv-CNN-adapted | accuracy | 0.023053 | -0.026404 | 0.075201 |
| TimeConv-CNN-adapted | f1 | -0.041927 | -0.099745 | 0.015466 |
| TimeConv-CNN-adapted | macro_f1 | 0.015782 | -0.021771 | 0.052186 |
| TimeConv-CNN-adapted | auroc | -0.020204 | -0.072391 | 0.031709 |
| TimeConv-CNN-adapted | auprc | -0.045464 | -0.092418 | 0.000479 |
| TimeConv-CNN-adapted | mrr | -0.018599 | -0.111790 | 0.074023 |
| EEGNet-adapted | accuracy | -0.000371 | -0.038230 | 0.039078 |
| EEGNet-adapted | f1 | 0.085588 | 0.037413 | 0.133005 |
| EEGNet-adapted | macro_f1 | 0.056428 | 0.030423 | 0.083991 |
| EEGNet-adapted | auroc | -0.012214 | -0.057996 | 0.034048 |
| EEGNet-adapted | auprc | -0.019153 | -0.049510 | 0.011118 |
| EEGNet-adapted | mrr | -0.025271 | -0.105203 | 0.053368 |
| SEEG-Net-adapted | accuracy | -0.053112 | -0.087475 | -0.014995 |
| SEEG-Net-adapted | f1 | 0.183714 | 0.135656 | 0.230384 |
| SEEG-Net-adapted | macro_f1 | 0.074200 | 0.047150 | 0.102663 |
| SEEG-Net-adapted | auroc | 0.110988 | 0.060575 | 0.162356 |
| SEEG-Net-adapted | auprc | 0.077793 | 0.040817 | 0.118007 |
| SEEG-Net-adapted | mrr | 0.162300 | 0.066917 | 0.257528 |
| SEEGformer-adapted | accuracy | 0.027232 | -0.015500 | 0.072703 |
| SEEGformer-adapted | f1 | 0.094062 | 0.041014 | 0.145971 |
| SEEGformer-adapted | macro_f1 | 0.079103 | 0.050682 | 0.108045 |
| SEEGformer-adapted | auroc | 0.049106 | -0.004253 | 0.102173 |
| SEEGformer-adapted | auprc | 0.018901 | -0.016066 | 0.053599 |
| SEEGformer-adapted | mrr | -0.004548 | -0.094899 | 0.083917 |

解读：

- TimeConv-CNN：Macro-F1 小幅增加，但其 CI 跨零；AUROC/AP、EZ-F1 和 MRR 点估计下降。NEZ-F1 增加不能等同于 EZ 定位改善。
- EEGNet：Macro-F1/EZ-F1 增加且区间排除零，但 AUROC/AP 未增加；主要收益体现在冻结阈值下分类而非可靠的排序提升。
- SEEG-Net：PR 同时提高 AUROC、AP、Macro-F1、EZ-F1、MRR，所列增量区间均排除零；ACC 反而下降。Original 的患者平均敏感度仅 0.049840、特异度 0.935378，EZ-Frac. Bias 为 −0.176567，较高 ACC 主要伴随大量少报 EZ，不能据 ACC 宣称更好的定位。
- SEEGformer：Macro-F1/EZ-F1 增加且区间排除零；AUROC/AP 增量 CI 跨零，MRR 点估计略降、Hit@1 不变。分类收益并未一致转化为患者内前列检索收益。

## 4. 完整扩展分类与概率指标（patient-macro）

下表未显示的 NEZ-F1、ACC、AUROC/AP、MCC 和比例偏差见主表。ECE 使用 10 个等宽概率 bin；概率没有 Outer 标签校准。所有均值排除不可定义值，并保留覆盖率。

| 模型 | 版本 | Precision | Recall | Specificity | Weighted-F1 | Macro-P | Macro-R | Kappa | NPV | FPR | FNR | Brier | Log loss | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN-adapted | Original | 0.423798 | 0.362453 | 0.766740 | 0.628165 | 0.625929 | 0.564596 | 0.132697 | 0.792379 | 0.233260 | 0.637547 | 0.376366 | 1.149805 | 0.427118 |
| TimeConv-CNN-adapted | PR | 0.353754 | 0.231199 | 0.835408 | 0.679735 | 0.571772 | 0.533303 | 0.066883 | 0.770995 | 0.164592 | 0.768801 | 0.288889 | 0.907103 | 0.333131 |
| EEGNet-adapted | Original | 0.305780 | 0.188472 | 0.826700 | 0.632595 | 0.616150 | 0.507586 | 0.013940 | 0.759510 | 0.173300 | 0.811528 | 0.248983 | 0.692119 | 0.282528 |
| EEGNet-adapted | PR | 0.279019 | 0.240353 | 0.827156 | 0.671570 | 0.525134 | 0.533755 | 0.043890 | 0.771249 | 0.172844 | 0.759647 | 0.258410 | 0.714851 | 0.310610 |
| SEEG-Net-adapted | Original | 0.205305 | 0.049840 | 0.935378 | 0.649384 | 0.694229 | 0.492609 | -0.011370 | 0.754270 | 0.064622 | 0.950160 | 0.308812 | 0.813562 | 0.358896 |
| SEEG-Net-adapted | PR | 0.285758 | 0.250972 | 0.820129 | 0.662987 | 0.530313 | 0.535551 | 0.060368 | 0.771440 | 0.179871 | 0.749028 | 0.256325 | 0.712479 | 0.300088 |
| SEEGformer-adapted | Original | 0.231280 | 0.273783 | 0.734232 | 0.586262 | 0.550144 | 0.504008 | 0.011199 | 0.763006 | 0.265768 | 0.726217 | 0.225459 | 0.649892 | 0.230111 |
| SEEGformer-adapted | PR | 0.271459 | 0.342255 | 0.758389 | 0.662365 | 0.523106 | 0.550322 | 0.050541 | 0.774754 | 0.241611 | 0.657745 | 0.251689 | 0.697505 | 0.300106 |

TP/TN/FP/FN 的整体及折/中心统计见 [confusion_matrices.csv](results/confusion_matrices.csv)、[fold_metrics.csv](results/fold_metrics.csv)、[center_metrics.csv](results/center_metrics.csv)。不计算或报告禁止的辅助指标。

## 5. 患者内排名

按连续 EZ 分数排序，不使用二值预测；相同分数以规范化通道名确定稳定顺序。排名先患者内计算，再患者等权平均。

| 模型 | 版本 | MRR | Hit@1 | Hit@3 | Hit@5 | Precision@3 | Recall@3 | NDCG@5 |
|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN-adapted | Original | 0.570050 | 0.437500 | 0.650000 | 0.725000 | 0.462500 | 0.098280 | 0.447805 |
| TimeConv-CNN-adapted | PR | 0.551452 | 0.412500 | 0.650000 | 0.737500 | 0.379167 | 0.081175 | 0.391397 |
| EEGNet-adapted | Original | 0.475009 | 0.300000 | 0.562500 | 0.712500 | 0.329167 | 0.075835 | 0.335046 |
| EEGNet-adapted | PR | 0.449738 | 0.262500 | 0.562500 | 0.725000 | 0.304167 | 0.063355 | 0.311934 |
| SEEG-Net-adapted | Original | 0.341000 | 0.225000 | 0.362500 | 0.437500 | 0.208333 | 0.023587 | 0.215250 |
| SEEG-Net-adapted | PR | 0.503300 | 0.325000 | 0.637500 | 0.750000 | 0.337500 | 0.074399 | 0.326299 |
| SEEGformer-adapted | Original | 0.443752 | 0.262500 | 0.562500 | 0.675000 | 0.270833 | 0.057147 | 0.277172 |
| SEEGformer-adapted | PR | 0.439205 | 0.262500 | 0.537500 | 0.662500 | 0.300000 | 0.070329 | 0.290994 |

## 6. 五折均值 ± 标准差

以下对每折 patient-macro 指标再取五折等权均值和样本 SD（ddof=1），因此不必等于 80 人加权后的总 patient-macro。单折患者数分别 16/16/17/15/16，通道数 1631/1498/1746/1425/1335。完整 patient-macro 与 pooled 的每折和 mean±SD 都在 CSV 中。

| 模型 | 版本 | ACC | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---|---|---|---|---|---|
| TimeConv-CNN-adapted | Original | 0.665727 ± 0.078616 | 0.618875 ± 0.123672 | 0.430866 ± 0.127766 | 0.490158 ± 0.080749 | 0.267365 ± 0.145358 |
| TimeConv-CNN-adapted | PR | 0.688647 ± 0.100229 | 0.597500 ± 0.092794 | 0.384835 ± 0.075843 | 0.506408 ± 0.029749 | 0.225695 ± 0.043447 |
| EEGNet-adapted | Original | 0.677435 ± 0.100042 | 0.560675 ± 0.058258 | 0.346898 ± 0.048997 | 0.441245 ± 0.023651 | 0.126363 ± 0.106248 |
| EEGNet-adapted | PR | 0.678847 ± 0.055246 | 0.548766 ± 0.047436 | 0.328165 ± 0.076770 | 0.498231 ± 0.017410 | 0.211689 ± 0.035990 |
| SEEG-Net-adapted | Original | 0.726969 ± 0.048853 | 0.451011 ± 0.011429 | 0.265502 ± 0.072997 | 0.422005 ± 0.013824 | 0.031811 ± 0.034981 |
| SEEG-Net-adapted | PR | 0.674880 ± 0.074984 | 0.562172 ± 0.039520 | 0.343178 ± 0.049037 | 0.495885 ± 0.031912 | 0.213774 ± 0.062361 |
| SEEGformer-adapted | Original | 0.622772 ± 0.130982 | 0.533666 ± 0.053171 | 0.319452 ± 0.067749 | 0.423594 ± 0.039059 | 0.161948 ± 0.102879 |
| SEEGformer-adapted | PR | 0.652652 ± 0.074246 | 0.582911 ± 0.021625 | 0.338742 ± 0.063476 | 0.503032 ± 0.023561 | 0.252942 ± 0.077823 |

### 中心汇总（patient-macro）

中心仅作预定描述分析，不按中心重新训练、选择阈值或排除患者。

| 模型 | 版本 | 中心 | 人数 | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---|---|---|---|---|---|---|
| TimeConv-CNN-adapted | Original | hup | 36 | 0.658067 | 0.438661 | 0.534001 | 0.260461 |
| TimeConv-CNN-adapted | Original | lzu | 21 | 0.597579 | 0.496230 | 0.477134 | 0.318003 |
| TimeConv-CNN-adapted | Original | multicenter | 15 | 0.635819 | 0.420677 | 0.489871 | 0.222422 |
| TimeConv-CNN-adapted | Original | pediatric | 8 | 0.483600 | 0.250036 | 0.337338 | 0.245799 |
| TimeConv-CNN-adapted | PR | hup | 36 | 0.630167 | 0.358193 | 0.530504 | 0.245438 |
| TimeConv-CNN-adapted | PR | lzu | 21 | 0.548854 | 0.469100 | 0.472916 | 0.241562 |
| TimeConv-CNN-adapted | PR | multicenter | 15 | 0.665682 | 0.412160 | 0.516359 | 0.190710 |
| TimeConv-CNN-adapted | PR | pediatric | 8 | 0.479019 | 0.244691 | 0.472299 | 0.154251 |
| EEGNet-adapted | Original | hup | 36 | 0.532050 | 0.264942 | 0.432924 | 0.057668 |
| EEGNet-adapted | Original | lzu | 21 | 0.508016 | 0.441383 | 0.417493 | 0.220340 |
| EEGNet-adapted | Original | multicenter | 15 | 0.670085 | 0.411063 | 0.481196 | 0.150688 |
| EEGNet-adapted | Original | pediatric | 8 | 0.616672 | 0.330775 | 0.470595 | 0.136339 |
| EEGNet-adapted | PR | hup | 36 | 0.551964 | 0.263318 | 0.513954 | 0.220175 |
| EEGNet-adapted | PR | lzu | 21 | 0.512491 | 0.441161 | 0.447867 | 0.200506 |
| EEGNet-adapted | PR | multicenter | 15 | 0.639639 | 0.346613 | 0.523980 | 0.221355 |
| EEGNet-adapted | PR | pediatric | 8 | 0.450264 | 0.267973 | 0.510288 | 0.180497 |
| SEEG-Net-adapted | Original | hup | 36 | 0.438612 | 0.208172 | 0.440111 | 0.015253 |
| SEEG-Net-adapted | Original | lzu | 21 | 0.483126 | 0.422949 | 0.378077 | 0.055879 |
| SEEG-Net-adapted | Original | multicenter | 15 | 0.398393 | 0.180585 | 0.428005 | 0.045560 |
| SEEG-Net-adapted | Original | pediatric | 8 | 0.517312 | 0.252213 | 0.446934 | 0.006944 |
| SEEG-Net-adapted | PR | hup | 36 | 0.558257 | 0.279949 | 0.508875 | 0.210792 |
| SEEG-Net-adapted | PR | lzu | 21 | 0.552826 | 0.460798 | 0.441960 | 0.191115 |
| SEEG-Net-adapted | PR | multicenter | 15 | 0.626536 | 0.367979 | 0.551950 | 0.292061 |
| SEEG-Net-adapted | PR | pediatric | 8 | 0.478054 | 0.256436 | 0.479409 | 0.146978 |
| SEEGformer-adapted | Original | hup | 36 | 0.545648 | 0.275143 | 0.444153 | 0.114040 |
| SEEGformer-adapted | Original | lzu | 21 | 0.507748 | 0.444166 | 0.410567 | 0.230595 |
| SEEGformer-adapted | Original | multicenter | 15 | 0.549082 | 0.290199 | 0.474024 | 0.127857 |
| SEEGformer-adapted | Original | pediatric | 8 | 0.527370 | 0.240128 | 0.279964 | 0.237608 |
| SEEGformer-adapted | PR | hup | 36 | 0.578159 | 0.266771 | 0.502541 | 0.220559 |
| SEEGformer-adapted | PR | lzu | 21 | 0.553015 | 0.463429 | 0.479901 | 0.315771 |
| SEEGformer-adapted | PR | multicenter | 15 | 0.631625 | 0.339233 | 0.538357 | 0.261115 |
| SEEGformer-adapted | PR | pediatric | 8 | 0.598529 | 0.324305 | 0.505620 | 0.225441 |

## 7. 可估计性与覆盖

全部八组均覆盖 80 人/7,635 通道；AUROC、AP、Macro-F1、EZ-F1、NEZ-F1、ACC 和排名指标各有 80 位有效患者。以下为存在未定义值的指标，所有未列指标均为 80/80。

| 模型 | 版本 | 指标 | 有效人数 | 未定义人数 |
|---|---|---|---|---|
| TimeConv-CNN-adapted | Original | precision | 64 | 16 |
| TimeConv-CNN-adapted | Original | mcc | 59 | 21 |
| TimeConv-CNN-adapted | Original | npv | 75 | 5 |
| TimeConv-CNN-adapted | PR | precision | 77 | 3 |
| TimeConv-CNN-adapted | PR | mcc | 77 | 3 |
| EEGNet-adapted | Original | precision | 51 | 29 |
| EEGNet-adapted | Original | mcc | 51 | 29 |
| SEEG-Net-adapted | Original | precision | 18 | 62 |
| SEEG-Net-adapted | Original | mcc | 17 | 63 |
| SEEG-Net-adapted | Original | npv | 79 | 1 |
| SEEG-Net-adapted | PR | precision | 79 | 1 |
| SEEG-Net-adapted | PR | mcc | 79 | 1 |
| SEEGformer-adapted | Original | precision | 60 | 20 |
| SEEGformer-adapted | Original | mcc | 59 | 21 |
| SEEGformer-adapted | Original | npv | 79 | 1 |

没有预测阳性时 EZ precision 未定义；常量预测使 MCC 分母为零时 MCC 未定义。未将这些情况填成 0 或 0.5。Macro-Precision 对可定义分量求均值，可能掩盖没有预测 EZ 的情况，应联合 coverage、EZ recall、confusion 和 EZ-Frac. Bias 解读。

## 8. 与冻结 D1/M8 参考比较

| 模型 | ACC | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | MRR | Hit@1 | EZ-Frac. Bias |
|---|---|---|---|---|---|---|---|---|---|
| D1 | 0.748950 | 0.732636 | 0.527175 | 0.616556 | 0.408531 | 0.824582 | 0.689399 | 0.575000 | -0.027451 |
| M8 | 0.760166 | 0.762397 | 0.554067 | 0.634097 | 0.434087 | 0.834107 | 0.749310 | 0.637500 | -0.029065 |

已核对相同 80 人、7,635 通道、Outer 五折与标签方向，并用既有冻结结果/独立验证而非重新评估参考模型。参考冻结 SHA256 `4882d62e0492618a277897e56e434578b8481b7a8192213d01cd3a5b5f420d5d`，预测封存 SHA256 `aa5ae8daeec58fb392e6587a89f24cb26ce34fc83a1125f8f6ae4a575ab81e02`。

最强 Raw 排序为 TimeConv-CNN Original：AUROC 0.620571、AP 0.431539。最强 Raw+PR 的 AUROC/AP 仍是 TimeConv-CNN+PR：0.600367/0.386075；它也是本轮最高 Macro-F1（0.506914）。两者均低于 D1 和 M8 的点估计，本轮没有新做 Raw 对 D1/M8 的差值显著性检验。

这不是严格只改变 backbone 的同输入消融：D1 使用生理 DELTA，M8 另有 14D 生理 DELTA 基础、7D guidance、pre-reference latent subtraction、早/晚 Raw 池化、残差结构与不同目标。本轮八组只共享 Raw ABS 可见范围与统一完整窗口聚合。不能单凭本轮差值证明某一个 M8 模块导致优势，也不能用参考结果反向调参。

## 9. 来源、输入、PR 和训练协议

实现细节和源文件完整哈希见 [README](README.md)、[SOURCE_AUDIT](configs/SOURCE_AUDIT.json)、[PROTOCOL_LOCK](configs/PROTOCOL_LOCK.json)。

- TimeConv-CNN-adapted：逐字节复用归档 OmniTimeConvCNNEncoder，STFT→224 插值频率 bins→TimeConv→完整 ResNet18→128D。不是 60s Morlet 官方 Omni CNN；scratch 初始化，未复现或重训另一套 CNN baseline。
- EEGNet-adapted：EEGNet-8,2 核心 temporal/depthwise/separable convolution，单通道输入使空间卷积退化为 depthwise 1×1；核125/31、池化4/8、dropout0.5，保留 max-norm。没有人为固定电极数或增加注意力。
- SEEG-Net-adapted：归档多尺度 Conv1D、SE/残差、BiLSTM、时间注意力；500 samples、250 Hz，保留完整结构。
- SEEGformer-adapted：归档 FFT real/imag/amplitude 三分支、masked 跨通道 Transformer，每支2层4头64D；FFT1000、0.5–125 Hz，无 MRI/虚构坐标。保留三分类头和分支加权；为通道均值/SD 聚合调整 head 输入维度。不是原论文完整协议。
- Raw：现存 hash-bound 250Hz、2s/500samples，约 −29 至 +29s 的全部有效窗口；沿用无效填充/缺失通道 mask，未新抽取、采样窗口或删患者。250Hz Nyquist=125Hz，不虚构高于125Hz的信息。
- 八组相同 channel-local 窗口均值/populationSD 标准化（floor1e-6、clip±8），这不是患者跨通道 PR。每次发作/通道先对所有有效窗口 embedding 求均值，再跨可用发作计算 mean/populationSD 并交错拼接，保持可微。
- PR 在最终 channel embedding 与相同 head 之间：当前患者全部有效通道、逐维 float64 mean/populationSD；SD>1e-8 用 SD，否则分母1，非 additive-epsilon。无患者 ID、标签或跨患者共同统计；推理需无标签全患者通道上下文。
- Original 与 PR 按折独立训练；fold seed=42+1009×fold，完整初始状态 hash 完全相同；未将 PR 事后插入 Original logits。
- AdamW、wd0.0001、gradient clip1、患者等权 weightedBCE；FIT-only NEZ class weight，内部 NEZ=1，报告 yEZ=1−yNEZ、pEZ=1−sigmoid(NEZ logit)。lr：TimeConv0.0003、EEGNet0.001、另两者0.0005。max30/min6/patience6。
- 每折 FIT/validation/Outer 患者严格互斥。validation patient Macro-F1→EZ-F1→更早 epoch 选择 checkpoint；EZ 阈值 grid0:0.005:1，以 validation patient Macro-F1→EZ-F1→接近0.5→较低阈值选择。没有 Outer 校准或调参。
- BF16 训练、FP32 validation/Outer、TF32关闭、cuDNN保留。固定 chunk128/1024/256/16，Exact two-pass gradient cache 保留全窗口及完整聚合梯度、RNG 和 BN状态，不是冻结 encoder 或截断梯度。

模型参数：TimeConv 11,336,737；EEGNet 2,329；SEEG-Net 279,950；SEEGformer 310,278。PR 不增加参数。40 个实际选择的 epoch/阈值/参数/初始化哈希见 [model_parameters.csv](results/model_parameters.csv)。

## 10. 工程经过、独立复算和限制

正式训练前四模型 CUDA forward/backward、全患者 PR、mask/permutation、同初始化和 direct/cache 梯度/BN、FP32 chunk 推理容差均 PASS。此前 BF16 推理的数值漂移在正式训练前由无结果 synthetic admission 发现，因此 validation/Outer 固定 FP32；没有丢弃已完成正式 epoch。

用户要求加速后，以完成 epoch 的 atomic LAST 作切换并验证整文件备份哈希，原 source/protocol/input binding 不变。外部 wrapper 只延后冗余 finite-check host 同步；实际完整 FIT 的 logits/loss/所有梯度/BN/RNG bit-identical（最大梯度差0），四模型 admission 再次 PASS。双线并行执行不同 backbone，无重复训练/测试。实际 FIT 交替计时均值 1.6077s→1.3211s，不能推算为整个实验同比加速。见 [加速审计](../../tools/m8_raw_backbones_pr_acceleration_v1/README.md)。

所有科学训练与 Outer 已于约 01:04 完成。原 final verifier 在 log loss 断言失败：保存概率为 float32，sklearn 采用对应精度的裁剪，而冻结指标 engine 明确先转 float64 并按1e-15裁剪。外部 `verify_reporting_v2.py` 只在内存中将 verifier 的概率读入改成 float64，保留原脚本字节、原指标与1e-12容差，不重训、不重预测、不改阈值。旧最大差0.1690653713；修正后与原 engine 的640条 log loss 最大差为0。57个受保护预测、指标、冻结/封存和最佳权重文件哈希前后未变。原失败日志留服务器。共享 RUN_STATUS 只是某一执行线最后写入的训练快照，不是最终终态；以四个 OUTER_STATUS 和 INDEPENDENT_VERIFICATION/REPORTING_VERIFIER_AUDIT 为准。

独立复核：8组×80人=640条患者指标、40个已冻结选择、每组7,635通道、相同初始状态、validation-only阈值、原始封存/权重哈希、所有公开表与10,000次配对 bootstrap均PASS；bootstrap最大误差1.39e-16，未放宽1e-12容差。

限制：只有seed42；固定适配模型、scratch和统一patient-level目标，不等于各原论文的最优配置。低表现不能单独归因于原模型、PR或某一训练因素；本轮没有做允许之外的调参或额外模型。SEEG-Net Original 的排序低于0.5已核查标签方向并完整保留，不事后翻转其分数。全部训练选择与评价是 patient-held-out；此前队列被查看，因此最终仍是 exploratory repeated-test。

## 11. 五个问题的直接回答

1. **最强 Raw backbone？** 本轮 AUROC/AP/MRR/Hit@1 最好的是 TimeConv-CNN Original；Macro-F1 最好的是 TimeConv-CNN+PR。不存在单个配置在全部指标上最好。
2. **PR 跨架构一致改善 AUROC/AP？** 否。两种增加、两种降低；仅 SEEG-Net 的两项配对区间均排除零。
3. **收益主要体现在哪里？** 四种 Macro-F1 点估计均增加，但 TimeConv 区间跨零；EEGNet/SEEGformer 未显示可靠一致的排序收益。SEEG-Net 有排序与分类的共同改善，同时ACC下降。
4. **最好 Raw+PR 超过 D1？** 否，AUROC/AP/Macro-F1/EZ-F1 均低于同队列 D1 点估计。
5. **M8 保留优势？** 是，所列主要指标点估计仍高于本轮最强独立 Raw 或 Raw+PR，但输入生理上下文、结构与目标不同，不能解释为严格单因素因果结论。

## 12. 交付与隐私

[main_results](results/main_results.csv)、[pr_improvement](results/pr_improvement.csv)、[extended_metrics](results/extended_metrics.csv)、[pooled_metrics](results/pooled_metrics.csv)、[fold_metrics](results/fold_metrics.csv)、[fold_mean_std](results/fold_mean_std.csv)、[center_metrics](results/center_metrics.csv)、[ranking_metrics](results/ranking_metrics.csv)、[metric_coverage](results/metric_coverage.csv)、[paired_pr_intervals](results/paired_pr_intervals.csv)、[INDEPENDENT_VERIFICATION](results/INDEPENDENT_VERIFICATION.json)、[REPORTING_VERIFIER_AUDIT](results/REPORTING_VERIFIER_AUDIT.json)。

逐患者/通道预测、个人指标、embedding、原波形、缓存、权重、训练日志、患者/通道身份和可信服务器路径只留服务器；公开包不包含这些文件。匿名折/中心汇总及完整公开文件 SHA256 可独立核对；拥有可信服务器权限者可从封存预测运行复核，而非以公开个人数据换取可复现性。
