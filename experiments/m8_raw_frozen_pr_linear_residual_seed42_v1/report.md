# Frozen Encoder/Classifier + PR Linear Residual — Seed42

## 完成与主要结论

**四个 Original backbone × 五折 = 20 个线性残差头全部完成**，并完成八组（Original/PRLinearResidual）完整 Outer 配对评估。每组 80 位患者、7,635 通道；640 条患者级指标、20 个验证选择及10,000次患者配对bootstrap独立复核PASS。终态：`FROZEN_PR_LINEAR_RESIDUAL_ALL_FIVE_OUTER_COMPLETE`。完成日期：2026-10-16，Asia/Shanghai。

只训练 PR 线性残差，原 encoder、分类头和 BN buffer 全部冻结。完整原始 Outer 前向重放通过；随后精确复用 hash-bound 原始表示/logit/概率作为固定锚点，Original 的所有最终指标与上一版逐患者完全一致。没有训练新的 Raw encoder，没有调 Outer 阈值。

四个模型的 AUROC/AP/Macro-F1/EZ-F1 点估计均提高。四个 Macro-F1 与 EZ-F1 配对95%区间均排除零；只有 SEEG-Net 的 AUROC/AP 区间均排除零。最强配置为 TimeConv-CNN+PR Linear Residual：AUROC **0.646821**、AP **0.445751**、Macro-F1 **0.548726**、EZ-F1 **0.359109**。仍低于同队列 D1/M8 参考的主要指标。

科学状态：**EXPLORATORY_REPEATED_TEST**。这是同一已查看队列上的新冻结分析，不是新的盲测确认。所有模型名称均指前一实验中已审计的 adapted 实现，不冒充论文完整官方协议。

## 1. Patient-macro 主结果

先逐患者计算，再80人等权平均；EZ=1；AP指Average Precision；不是 pooled AUROC。EZ-Frac. Bias 是平均的预测EZ比例减真实EZ比例，负值表示少报EZ。

| 模型 | 版本 | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | ACC | MCC | EZ-Frac. Bias |
|---|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN | Original | 0.620571 | 0.431539 | 0.491133 | 0.266967 | 0.715298 | 0.667590 | 0.194159 | 0.019725 |
| TimeConv-CNN | PRLinearResidual | 0.646821 | 0.445751 | 0.548726 | 0.359109 | 0.738344 | 0.674735 | 0.223158 | 0.045213 |
| EEGNet | Original | 0.560085 | 0.345239 | 0.441691 | 0.125678 | 0.757705 | 0.679370 | 0.031206 | -0.061747 |
| EEGNet | PRLinearResidual | 0.589277 | 0.359876 | 0.500969 | 0.206613 | 0.795324 | 0.693763 | 0.085648 | -0.076551 |
| SEEG-Net | Original | 0.450626 | 0.263783 | 0.422239 | 0.030769 | 0.813710 | 0.728502 | -0.075213 | -0.176567 |
| SEEG-Net | PRLinearResidual | 0.601770 | 0.371180 | 0.512852 | 0.242121 | 0.783583 | 0.676798 | 0.088196 | -0.044115 |
| SEEGformer | Original | 0.534515 | 0.318833 | 0.424519 | 0.159583 | 0.689454 | 0.625904 | 0.015576 | 0.025714 |
| SEEGformer | PRLinearResidual | 0.557583 | 0.339779 | 0.484604 | 0.227220 | 0.741989 | 0.641871 | 0.059939 | 0.011914 |

## 2. 配对提升及95%区间

Δ=PRLinearResidual−Original。每次对同一80位患者ID配对重采样，10,000次、seed42，百分位95%CI；模型与数值阈值固定，不在bootstrap内重训练/优化。区间未做多重比较校正，也不代表跨训练seed不确定性。未定义指标按配对有效患者计算。

| 模型 | 指标 | Δ | 95%CI下界 | 95%CI上界 |
|---|---|---|---|---|
| TimeConv-CNN | accuracy | 0.007144 | -0.017095 | 0.032222 |
| TimeConv-CNN | f1 | 0.092142 | 0.053672 | 0.130898 |
| TimeConv-CNN | macro_f1 | 0.057594 | 0.034799 | 0.081373 |
| TimeConv-CNN | auroc | 0.026251 | -0.007904 | 0.062412 |
| TimeConv-CNN | auprc | 0.014213 | -0.008091 | 0.037793 |
| TimeConv-CNN | mrr | 0.036813 | -0.031351 | 0.106910 |
| EEGNet | accuracy | 0.014394 | -0.017269 | 0.048415 |
| EEGNet | f1 | 0.080936 | 0.042502 | 0.119032 |
| EEGNet | macro_f1 | 0.059277 | 0.037026 | 0.083228 |
| EEGNet | auroc | 0.029192 | -0.007883 | 0.068820 |
| EEGNet | auprc | 0.014637 | -0.018825 | 0.049231 |
| EEGNet | mrr | 0.049634 | -0.047006 | 0.148676 |
| SEEG-Net | accuracy | -0.051703 | -0.086426 | -0.010305 |
| SEEG-Net | f1 | 0.211352 | 0.174999 | 0.248654 |
| SEEG-Net | macro_f1 | 0.090612 | 0.063409 | 0.123556 |
| SEEG-Net | auroc | 0.151144 | 0.094523 | 0.209257 |
| SEEG-Net | auprc | 0.107397 | 0.061333 | 0.156869 |
| SEEG-Net | mrr | 0.200458 | 0.084101 | 0.315375 |
| SEEGformer | accuracy | 0.015967 | -0.021793 | 0.055157 |
| SEEGformer | f1 | 0.067637 | 0.027426 | 0.108581 |
| SEEGformer | macro_f1 | 0.060086 | 0.033627 | 0.088268 |
| SEEGformer | auroc | 0.023068 | -0.012783 | 0.060134 |
| SEEGformer | auprc | 0.020946 | -0.002097 | 0.045255 |
| SEEGformer | mrr | 0.059253 | -0.018157 | 0.135299 |

TimeConv-CNN 的 Macro-F1 +0.057594 [0.034799,0.081373]，EZ-F1 +0.092142 [0.053672,0.130898]；但 AUROC +0.026251、AP +0.014213 的区间跨零，不能声称可靠的排序提升。

SEEG-Net 的 AUROC +0.151144 [0.094523,0.209257]、AP +0.107397 [0.061333,0.156869]。Original 极低敏感度/少报EZ得以缓解，但 ACC 从0.728502下降到0.676798，不能只挑增益指标。其余两模型的 AUROC/AP 增量区间也跨零。

## 3. 与上一版端到端 PR 区分

上一版 Original/+PR 是分别从相同初始化端到端训练；本版保留 Original 的已训练encoder和分类头，只添加一个线性logit残差。两者不是同一种训练方法，不应混称。下表只做已冻结结果的描述比较，没有据此更改任何本版设置。

| 模型 | 上一版端到端PR AUROC | 本版残差 AUROC | 上一版端到端PR Macro-F1 | 本版残差 Macro-F1 |
|---|---|---|---|---|
| TimeConv-CNN | 0.600367 | 0.646821 | 0.506914 | 0.548726 |
| EEGNet | 0.547871 | 0.589277 | 0.498119 | 0.500969 |
| SEEG-Net | 0.561613 | 0.601770 | 0.496440 | 0.512852 |
| SEEGformer | 0.583621 | 0.557583 | 0.503621 | 0.484604 |

本版 TimeConv、EEGNet、SEEG-Net 的 AUROC/Macro-F1 点估计均高于上一版端到端PR；SEEGformer则更低。因此不能声称冻结残差在所有backbone上都优于端到端PR；本轮未做两种PR方法间的新增配对显著性检验。

## 4. 固定模型与训练细节

公式为：

`logit_EZ_new = -logit_NEZ_original + w^T PR(h) + b`

其中h是原完整窗口/发作可微聚合后的channel embedding；只有w,b训练。PR为全患者有效通道逐维float64 population mean/SD；SD>1e-8以SD为分母，否则为1，再转float32。不是logit Z、minibatch PR、跨患者归一化或新人工特征。线性层零初始化，epoch0作为no-op候选。PR使用推理时可获得的无标签完整患者通道集。

沿用原五折 FIT/validation/Outer patient-disjoint划分，不重新划分、不将validation纳入头拟合。每折原encoder自身是已由同一合法validation选择的checkpoint；不能把它称为额外独立的encoder验证集。

头训练：CPU float32 AdamW，lr0.001、weight decay0.001、clip1，max200/min10/patience20，无搜索、seed42。每epoch一次完整FIT更新；NEZ-positive weightedBCE权重=FIT EZ/NEZ，先患者内均值再患者等权均值。原Raw BCE方向不变，报告统一转EZ=1。

按validation patient Macro-F1→EZ-F1→更早epoch选择；阈值在parent的0:0.005:1网格上由validation patient Macro-F1→EZ-F1→接近0.5→较低阈值选择。全部20个残差权重/阈值一起冻结后才生成残差Outer分数，封存后才读取完整Outer标签评分。没有final-refit到validation，也没有Outer驱动的epoch/阈值/残差比例改变。

新增参数量：TimeConv257、EEGNet481、SEEG-Net513、SEEGformer385；原数百万参数encoder/classifier全部不更新。

### 实际选择与轻量头诊断

| 模型 | 折 | 最佳epoch | 实际epochs | 阈值 | 参数 | w L2 | bias | epoch0验证Macro-F1 | 最佳验证Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN | 1 | 194 | 200 | 0.745000 | 257 | 1.429281 | -0.188753 | 0.462893 | 0.603133 |
| TimeConv-CNN | 2 | 15 | 35 | 0.605000 | 257 | 0.114714 | -0.015043 | 0.542582 | 0.612142 |
| TimeConv-CNN | 3 | 20 | 40 | 0.975000 | 257 | 0.175032 | -0.019941 | 0.570721 | 0.590263 |
| TimeConv-CNN | 4 | 42 | 62 | 0.700000 | 257 | 0.264509 | -0.041902 | 0.544040 | 0.596223 |
| TimeConv-CNN | 5 | 28 | 48 | 0.510000 | 257 | 0.206366 | 0.027044 | 0.552163 | 0.597032 |
| EEGNet | 1 | 4 | 24 | 0.645000 | 481 | 0.072725 | -0.004001 | 0.516182 | 0.563129 |
| EEGNet | 2 | 62 | 82 | 0.645000 | 481 | 0.705228 | 0.021115 | 0.496339 | 0.561000 |
| EEGNet | 3 | 19 | 39 | 0.595000 | 481 | 0.297474 | 0.017205 | 0.485735 | 0.537306 |
| EEGNet | 4 | 1 | 21 | 0.665000 | 481 | 0.021909 | -0.001000 | 0.488337 | 0.509008 |
| EEGNet | 5 | 130 | 150 | 0.610000 | 481 | 1.248774 | -0.125587 | 0.448015 | 0.534712 |
| SEEG-Net | 1 | 5 | 25 | 0.700000 | 513 | 0.086561 | -0.005002 | 0.433732 | 0.519018 |
| SEEG-Net | 2 | 12 | 32 | 0.625000 | 513 | 0.165762 | -0.012026 | 0.474970 | 0.591452 |
| SEEG-Net | 3 | 20 | 40 | 0.660000 | 513 | 0.268311 | -0.020106 | 0.454537 | 0.580026 |
| SEEG-Net | 4 | 4 | 24 | 0.605000 | 513 | 0.067862 | -0.004005 | 0.433058 | 0.572399 |
| SEEG-Net | 5 | 25 | 45 | 0.610000 | 513 | 0.260468 | -0.025726 | 0.443089 | 0.499410 |
| SEEGformer | 1 | 17 | 37 | 0.500000 | 385 | 0.268111 | 0.006791 | 0.447008 | 0.515142 |
| SEEGformer | 2 | 43 | 63 | 0.560000 | 385 | 0.505099 | 0.037317 | 0.466575 | 0.530238 |
| SEEGformer | 3 | 11 | 31 | 0.560000 | 385 | 0.188233 | 0.010701 | 0.503339 | 0.506430 |
| SEEGformer | 4 | 19 | 39 | 0.610000 | 385 | 0.276311 | -0.019795 | 0.477275 | 0.523282 |
| SEEGformer | 5 | 36 | 56 | 0.550000 | 385 | 0.482894 | -0.039184 | 0.489879 | 0.535537 |

bias也是新增可训练参数，因此本结果不能将增量唯一归因于PR归一化本身；新增线性解码能力、截距与validation决策阈值的作用尚未拆分。不能据此宣称纯PR的独立因果贡献。

## 5. 原始预测复现、冻结和独立验证

- 四个backbone全部五折均真实运行冻结完整encoder+分类头前向，保持原FP32、chunk、raw/mask和完整聚合；没有只检查缓存文件而省略encoder重放。
- 预先锁定概率/logit/embedding最大误差门槛分别1e-6/1e-5/1e-4。实际三者全体最大值均1.1920928955078125e-7；validation概率重放最大差5.960464477539063e-8。
- 通过门槛后Outer精确复用历史hash-bound表示/logit/概率，Original所有患者级指标完全一致；这不是调整任何原预测。FIT/validation使用同checkpoint新提取表示。
- 原模型所有参数requires_grad=False、eval、提取前后完整state hash不变。原input、checkpoint、封存、个人预测和指标文件哈希均未变。
- 合成zero-head输出logit/概率严格identity，PR mask/常量维/通道置换和非零梯度测试PASS。
- 独立复核8组/640条患者指标与sklearn一致（float64概率裁剪），20个epoch0+全history argmax、validation阈值均重算PASS；10,000次配对bootstrap最大误差1.5265566588595902e-16，保持1e-12容差。
- 本版Outer freeze SHA256：`c244cf17f4f892f1412475eaa5ff56e4eba4d9556435c7e88ed5e23517aeffd8`；预测seal SHA256：`fb1870e8aab1288f2bdd36fb1e4733f8bb99c4544d45b9b212023230134535dc`。

首个import-only启动因parent wildcard import覆盖本地ROOT常量而未能找到新protocol文件；当时尚无新input freeze、表示、头训练或Outer输出。将本地变量改名TASK_ROOT后固定代码重新启动，失败源码和日志保留私有。没有改变科学配置或丢弃已完成训练。

## 6. Pooled channel-level结果

与patient-macro口径不同：每组7,635通道合并，仍使用已冻结的各折阈值。

| 模型 | 版本 | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | ACC |
|---|---|---|---|---|---|---|---|
| TimeConv-CNN | Original | 0.527132 | 0.256007 | 0.547280 | 0.317747 | 0.776812 | 0.663654 |
| TimeConv-CNN | PRLinearResidual | 0.563627 | 0.278633 | 0.568851 | 0.358331 | 0.779372 | 0.671644 |
| EEGNet | Original | 0.541994 | 0.250080 | 0.524833 | 0.238550 | 0.811116 | 0.697315 |
| EEGNet | PRLinearResidual | 0.561415 | 0.269079 | 0.522166 | 0.227831 | 0.816502 | 0.703471 |
| SEEG-Net | Original | 0.478290 | 0.227562 | 0.473313 | 0.096596 | 0.850031 | 0.742764 |
| SEEG-Net | PRLinearResidual | 0.538048 | 0.253197 | 0.517516 | 0.240222 | 0.794810 | 0.676883 |
| SEEGformer | Original | 0.536804 | 0.245160 | 0.502224 | 0.249000 | 0.755448 | 0.631041 |
| SEEGformer | PRLinearResidual | 0.545076 | 0.258058 | 0.513775 | 0.259868 | 0.767682 | 0.646365 |

## 7. 扩展分类/概率指标（patient-macro）

ECE为10等宽bin，不用Outer标签校准。未定义值保留NaN及覆盖率，不填0.5。

| 模型 | 版本 | Precision | Recall | Specificity | Weighted-F1 | Macro-P | Macro-R | Kappa | NPV | FPR | FNR | Brier | LogLoss | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN | Original | 0.423798 | 0.362453 | 0.766740 | 0.628165 | 0.625929 | 0.564596 | 0.132697 | 0.792379 | 0.233260 | 0.637547 | 0.376366 | 1.149805 | 0.427118 |
| TimeConv-CNN | PRLinearResidual | 0.417628 | 0.474080 | 0.758003 | 0.667140 | 0.612940 | 0.616041 | 0.194215 | 0.810271 | 0.241997 | 0.525920 | 0.358485 | 1.123302 | 0.421815 |
| EEGNet | Original | 0.305780 | 0.188472 | 0.826700 | 0.632595 | 0.616150 | 0.507586 | 0.013940 | 0.759510 | 0.173300 | 0.811528 | 0.248983 | 0.692119 | 0.282528 |
| EEGNet | PRLinearResidual | 0.336033 | 0.219496 | 0.851358 | 0.675873 | 0.569232 | 0.535427 | 0.065520 | 0.773589 | 0.148642 | 0.780504 | 0.253078 | 0.706673 | 0.298172 |
| SEEG-Net | Original | 0.205305 | 0.049840 | 0.935378 | 0.649384 | 0.694229 | 0.492609 | -0.011370 | 0.754270 | 0.064622 | 0.950160 | 0.308812 | 0.813562 | 0.358896 |
| SEEG-Net | PRLinearResidual | 0.291361 | 0.294961 | 0.819848 | 0.671031 | 0.534271 | 0.557404 | 0.081297 | 0.777182 | 0.180152 | 0.705039 | 0.304870 | 0.808021 | 0.367357 |
| SEEGformer | Original | 0.231280 | 0.273783 | 0.734232 | 0.586262 | 0.550144 | 0.504008 | 0.011199 | 0.763006 | 0.265768 | 0.726217 | 0.225459 | 0.649892 | 0.230111 |
| SEEGformer | PRLinearResidual | 0.281704 | 0.321191 | 0.757273 | 0.642740 | 0.531716 | 0.539232 | 0.042830 | 0.771497 | 0.242727 | 0.678809 | 0.236995 | 0.684609 | 0.264298 |

整体TP/TN/FP/FN见[confusion_matrices](results/confusion_matrices.csv)。

## 8. 患者内排名

| 模型 | 版本 | MRR | Hit@1 | Hit@3 | Hit@5 | Precision@3 | Recall@3 | NDCG@5 |
|---|---|---|---|---|---|---|---|---|
| TimeConv-CNN | Original | 0.570050 | 0.437500 | 0.650000 | 0.725000 | 0.462500 | 0.098280 | 0.447805 |
| TimeConv-CNN | PRLinearResidual | 0.606863 | 0.450000 | 0.712500 | 0.775000 | 0.475000 | 0.097897 | 0.474592 |
| EEGNet | Original | 0.475009 | 0.300000 | 0.562500 | 0.712500 | 0.329167 | 0.075835 | 0.335046 |
| EEGNet | PRLinearResidual | 0.524643 | 0.387500 | 0.600000 | 0.687500 | 0.362500 | 0.077097 | 0.360899 |
| SEEG-Net | Original | 0.341000 | 0.225000 | 0.362500 | 0.437500 | 0.208333 | 0.023587 | 0.215250 |
| SEEG-Net | PRLinearResidual | 0.541458 | 0.387500 | 0.612500 | 0.750000 | 0.379167 | 0.088668 | 0.362703 |
| SEEGformer | Original | 0.443752 | 0.262500 | 0.562500 | 0.675000 | 0.270833 | 0.057147 | 0.277172 |
| SEEGformer | PRLinearResidual | 0.503005 | 0.325000 | 0.600000 | 0.737500 | 0.329167 | 0.069942 | 0.333978 |

## 9. 五折均值±样本标准差（patient-macro）

五折患者数16/16/17/15/16；五折等权 mean±SD(ddof=1)与80人整体patient-macro不同，不能互换。

| 模型 | 版本 | ACC | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---|---|---|---|---|---|
| TimeConv-CNN | Original | 0.665727 ± 0.078616 | 0.618875 ± 0.123672 | 0.430866 ± 0.127766 | 0.490158 ± 0.080749 | 0.267365 ± 0.145358 |
| TimeConv-CNN | PRLinearResidual | 0.673018 ± 0.059118 | 0.645099 ± 0.096200 | 0.445424 ± 0.103810 | 0.547696 ± 0.066100 | 0.359317 ± 0.105810 |
| EEGNet | Original | 0.677435 ± 0.100042 | 0.560675 ± 0.058258 | 0.346898 ± 0.048997 | 0.441245 ± 0.023651 | 0.126363 ± 0.106248 |
| EEGNet | PRLinearResidual | 0.692643 ± 0.060164 | 0.588916 ± 0.051860 | 0.360869 ± 0.040977 | 0.499726 ± 0.046929 | 0.204970 ± 0.064352 |
| SEEG-Net | Original | 0.726969 ± 0.048853 | 0.451011 ± 0.011429 | 0.265502 ± 0.072997 | 0.422005 ± 0.013824 | 0.031811 ± 0.034981 |
| SEEG-Net | PRLinearResidual | 0.675282 ± 0.058838 | 0.601187 ± 0.060038 | 0.372052 ± 0.049490 | 0.512149 ± 0.043221 | 0.242063 ± 0.059348 |
| SEEGformer | Original | 0.622772 ± 0.130982 | 0.533666 ± 0.053171 | 0.319452 ± 0.067749 | 0.423594 ± 0.039059 | 0.161948 ± 0.102879 |
| SEEGformer | PRLinearResidual | 0.640130 ± 0.109010 | 0.557204 ± 0.057586 | 0.340787 ± 0.071992 | 0.484047 ± 0.023705 | 0.227579 ± 0.079498 |

## 10. 中心汇总（patient-macro）

描述性汇总，不按中心拟合/调阈值或选择性剔除患者。

| 模型 | 版本 | 中心 | 患者数 | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---|---|---|---|---|---|---|
| TimeConv-CNN | Original | hup | 36 | 0.658067 | 0.438661 | 0.534001 | 0.260461 |
| TimeConv-CNN | Original | lzu | 21 | 0.597579 | 0.496230 | 0.477134 | 0.318003 |
| TimeConv-CNN | Original | multicenter | 15 | 0.635819 | 0.420677 | 0.489871 | 0.222422 |
| TimeConv-CNN | Original | pediatric | 8 | 0.483600 | 0.250036 | 0.337338 | 0.245799 |
| TimeConv-CNN | PRLinearResidual | hup | 36 | 0.689078 | 0.453167 | 0.592401 | 0.364559 |
| TimeConv-CNN | PRLinearResidual | lzu | 21 | 0.591831 | 0.488638 | 0.517364 | 0.366645 |
| TimeConv-CNN | PRLinearResidual | multicenter | 15 | 0.684058 | 0.475379 | 0.579241 | 0.377771 |
| TimeConv-CNN | PRLinearResidual | pediatric | 8 | 0.531198 | 0.244247 | 0.377301 | 0.279811 |
| EEGNet | Original | hup | 36 | 0.532050 | 0.264942 | 0.432924 | 0.057668 |
| EEGNet | Original | lzu | 21 | 0.508016 | 0.441383 | 0.417493 | 0.220340 |
| EEGNet | Original | multicenter | 15 | 0.670085 | 0.411063 | 0.481196 | 0.150688 |
| EEGNet | Original | pediatric | 8 | 0.616672 | 0.330775 | 0.470595 | 0.136339 |
| EEGNet | PRLinearResidual | hup | 36 | 0.612630 | 0.301100 | 0.513582 | 0.194710 |
| EEGNet | PRLinearResidual | lzu | 21 | 0.516693 | 0.449209 | 0.463049 | 0.236305 |
| EEGNet | PRLinearResidual | multicenter | 15 | 0.666801 | 0.416281 | 0.548507 | 0.270107 |
| EEGNet | PRLinearResidual | pediatric | 8 | 0.529365 | 0.284111 | 0.454611 | 0.063187 |
| SEEG-Net | Original | hup | 36 | 0.438612 | 0.208172 | 0.440111 | 0.015253 |
| SEEG-Net | Original | lzu | 21 | 0.483126 | 0.422949 | 0.378077 | 0.055879 |
| SEEG-Net | Original | multicenter | 15 | 0.398393 | 0.180585 | 0.428005 | 0.045560 |
| SEEG-Net | Original | pediatric | 8 | 0.517312 | 0.252213 | 0.446934 | 0.006944 |
| SEEG-Net | PRLinearResidual | hup | 36 | 0.625626 | 0.329038 | 0.543672 | 0.270376 |
| SEEG-Net | PRLinearResidual | lzu | 21 | 0.553683 | 0.458561 | 0.457603 | 0.215405 |
| SEEG-Net | PRLinearResidual | multicenter | 15 | 0.647567 | 0.422554 | 0.529612 | 0.244325 |
| SEEG-Net | PRLinearResidual | pediatric | 8 | 0.534774 | 0.235113 | 0.487765 | 0.180970 |
| SEEGformer | Original | hup | 36 | 0.545648 | 0.275143 | 0.444153 | 0.114040 |
| SEEGformer | Original | lzu | 21 | 0.507748 | 0.444166 | 0.410567 | 0.230595 |
| SEEGformer | Original | multicenter | 15 | 0.549082 | 0.290199 | 0.474024 | 0.127857 |
| SEEGformer | Original | pediatric | 8 | 0.527370 | 0.240128 | 0.279964 | 0.237608 |
| SEEGformer | PRLinearResidual | hup | 36 | 0.565916 | 0.279936 | 0.498621 | 0.221770 |
| SEEGformer | PRLinearResidual | lzu | 21 | 0.512128 | 0.443666 | 0.457344 | 0.272527 |
| SEEGformer | PRLinearResidual | multicenter | 15 | 0.607796 | 0.374430 | 0.491612 | 0.183998 |
| SEEGformer | PRLinearResidual | pediatric | 8 | 0.545256 | 0.271402 | 0.479949 | 0.213852 |

## 11. 覆盖与限制

八组主要指标和排名均有80位有效患者，全部患者均保留。以下为存在未定义值的指标；其他指标均80/80。

| 模型 | 版本 | 指标 | 有效患者 | 未定义患者 |
|---|---|---|---|---|
| TimeConv-CNN | Original | precision | 64 | 16 |
| TimeConv-CNN | Original | mcc | 59 | 21 |
| TimeConv-CNN | Original | npv | 75 | 5 |
| TimeConv-CNN | PRLinearResidual | precision | 79 | 1 |
| TimeConv-CNN | PRLinearResidual | mcc | 77 | 3 |
| TimeConv-CNN | PRLinearResidual | npv | 78 | 2 |
| EEGNet | Original | precision | 51 | 29 |
| EEGNet | Original | mcc | 51 | 29 |
| EEGNet | PRLinearResidual | precision | 74 | 6 |
| EEGNet | PRLinearResidual | mcc | 74 | 6 |
| SEEG-Net | Original | precision | 18 | 62 |
| SEEG-Net | Original | mcc | 17 | 63 |
| SEEG-Net | Original | npv | 79 | 1 |
| SEEGformer | Original | precision | 60 | 20 |
| SEEGformer | Original | mcc | 59 | 21 |
| SEEGformer | Original | npv | 79 | 1 |
| SEEGformer | PRLinearResidual | precision | 78 | 2 |
| SEEGformer | PRLinearResidual | mcc | 78 | 2 |

没有预测阳性时precision未定义；常量预测时MCC可能未定义。其均值比较需注意有效患者不同。Macro-Precision对可定义分量求均值，不能替代EZ recall与coverage。

本轮只做固定seed42与固定线性设置，未寻优。最佳TimeConv残差仍低于原同队列D1（AUROC0.732636、AP0.527175、Macro-F10.616556、EZ-F10.408531）及M8（0.762397、0.554067、0.634097、0.434087）点估计。D1/M8具有不同生理上下文/结构/训练目标，本轮没有重新评估它们，也没有新的差值显著性检验。

## 12. 交付

父分支：`codex/m8-raw-backbones-pr-seed42-v1`，父提交`5f660b5bb3861aa97c879122624132ce5d024171`。本地新分支：`codex/m8-raw-frozen-pr-linear-residual-seed42-v1`；预结果协议/代码提交`26d3bd2`。数据与旧模型不改动。

完整[主表](results/main_results.csv)、[增量](results/residual_improvement.csv)、[配对区间](results/paired_residual_intervals.csv)、[pooled](results/pooled_metrics.csv)、[折指标](results/fold_metrics.csv)、[中心](results/center_metrics.csv)、[coverage](results/metric_coverage.csv)、[重放审计](results/REPRESENTATION_REPLAY_AUDIT.csv)、[验证选择](results/VALIDATION_SELECTION.csv)、[参数诊断](results/HEAD_PARAMETER_DIAGNOSTICS.csv)、[独立指标复核](results/INDEPENDENT_VERIFICATION.json)、[独立选择复核](results/SELECTION_HISTORY_VERIFICATION.json)、[最终状态](results/FINAL_STATUS.json)、[公开文件哈希](results/PUBLIC_SHA256.json)。

逐患者/通道表示、logit、预测、标签、原始信号、缓存、权重、逐记录身份及运行日志只留可信服务器；这里只保存代码、哈希及匿名汇总。没有访问或清理其他实验文件，没有运行已暂停的旧训练自动任务。
