# 无Early FiLM / 无Late Gate：seed42、44、62完整消融

已完成17组×3seeds×原5折＝255个模型单元：50个严格复用、205次新训练，共2405实际训练轮次。每组/seed保留原80人、7635通道、256次发作及1471965有效Raw250窗口。Variance和Head Interaction始终删除。

独立核验通过4080行患者/seed结果、36720项指标、306项名义区间及32项主要比较族区间。全部255个选择单元先冻结，随后255个Outer预测先全部封存，再汇总结果。没有按本轮Outer重选主模型、seed、阈值或特征。

**解释边界：** 用户是在查看过历史Outer和各seed结果后指定本主模型及seed42/44/62。因此这是结果知情选择seed子集的探索性重复测试，不是随机seed稳健性证据，也不是独立确认。主模型完整五个已看seed的描述性均值Macro-F1为0.639618；本次指定三seed为0.643771。未隐藏43/52，见[seed选择审计](public/SEED_SELECTION_AUDIT.csv)。

## 指定主模型

| 患者平均指标 | 三seed均值 | seed样本标准差 |
| --- | ---: | ---: |
| Macro-F1 | 0.643771 | 0.002070 |
| AUROC | 0.766660 | 0.006766 |
| AP | 0.570795 | 0.008784 |
| EZ-F1 | 0.451637 | 0.005785 |
| NEZ-F1 | 0.835906 | 0.003354 |
| ACC | 0.760550 | 0.003398 |
| MRR | 0.787698 | 0.015242 |
| Top-1 | 0.695833 | 0.026021 |
| EZ-Frac. Bias | -0.026136 | 0.009483 |

以上是患者等权指标的三seed算术均值，不是最佳seed、预测集成或pooled通道指标。Macro-F1名义患者bootstrap95%区间[0.616862,0.671263]，AUROC[0.732656,0.801281]；seed标准差不是置信区间。

## 全部配置

| 配置 | Macro-F1 | AUROC | AP | EZ-F1 |
| --- | ---: | ---: | ---: | ---: |
| Full：无 Early FiLM / Late Gate | 0.643771 | 0.766660 | 0.570795 | 0.451637 |
| 恢复 Early FiLM（对照） | 0.641733 | 0.765902 | 0.592297 | 0.450877 |
| w/o Feature PR | 0.621511 | 0.736473 | 0.543781 | 0.407149 |
| w/o Raw PR | 0.627177 | 0.763588 | 0.561702 | 0.438996 |
| w/o BOTH PR | 0.492631 | 0.749737 | 0.554090 | 0.370815 |
| w/o Pre-reference | 0.616957 | 0.731702 | 0.535751 | 0.410349 |
| Raw-only | 0.614705 | 0.728789 | 0.535852 | 0.392629 |
| Physiology-only | 0.624862 | 0.749580 | 0.542016 | 0.424626 |
| 删除 β | 0.642765 | 0.766926 | 0.570446 | 0.449590 |
| 删除低γ | 0.647015 | 0.769884 | 0.576576 | 0.456121 |
| 删除部分高γ | 0.643546 | 0.766981 | 0.567885 | 0.450601 |
| 删除 RMS | 0.639698 | 0.760393 | 0.558835 | 0.442262 |
| 删除 Line Length | 0.647034 | 0.765649 | 0.573932 | 0.457006 |
| 删除 Spectral Entropy | 0.642594 | 0.761528 | 0.562885 | 0.448509 |
| P1 Temporal Statistics | 0.642490 | 0.766303 | 0.564955 | 0.449103 |
| P2 Time-aware Attention | 0.644494 | 0.767045 | 0.572194 | 0.452035 |
| Capacity-matched 无FiLM | 0.643904 | 0.770690 | 0.572816 | 0.450481 |

所有9项指标完整列于[Outer报告](public/REPORT.md)和[MAIN_RESULTS.csv](public/MAIN_RESULTS.csv)。另有[每seed](public/SEED_RESULTS.csv)、[seed样本SD](public/SEED_SD.csv)、[255个逐折结果](public/FOLD_RESULTS.csv)、[折间SD](public/FOLD_SD.csv)、[跨来源](public/SOURCE_RESULTS.csv)、[pooled单列](public/POOLED_RESULTS.csv)、[覆盖率](public/RAW_COVERAGE.csv)、[306项名义区间](public/INTERVALS.csv)、[32项比较族区间](public/PRIMARY_FAMILY_INTERVALS.csv)、[逐seed/折方向](public/FOLD_SEED_DIRECTIONS.csv)、[PR交互](public/INTERVALS.csv)和[独立核验](public/INDEPENDENT_OUTER_VERIFICATION.json)。

## 主要发现

- 去掉两类PR后Macro-F1降15.11pp，AUROC降1.69pp。Macro-F1比较族区间排除0，但AUROC连名义95%区间仍跨0；不能将两者都写成统计上确定下降。无PR的EZ预测比例偏差为+0.192474，主模型为-0.026136，说明分类工作点/类比例也是重要问题，而非排序完全失效。
- 去掉Feature PR，Macro-F1/AUROC分别下降2.23/3.02pp；去掉Raw PR下降1.66/0.31pp。后者不确定性较大；不能声称两个PR单支路都对排序有确定贡献。
- 主模型相对Raw-only、相对去掉自身发作前参考，在主要Macro-F1/AUROC比较族区间上均保留正向差异。
- 六项逐删中，低γ删除的两项点估计均提高，Line Length删除Macro-F1提高；置信区间不足以支持据此改主模型，单删也不能证明全局最优或某生理属性无效。
- 恢复Early FiLM并未改善平均Macro-F1/AUROC，却有更高AP/MRR/Top-1。无FiLM不是在所有指标上占优。P1/P2及普通容量匹配没有确立Macro-F1/AUROC的可靠统一优势。

操作定义、复用及额外参数差异见[冻结协议](PROTOCOL_LOCK.json)，详细证据与解释见[FINDINGS](FINDINGS.md)。四个已无实际操作对象的删除别名不重复训练、不重复计假设；FiLM错配及Early/Late交互不适用于本主模型。

## 可复现与隐私

源代码、真实服务器准入及复用证据在正式训练前推送并核对远端。全程保持原FP32/TF32off、4096窗口分块、完整患者更新、损失、优化器、预算及合法内部选轮次/阈值规则，不新增数据/特征/提取或性能入围门槛。

Outer全部封存后，原汇总调用缺少历史目录上下文；外部[报告调用修复](../../tools/drechan_nga_reporting_context_v1/README.md)仅调用原common.setup()，14个科学源码、全部预测和已生成汇总前缀逐字节不变。无重训、重预测或数值/阈值改动，见[运行审计](RUN_AUDIT.md)。

只发布代码、哈希、匿名汇总与审计；原始波形、身份、逐患者/通道预测、权重、预处理器、缓存和日志保留可信服务器。
