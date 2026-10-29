"""Human-readable main-table-matched A-C report from anonymous releases."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def table(name):return (ROOT/'tables'/(name+'.md')).read_text(encoding='utf-8').split('\n',2)[2].strip()
text='''# PR 排序、患者稳健性与有限通道审阅：主表一致版 A–C

本版 A、B、C 已完成，覆盖全部 **80 人、7,635 通道、256 次发作、四个来源**。TimeConv-CNN 使用与主表相同的 **PR Linear Residual**：原 ictal 编码器和分类器冻结，患者相对表征通过已训练的线性残差头修正 EZ logit。SVM 与 DREChanNet 的 No PR、Full PR、Score-Z，以及 TimeConv 的 No PR、Score-Z，均沿用原封存版本。

本轮新增训练为 **0**。五个 TimeConv 残差头在 80 名 outer 患者的已有表征上重放，概率最大绝对误差 **0**；TimeConv Original 和 PR Linear Residual 的全部 11 项患者等权指标与主表原结果一致。

## A：患者内 EZ 排序

'''+table('Table_A')+'''

TimeConv 的 Macro-F1 从 **49.11% 升至 54.87%**，AUROC 从 **62.06% 升至 64.68%**，AP 从 **43.15% 升至 44.58%**。PR 相对 No PR 的 AUROC、AP 平均增益为正，但其配对 95% 区间均跨零。MRR、Hit@3 和 Hit@5 的均值提高，Hit@1 从 43.75% 升至 45.00%。

三种模型的 Score-Z 与各自 No PR 的 AUROC、AP、MRR、Hit@1/3/5 在每名患者上完全相同。Score-Z 可改变分类阈值下的结果，不能改变通道排序。

## B：患者与来源稳健性

### 患者配对增益

正、零、负列为对应患者比例。区间单位为百分点；12 项家族是三模型 × AUROC/AP/Macro-F1/EZ-F1，独立于下方六项 Score-Z 家族。

'''+table('Table_B')+'''

TimeConv 的 Macro-F1 和 EZ-F1 增益通过该 12 项家族调整；AUROC 提高的患者占 **53.75%**，AP 提高的患者占 **51.25%**。SVM 的 AP 和分类增益有调整后区间支持；其 AUROC 的 12 项调整区间跨零。DREChanNet 的分类增益有支持，AUROC/AP 的增益区间跨零。

### 与输出 Score-Z 的比较

'''+table('Table_B_score')+'''

TimeConv 相对 Score-Z 的 Macro-F1 差为 **+0.75 pp**，95% CI **[−1.98, 3.44]**；AUROC 差为 **+2.63 pp**，95% CI **[−0.73, 6.28]**。现有结果未支持它在这两项上优于 Score-Z。SVM 的两项差在六项家族调整后仍为正；DREChanNet 的两项差区间跨零。

### 四来源结果

这是同一五折 **patient-held-out outer** 预测按来源分组，并非 LOSO。CM 为四来源等权平均；Worst 为该指标四来源均值的最小值，各指标可来自不同最差来源。HUP/LZU/Multicenter/Pediatric 分别有 36/21/15/8 人。

Macro-F1（%）：

'''+table('Table_C_macro_f1')+'''

AUROC（%）：

'''+table('Table_C_auroc')+'''

AP（%）：

'''+table('Table_C_auprc')+'''

TimeConv 的 PR 相对 No PR 在四来源上的 Macro-F1 均值均提高；AUROC 在 LZU 由 59.76% 降至 59.18%。PR 的 Macro-F1 CM/Worst 为 **51.66%/37.73%**，低于 Score-Z 的 **52.29%/47.08%**。总体平均提高不代表每个来源或每项指标都提高。

## C：有限通道审阅

每名患者按预测 EZ 分数排序，取前 `max(1, ceil(q × 通道数))` 个通道，再对患者等权平均。EZ recall 衡量已标注 EZ 的覆盖比例；precision 衡量审阅通道中的 EZ 比例。Lift 是该 precision 相对本患者 EZ 比例的比值。

'''+table('Table_D')+'''

配对 EZ recall 增益（pp）：

'''+table('Table_D_paired')+'''

TimeConv 在审阅前 10% 通道时，EZ recall 从 **22.74% 升至 25.89%**，配对增益 **+3.14 pp**；95% CI **[0.97, 5.47]**，九项家族调整区间 **[0.17, 6.40]**。其 5% 和 20% 预算差的调整区间均跨零。SVM 的 5% 预算差通过九项调整；DREChanNet 三个预算的差区间均跨零。

固定 1/3/5 个通道（小于预算时取患者实际通道数）：

'''+table('Table_D_fixed')+'''

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
'''
(ROOT/'report.md').write_text(text,encoding='utf-8')
print('CHINESE_REPORT_BUILT')
