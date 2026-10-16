"""Report only completed, independently reconstructed anonymous internal results.

This file is outside the frozen scientific source list. It never imports a model,
opens a private artifact, selects a new candidate, or changes a decision rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    'C0_WINDOW_SHORT': 'C0 Window / short CNN',
    'C1_PRE_SHORT': 'C1 Pre-scale / short CNN',
    'C2_WINDOW_LONG': 'C2 Window / long CNN',
    'C3_PRE_LONG': 'C3 Pre-scale / long CNN',
    'C4_PRE_LONG_CONTEXT': 'C4 Pre-scale / long CNN + context',
    'DROP_PRE_REFERENCE': 'Drop pre-latent reference',
    'DROP_TWO_PHASES': 'Drop temporal phases',
    'DROP_SEIZURE_SD': 'Drop seizure SD',
    'DROP_PATIENT_Z': 'Drop patient Z',
}
METRICS = {
    'auroc': 'AUROC', 'auprc': 'AP', 'f1': 'EZ-F1',
    'macro_f1': 'Macro-F1', 'nez_f1': 'NEZ-F1', 'accuracy': 'ACC',
    'mrr': 'MRR', 'hit_at_1': 'Top-1', 'ez_fraction_bias': 'EZ-Frac. Bias',
}
COMPONENTS = {
    'DROP_PRE_REFERENCE': '发作前潜变量参考',
    'DROP_TWO_PHASES': '0–8 / 8–20秒分阶段',
    'DROP_SEIZURE_SD': '跨发作SD',
    'DROP_PATIENT_Z': '完整患者Patient-wise Z',
}


def table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return '' if pd.isna(value) else f'{value:.6f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    rows = ['| ' + ' | '.join(map(cell, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(map(cell, r)) + ' |'
             for r in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def load_complete(root):
    read = lambda n: json.loads((root / n).read_text(encoding='utf-8'))
    terminal, verification = read('INTERNAL_TERMINAL.json'), read('INDEPENDENT_VERIFICATION.json')
    assert terminal['status'] == 'COMPLETE_VERIFIED_INTERNAL_INFORMATION_RETENTION_STUDY'
    assert terminal['new_Outer_passes'] == 0 and verification['status'] == 'PASS'
    main = pd.read_csv(root / 'FIRST_PATH_MAIN_RESULTS.csv').set_index('arm')
    assert set(main.index) == set(LABELS) and main.index.is_unique
    assert np.isfinite(main[list(METRICS)].to_numpy()).all()
    return terminal, verification, main


def main(root, output):
    terminal, verification, values = load_complete(root)
    pairs = pd.read_csv(root / 'FIRST_PATH_PAIRED_INTERVALS.csv')
    absolute = pd.read_csv(root / 'FIRST_PATH_ABSOLUTE_INTERVALS.csv')
    folds = pd.read_csv(root / 'FIRST_PATH_FOLD_RESULTS.csv')
    sources = pd.read_csv(root / 'FIRST_PATH_SOURCE_RESULTS.csv')
    admission = json.loads((root / 'GRADIENT_ADMISSION.json').read_text(encoding='utf-8'))
    assert admission['status'] == 'PASS'
    parameters = {r['arm']: r['parameters'] for r in admission['reports']}
    primary = values.loc[list(LABELS), list(METRICS)].reset_index()
    primary.insert(1, 'parameters', primary.arm.map(parameters))
    primary['arm'] = primary.arm.map(LABELS)
    primary = primary.rename(columns={'arm': '配置', 'parameters': '参数量', **METRICS})
    effects = []
    for arm, label in COMPONENTS.items():
        for metric in ['f1', 'auroc', 'auprc', 'macro_f1', 'nez_f1', 'accuracy', 'mrr', 'hit_at_1']:
            row = pairs[(pairs.left == arm) & (pairs.right == 'C0_WINDOW_SHORT') & (pairs.metric == metric)]
            assert len(row) == 1
            r = row.iloc[0]
            # Existing frozen summaries report deletion minus baseline. Reverse
            # the direction, including the endpoints, to report retention benefit.
            effects.append({'组件': label, '指标': METRICS[metric],
                            '保留减删除': -r['mean'], '95%下界': -r.high,
                            '95%上界': -r.low})
    effect_frame = pd.DataFrame(effects)
    contrasts = []
    for left, right in [('C1_PRE_SHORT', 'C0_WINDOW_SHORT'),
                        ('C2_WINDOW_LONG', 'C0_WINDOW_SHORT'),
                        ('C3_PRE_LONG', 'C2_WINDOW_LONG'),
                        ('C3_PRE_LONG', 'C1_PRE_SHORT'),
                        ('C4_PRE_LONG_CONTEXT', 'C3_PRE_LONG')]:
        for metric in ['f1', 'auroc', 'auprc', 'macro_f1', 'mrr', 'hit_at_1']:
            r = pairs[(pairs.left == left) & (pairs.right == right) & (pairs.metric == metric)]
            assert len(r) == 1
            r = r.iloc[0]
            contrasts.append({'配置': LABELS[left], '对照': LABELS[right],
                              '指标': METRICS[metric], '配对差': r['mean'],
                              '95%下界': r.low, '95%上界': r.high})
    stability = folds.groupby('arm')[['f1', 'auroc', 'auprc']].agg(['mean', 'std', 'min', 'max'])
    stability.columns = [f'{METRICS[a]} {b}' for a, b in stability.columns]
    stability = stability.reindex(list(LABELS)).reset_index()
    stability['arm'] = stability.arm.map(LABELS)
    stability = stability.rename(columns={'arm': '配置'})
    additional = pd.read_csv(root / 'ADDITIONAL_PATH_RESULTS.csv')
    raced_intervals = pd.read_csv(root / 'RACED_PATH_PAIRED_INTERVALS.csv')
    raced_intervals = raced_intervals[raced_intervals.metric.isin(['f1', 'auroc', 'auprc', 'macro_f1', 'mrr', 'hit_at_1'])]
    raced_display = raced_intervals[['scope', 'fold', 'left', 'metric', 'mean', 'low', 'high',
                                    'logical_patient_appearances', 'unique_patients']].copy()
    raced_display['metric'] = raced_display.metric.map(METRICS)
    races = json.loads((root / 'RACING_DECISIONS.json').read_text(encoding='utf-8'))['decisions']
    assert len(races) == 5
    race_table = pd.DataFrame(races)[['fold', 'chosen', 'advanced', 'initial_f1_gain', 'initial_auroc_gain']]
    race_table['chosen'] = race_table.chosen.map(LABELS)
    source_shown = sources[['arm', 'center', 'unique_patients', 'logical_appearances', 'suppressed', *METRICS]].copy()
    source_shown['arm'] = source_shown.arm.map(LABELS)
    source_shown = source_shown.rename(columns=METRICS)
    def paired_text(left, right, metric):
        r = pairs[(pairs.left == left) & (pairs.right == right) & (pairs.metric == metric)].iloc[0]
        return f"{r['mean']:+.6f}，95%区间[{r.low:+.6f}, {r.high:+.6f}]"
    candidate_max = values.loc[list(LABELS)[:5], 'f1'].idxmax()
    max_row = absolute[(absolute.arm == candidate_max) & (absolute.metric == 'f1')].iloc[0]
    interaction = (values.loc['C3_PRE_LONG', 'f1'] - values.loc['C2_WINDOW_LONG', 'f1']
                   - values.loc['C1_PRE_SHORT', 'f1'] + values.loc['C0_WINDOW_SHORT', 'f1'])
    interaction_ci = pd.read_csv(root / 'NORMALIZATION_ENCODER_INTERACTION.csv').set_index('metric').loc['f1']
    assert abs(interaction - interaction_ci['mean']) < 1e-12
    oracle = pd.read_csv(root / 'INTERNAL_ORACLE_THRESHOLD_DIAGNOSTIC.csv')
    oracle_table = []
    for arm in LABELS:
        g = oracle[oracle.arm == arm].set_index('metric')
        assert abs(g.loc['actual_f1', 'mean'] - values.loc[arm, 'f1']) < 1e-12
        gap = g.loc['oracle_minus_actual']
        oracle_table.append({'配置': LABELS[arm], '实际内部EZ-F1': g.loc['actual_f1', 'mean'],
                             '标签已知的阈值上限': g.loc['oracle_f1', 'mean'],
                             '乐观上限减实际': gap['mean'],
                             '差值95%下界': gap.low, '差值95%上界': gap.high})
    max_oracle = float(oracle[oracle.metric == 'oracle_f1']['mean'].max())
    threshold_conclusion = ('这九组固定分数在本次内部患者上的标签已知阈值上限均值仍低于0.65，'
                            '因此仅改变这些分数的阈值不足以在这组样本中达到目标，需改善排序表征。'
                            if max_oracle < .65 else
                            '阈值上限只显示有限样本中的乐观余量，尚不能据此判断可部署规则能否达到目标。')
    race_policy = raced_intervals[(raced_intervals.scope == 'advanced_FITs_only_additional_paths') &
                                  (raced_intervals.metric == 'f1')]
    race_conclusion = ''
    if len(race_policy):
        rr = race_policy.iloc[0]
        race_conclusion = (f"实际触发追加的FIT中，screen2/3的配对EZ-F1平均差为{rr['mean']:+.6f}，"
                           f"区间[{rr.low:+.6f}, {rr.high:+.6f}]。")
    text = f'''# Raw信息保留与组件贡献：内部患者隔离实验

## 结论边界

本轮完成{terminal['executed_cells']}个实际训练单元、{terminal['formal_epochs']}个完整epoch。
九个配置的首路径使用相同患者角色，评价了每个配置
{verification['first_path_logical_patients_per_arm']}次患者出现，涉及
{verification['first_path_unique_patients']}个不同患者。它们是各原始FIT内部的候选筛选结果，
**不是新的Outer Test，也不能据此宣布EZ-F1>0.65的最终目标已经达到**。
所有首路径及追加路径均按预先固定的规则运行，未使用本轮结果修改模型、阈值或预算。

五个架构/归一化候选中的最高描述性EZ-F1为{max_row['mean']:.6f}
（{LABELS[candidate_max]}，区间[{max_row.low:.6f}, {max_row.high:.6f}]）。
这个全体表格最大值不是逐FIT的合法选择策略，也不是目标达成证据。

## 结果判断

患者上下文候选C4相对原C0的EZ-F1差为
{paired_text('C4_PRE_LONG_CONTEXT', 'C0_WINDOW_SHORT', 'f1')}，没有提供稳定优于原模型的证据。
它相对C3有所改善，但不能把修复较弱配置的收益等同于超过原基线。
{race_conclusion}

四项删除对照支持保留发作前潜变量参考；Patient-wise Z和分阶段池化改善了EZ-F1，
但它们的AUROC差异没有显示同样的排序收益。跨发作SD没有显示稳定增量。
这些结论限定于当前Raw CNN、损失、预算和内部划分，不能上升为普适生理必要性。
保留波形相对幅度或加长CNN并未形成稳定的基线优势，也不能因此宣布生理幅度无效。

## 模型与比较问题

每个模型仅输入原始250Hz iEEG，使用单一分类头，不输入人工生理特征、D1分数或其他模型残差。
Window归一化逐2秒波形去均值、除以自身SD；Pre-scale则除以同发作同通道真实发作前窗口的
平均方差平方根。该尺度只用于波形归一化，不作为额外特征输入。

Short CNN在全局平均前具有33样本局部非线性感受野；Long CNN使用残差深度卷积扩展至535样本。
两者都读取完整500样本窗口。Long CNN同时改变了宽度、深度、卷积核和参数量，
因此其差值不能被解释为单独的感受野因果效应。
C4在同一个Raw分类头内加入零初始化的潜变量患者分布调制，不增加独立决策分支。

四项删除实验均从C0重新端到端训练，初始化、损失、内部划分和训练预算一致，
只移除指定池化组件。删除后仍保留96个分类槽位，但有效信息维度可能减少。
DROP_SEIZURE_SD只检验跨发作SD在均值之外的作用；DROP_PATIENT_Z同时涉及信息表征与优化条件，
不能把性能差全部解释成生理信息缺失。

## 九项完整指标

患者等权均值，EZ为正类。Top-1和MRR使用患者内通道排序；ACC和NEZ-F1单独报告，
避免用多数NEZ通道的改善代替EZ定位改善。EZ-Frac. Bias为预测EZ比例减真实EZ比例。

{table(primary)}

## 组件保留的端到端增量

下表为C0减相应删除配置；正数表示保留该组件较好。区间来自10000次患者ID聚类重抽样，
同一患者在多个内部路径的重复出现被一起抽取。它们描述本次选择条件下的不确定性，
不是独立确认区间；未进行多重比较校正，也不以区间是否跨零修改筛选规则。

{table(effect_frame)}

## 波形归一化、CNN结构与患者潜变量上下文

{table(pd.DataFrame(contrasts))}

Pre-scale在两种CNN内分别与Window比较；Long与Short在两种归一化内分别比较。
这些对照用于识别归一化是否依赖架构，不能由全体结果中挑一个最高分再声称普适收益。

Pre-scale在Short CNN内的EZ-F1差为{paired_text('C1_PRE_SHORT', 'C0_WINDOW_SHORT', 'f1')}；
在Long CNN内为{paired_text('C3_PRE_LONG', 'C2_WINDOW_LONG', 'f1')}。
两差之差的点估计为{interaction:+.6f}，联合四臂患者聚类区间为
[{interaction_ci.low:+.6f}, {interaction_ci.high:+.6f}]，不是从两个边际区间拼接得到。
潜变量患者调制相对同一Long/Pre-scale主干的EZ-F1差为
{paired_text('C4_PRE_LONG_CONTEXT', 'C3_PRE_LONG', 'f1')}。

## 五个FIT内部路径的差异

下表的SD是五个路径点估计的样本标准差，不是五个独立测试集的标准误。
不同原始FIT及内部路径的患者存在重叠。

{table(stability)}

## 原注册的逐FIT追加路径规则

每个原始FIT独立从C1–C4中选择候选，使用内部EZ-F1减来源间EZ-F1样本SD的0.1倍，
在最高分0.005以内优先较小模型。只有该候选比C0至少提高0.02 EZ-F1且AUROC不低于
C0减0.01，才追加另外两条内部路径。没有跨原始Outer折合并选择全局最优模型。

{table(race_table)}

追加路径实际完成{len(additional)}个训练单元。未被触发的单元不填充、不外推。
追加结果仍是候选筛选数据，不能因其与第一条路径不同就称作独立验证。

{table(additional.rename(columns=METRICS)) if len(additional) else '没有候选触发追加路径门槛。'}

追加路径配对差及患者ID聚类区间如下，只包括实际触发门槛的FIT及其screen2/3，
不混入选出候选所用的screen1。不同行的患者仍可能重叠，这些追加路径不是独立复现。

{table(raced_display) if len(raced_display) else '没有追加路径配对区间。'}

## 跨来源结果

少于5个独立患者的来源组只显示覆盖数，不发布结果。

{table(source_shown)}

## 数据与执行

原始队列合同保持80患者、7635通道、256次发作、1471965个有效Raw250窗口。
本轮只使用每个原始FIT内部数据；无标签输入审计覆盖其并集79患者、1455327个有效窗口。
原始Development和各自Outer不参与本轮拟合、校准、候选选择或评价。
继承的混合存档由原有读取器加载后立即限制到各自FIT；不能据此声称其他角色标签
从未在读取器内部反序列化，但它们未进入本轮计算。

全部模型从头训练，使用相同EZ定向损失、AdamW、FP32、TF32关闭及确定性cuDNN。
每条路径的cal患者选择epoch和阈值，held患者用于候选筛选，均不进入该路径的梯度更新。
两遍梯度缓存保留完整患者、所有有效窗口和可微聚合。
首版梯度准入失败及跨后端梯度差异保留在工程附录；确定性执行修订在首个正式epoch之前固定，
未把失败比较改写为等价性通过，未丢弃已完成的正式epoch。

独立结果复算重新加载实际最佳权重，回放cal分数、阈值及全部已保存held分数的指标。
cal最大回放误差为{verification['calibration_replay_max_error']:.9g}，
共核验{verification['patient_rows']}条私有患者结果；身份、波形、归一化系数、权重和逐患者/通道分数不发布。

## 目标与下一步

本轮新Outer次数为0。历史Outer已经多次被查看，其既往结果只具有探索性解释。
内部筛选结果仅决定是否值得另行注册少量最终训练与冻结评价；不能回看历史Outer选择配置，
也不能把内部最大EZ-F1、Macro-F1、NEZ-F1或 pooled F1替代患者EZ-F1>0.65的目标。

从源码可确定，阶段均值对同一阶段内窗口编码的排列不敏感，跨发作均值/SD也不保留发作顺序。
Long CNN只改变每个2秒窗口内部的编码，并未补回阶段内时间顺序。
这是一项表征结构限制，不是本轮数据证明的生理机制，也不证明加入序列模块必然提高定位性能。
若继续优化，需先以新的有限预算注册检验该限制，不能把本轮预算事后扩成无限搜索。

### 阈值瓶颈诊断，不是模型性能

以下是训练开始后新增的事后内部诊断，不是原注册终点。
对于每个held患者的固定通道分数，在已知该患者标签时枚举所有可由一个分数阈值实现的分类集合，
计算最高EZ-F1；相同分数的通道必须一起纳入，不能按标签拆开平分。
因此它只是当前有限样本、当前固定分数的乐观阈值上限，既使用了评价标签，又存在最大化偏差，
不是合法预测规则，也不预测外部数据可达到的性能。
实际阈值、权重、追加规则未因该诊断改变，诊断本身没有新增推理或Outer访问。

{table(pd.DataFrame(oracle_table))}

这一诊断只能指出现有分数中还有多少阈值层面的余量，不能把“上限超过0.65”写成目标已经达成。
九组分数的最高上限均值为{max_oracle:.6f}。{threshold_conclusion}

源表：`results/deterministic_execution_v2/FIRST_PATH_*`、`ADDITIONAL_PATH_RESULTS.csv`、
`RACING_DECISIONS.json`及`INDEPENDENT_VERIFICATION.json`。图和所有数值由同一匿名汇总表生成。
'''
    output.write_text(text, encoding='utf-8')
    effect_frame.to_csv(output.parent / 'COMPONENT_RETENTION_CONTRASTS.csv', index=False)
    stability.to_csv(output.parent / 'FOLD_STABILITY.csv', index=False)
    # Metric tables must agree numerically with the independent server output.
    for arm in LABELS:
        for metric in METRICS:
            r = absolute[(absolute.arm == arm) & (absolute.metric == metric)]
            assert len(r) == 1 and abs(float(r.iloc[0]['mean']) - values.loc[arm, metric]) < 1e-12
    (output.parent / 'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps({
        'status': 'PASS', 'source': 'independently_reconstructed_anonymous_internal_aggregates',
        'arms': len(LABELS), 'new_outer_passes': 0,
        'target_established': False, 'no_private_inputs': True,
        'main_vs_interval_mean_tolerance': 1e-12,
        'component_interval_direction_reversed_with_endpoints': True,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.output)
