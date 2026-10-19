"""Reproducible Chinese findings from completed public aggregates only."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;PUB=HERE/'results'
LABELS={'FULL_F0':'主模型：六特征、无头部交互','WO_EARLY_FILM':'w/o Early FiLM','WO_BOTH_GUIDANCE':'w/o Both Guidance',
    'WO_BOTH_PR':'w/o 两类 PR','WO_RAW_PRE_REFERENCE':'w/o Raw Pre-reference','EARLY_FILM_MISMATCH':'Early FiLM 错配','RAW_ONLY':'Raw-only','PHYSIOLOGY_ONLY':'Physiology-only',
    'DROP_BETA':'删除 β','DROP_LOW_GAMMA':'删除低γ','DROP_PARTIAL_HIGH_GAMMA':'删除部分高γ（80–125 Hz）','DROP_RMS':'删除 RMS','DROP_LINE_LENGTH':'删除 Line Length','DROP_SPECTRAL_ENTROPY':'删除谱熵（10–125 Hz）'}

def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def span(q):return f"[{float(q['low']):.6f}, {float(q['high']):.6f}]"

def main():
    status=json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))
    assert status['status']=='COMPLETE_VERIFIED_NI_14_GROUP_FIVE_FOLD_OUTER'
    lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));arms=lock['arms'];metrics=lock['metrics']
    mainrows=table('MAIN_RESULTS');points={r['arm']:r for r in mainrows};intervals=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS');cells=table('TRAIN_CELLS')
    train=json.loads((PUB/'TRAIN_STATUS.json').read_text(encoding='utf-8'));ind=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    assert len(points)==14 and ind['status']=='PASS'
    md=['# 六特征、无头部交互主模型：完整五折 Outer 消融','',
        '主模型固定为上一版的 w/o Head Interaction，同时永久删除 Variance。保留 β、低γ、80–125 Hz 部分高γ、RMS、Line Length 和10–125 Hz谱熵，形成12个有效摘要坐标。早期 FiLM、晚期 Gate、两类 PR、Raw 发作前参考及64维隐藏单头保持不变。删除的是头部显式 Raw×生理隐藏乘积项；两分支仍经联合 GELU 映射，不能理解成模型完全没有跨分支非线性作用。','',
        f"本轮14组全部完成原五折及 Outer，合计70个选定折单元：55次真实新增训练，累计{train['epochs']}个完整训练轮次，另外15个单元为精确复用。原80人、7,635通道、256次发作、1,471,965个有效 Raw250窗口不变。所有70个模型、预处理、轮次和阈值均在新 Outer 前冻结；全部预测封存后才统一计算指标。",'',
        '主模型、Raw-only、Physiology-only 各复用五折。复用前检查初始化状态和原内部验证 logits 逐位一致，复用的 Outer 分数及阈值逐字节／哈希核验。其他变体均在无交互的新基准上从匹配的随机初始化重新训练，不沿用旧版含交互消融的结果。','',
        '## 1. 患者平均结果','',
        '每名患者先独立计算，再对80人等权平均。Macro-F1 是患者内 EZ-F1 与 NEZ-F1 的平均，不是 pooled-channel F1。EZ-Frac. Bias 是预测比例减真实比例，应按接近0解释，不能按有符号差值的正负判优。','',
        '| 配置 | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-Frac. Bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in arms:md.append('| '+LABELS[a]+' | '+' | '.join(f'{float(points[a][m]):.6f}' for m in metrics)+' |')
    md+=['','完整精度见[MAIN_RESULTS.csv](results/MAIN_RESULTS.csv)。[Pooled指标](results/POOLED_RESULTS.csv)单独列出，不代替上表。','',
        '## 2. 患者配对不确定性','',
        '差值统一为主模型减消融。以下只比较 Macro-F1，因此正值有利于主模型。固定 seed42、同一80人、10,000次患者级配对 bootstrap；13项额外区间采用 Bonferroni 水平的百分位尾概率0.05/(2×13)，属于近似探索性校正，不覆盖整个历史研究过程的选择效应。','',
        '| 对照 | Macro-F1 差值 | 名义95%区间 | 额外13项比较区间 |',
        '| --- | ---: | --- | --- |']
    for a in arms[1:]:
        label='FULL_minus_'+a;q=next(r for r in intervals if r['contrast']==label and r['metric']=='macro_f1');f=next(r for r in family if r['contrast']==label)
        md.append('| '+LABELS[a]+f" | {float(q['mean']):+.6f} | {span(q)} | {span(f)} |")
    positive=[LABELS[r['contrast'].removeprefix('FULL_minus_')] for r in family if float(r['low'])>0]
    negative=[LABELS[r['contrast'].removeprefix('FULL_minus_')] for r in family if float(r['high'])<0]
    uncertain=[LABELS[r['contrast'].removeprefix('FULL_minus_')] for r in family if float(r['low'])<=0<=float(r['high'])]
    md+=['',
        '额外13项比较区间完整高于0的对照：'+('、'.join(positive) if positive else '无')+'。',
        '额外13项比较区间完整低于0的对照：'+('、'.join(negative) if negative else '无')+'。',
        '区间包含0的对照：'+('、'.join(uncertain) if uncertain else '无')+'。这些分类只描述本轮固定模型预测下的患者配对差异，不据此重新选择主模型或开展下一轮筛选。','',
        '[全部243项名义区间](results/INTERVALS.csv)包含14组绝对结果和13组配对差值的全部九项指标；[额外13项Macro-F1区间](results/MACRO_FAMILY_INTERVALS.csv)单独提供。','',
        '本轮对两类 PR、Raw 发作前参考，以及相对 Raw-only 的显式生理增量，提供了区间未跨0的 Macro-F1 对比。相对 Physiology-only 的增量在额外13项比较区间下仍包含0，不能把两个方向的分支互补都写成已确证。早期 FiLM 删除、联合引导删除和早期错配的额外区间也包含0，当前证据不足以确证这些引导机制。','',
        f"去掉两类 PR 后，AUROC 为{float(points['WO_BOTH_PR']['auroc']):.6f}，但 Macro-F1 降为{float(points['WO_BOTH_PR']['macro_f1']):.6f}，NEZ-F1 降为{float(points['WO_BOTH_PR']['nez_f1']):.6f}，EZ比例偏差变为{float(points['WO_BOTH_PR']['ez_fraction_bias']):+.6f}。排序能力与按冻结阈值形成的错误类型并不相同；AUROC较高不能抵消F1和预测EZ比例的恶化。",'',
        f"单项特征证据中，低γ、RMS、Line Length 的 Macro-F1 条件贡献在额外区间下仍为正；β、部分高γ和谱熵尚不能排除零差异。不过删除低γ后的AP为{float(points['DROP_LOW_GAMMA']['auprc']):.6f}，高于主模型的{float(points['FULL_F0']['auprc']):.6f}。因此保留特征的依据具有指标依赖性，不能将六项组合描述为所有指标上的最佳方案。",'',
        '## 3. 模块、容量与特征解释','',
        '![核心模块配对区间](results/figures/CORE_PAIRED_INTERVALS.png)','',
        'w/o PR 同时取消14D生理摘要的患者内 Z 和最终96D Raw摘要的患者内 Z；不取消单窗口波形归一化、Raw参考扣除或原始 DELTA 定义。内部验证仍按原规则选轮次和阈值。AUROC/AP 与阈值型 F1 的变化应分别讨论，不能用排序指标相近推断分类决策相同。','',
        '错配仅改变早期 FiLM 的供体轨迹，限制在同患者、同发作、完整有效窗口掩码相同的通道组；晚期 Gate 和直接生理输入保持正确对应。参数数与主模型相同。单通道组不改变，覆盖统计见[RAW_COVERAGE.csv](results/RAW_COVERAGE.csv)。','',
        '| 配置 | 参数量 | 新增训练折数 | 精确复用折数 |',
        '| --- | ---: | ---: | ---: |']
    for a in arms:
        rr=[r for r in cells if r['arm']==a];params={int(r['parameters']) for r in rr};assert len(rr)==5 and len(params)==1
        reused=sum(r['reused']=='True' for r in rr)
        md.append(f'| {LABELS[a]} | {params.pop()} | {5-reused} | {reused} |')
    md+=['','Raw-only、Physiology-only 采用完整初始化中相应的输入权重切片、原缩放及64维隐藏头，不另加宽。分支或模块移除改变参数容量时，性能差异包含容量与优化效应，不能全部解释为信息差异。','',
        '![特征条件贡献](results/figures/FEATURE_CONTRIBUTION.png)','',
        '六项单删同步屏蔽两个摘要坐标及早期 FiLM、晚期 Gate 对应窗口坐标，并从头训练。剩余显式生理属性为五项，Variance 始终未参与。单删评估的是在其余特征和 Raw 分支存在时的条件预测贡献；Raw 可能补偿被删属性，不能将零差异写成该生理属性无效，也不能由六个单删证明当前组合全局最优。','',
        '按用户要求，本轮不做孤立 Late Gate 删除，也不做 Head Interaction 加回。因此不报告早期／晚期引导的完整四格交互。','',
        '## 4. 五折与来源稳定性','',
        '| 配置 | Macro-F1折间SD | AUROC折间SD | AP折间SD |',
        '| --- | ---: | ---: | ---: |']
    deviations={r['arm']:r for r in table('FOLD_SD')}
    for a in arms:md.append('| '+LABELS[a]+' | '+' | '.join(f'{float(deviations[a][m]):.6f}' for m in ['macro_f1','auroc','auprc'])+' |')
    md+=['','[五折全部九指标](results/FOLD_RESULTS.csv)、[全部折间样本标准差](results/FOLD_SD.csv)、[跨来源结果](results/SOURCE_RESULTS.csv)、[指标覆盖](results/METRIC_COVERAGE.csv)均独立列出。五折训练集有重叠，折间SD是描述性稳定性，不代表五次独立随机种子试验。来源患者数少于5时抑制指标，不删除这些患者。','',
        '## 5. 传统分类器：复用的六特征参考','',
        '以下七类分类器×有／无 Feature PR 来自上一版实际完成的同队列六特征实验。神经头部交互的删除不影响它们的12D输入或模型；本轮不重复训练，也不将其列入上述14组区间族。SVM校准、FIT-only预处理和原内部验证阈值保持原样。','',
        '| 分类器 | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-Frac. Bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table('HISTORICAL_CLASSIFIER_REFERENCE'):md.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in metrics)+' |')
    md+=['','这些基线沿用此前合法验证选定的固定超参数，不是对12D重新穷举优化。原10个MLP拟合均在400次迭代预算内记录 ConvergenceWarning，不能称作已证明优化收敛；本轮未据结果追加迭代。原始来源与哈希见[复用审计](results/HISTORICAL_CLASSIFIER_AUDIT.json)。','',
        '## 6. 证据边界与核验','',
        f"独立重算{ind['patient_rows']}行患者结果、{ind['patient_metric_checks']}个指标值，核验243个名义区间、13个额外区间、pooled指标、五折及来源汇总、所有有效窗口覆盖和15个复用预测。独立区间最大数值差为{ind['maximum_interval_difference']:.3g}。",'',
        '本轮主模型选择发生在既往 Outer 多次查看之后，所有结论均为探索性重复测试，不是独立确认。没有根据本轮 Outer 更改模型、阈值或筛选规则。固定一个seed不足以证明训练随机性稳定，也不能据此声称最优特征集合或外部中心泛化。','',
        '代码、匿名汇总、区间、图和执行审计上传 GitHub；原始波形、患者／通道身份、逐例预测、权重、预处理器、缓存及日志保留在可信服务器。','']
    (HERE/'FINDINGS.md').write_text('\n'.join(md),encoding='utf-8',newline='\n')
    print('NI_ACTUAL_CHINESE_FINDINGS_GENERATED_FROM_VERIFIED_AGGREGATES')

if __name__=='__main__':main()
