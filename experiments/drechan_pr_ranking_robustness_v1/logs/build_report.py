"""Build the research report from verified anonymous tables, retaining negatives."""
from pathlib import Path
import csv,json,argparse
ROOT=Path(__file__).resolve().parent.parent
NAMES={'LINEAR_SVM':'Linear SVM','TIMECONV':'TimeConv-CNN','DRECHANNET':'DREChanNet'}
def read(pub,n):
    with (pub/(n+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def signed(x):return f'{100*float(x):+.2f}'
def interval(r,adjusted=False):
    a,b=('adjusted_low','adjusted_high') if adjusted else ('low','high')
    return f"[{signed(r[a])}, {signed(r[b])}]"
def build(pub):
    status=json.loads((pub/'STATUS.json').read_text(encoding='utf-8'));done=status['D']=='COMPLETE'
    a={(r['model'],r['condition'],r['metric']):r for r in read(pub,'TABLE_A')};b={(r['model'],r['contrast'],r['metric']):r for r in read(pub,'TABLE_B')}
    out=['# PR排序、稳定性、审阅效率与发作数量：研究报告','',f"状态：{'A–D全部计算完成并通过独立复核' if done else 'A–C完成并通过独立复核；D尚未完成'}。",'',
         '## 数据与版本','',
         'A–C使用全部80人、7,635通道、256次发作及原五折患者外划分。所有指标先在患者内计算，再对患者等权平均。来源为HUP 36、LZU 21、多中心15、儿童8人。',
         '', '模型、检查点和开发集阈值均固定。SVM与DREChanNet复用已独立复核的提前outer版本；No PR、Full PR为原封存输出，DREChanNet Score-Z的原生logit重放使用FP32 CPU。TimeConv本轮复用已完成的发作期端到端No PR/Full PR模型；当前仍在训练的Global/Center-only/Scale-only条件不参与本轮核心模型选择。',
         '', 'TimeConv的端到端PR与JBHI原表中的冻结骨干PR残差读出属于不同版本。当前结果不替换原稿图表。该编码器沿用Omni-TimeConv架构实现，但此处权重来自本队列的发作期训练，未使用第二个发作间期benchmark的冻结权重。',
         '', '## A. 患者内排序','',
         '|模型|AUROC变化(pp)|AP变化(pp)|AUROC正向患者|AP正向患者|','|---|---:|---:|---:|---:|']
    for model,name in NAMES.items():
        u,v=b[model,'FULL_MINUS_NO_PR','auroc'],b[model,'FULL_MINUS_NO_PR','auprc']
        out.append(f"|{name}|{signed(u['mean'])} {interval(u)}|{signed(v['mean'])} {interval(v)}|{100*float(u['positive_fraction']):.2f}%|{100*float(v['positive_fraction']):.2f}%|")
    out += ['', '最终GPU主流程已与提前CPU版逐患者核对：1,120行、10,080个终点值全部逐值一致，最大差值为零。版本记录仍保留。', '',
            '上表区间为名义95%区间；预先声明的12项家族调整区间见Table B。SVM的AP增益通过该家族校正，AUROC的12项调整区间为[-0.19,+11.70]个百分点、跨零；既有六项Full PR−Score-Z家族中的AUROC区间为[+0.26,+11.21]。两套预先声明的判断均保留。DREChanNet的平均AUROC/AP差异区间跨零，且AUROC仅45%的患者改善，不能表述为患者内排序普遍提升。TimeConv端到端PR的两项排序点估计均下降，完整保留。',
            '', 'DREChanNet Hit@3由80.00%变为87.50%，Hit@5由86.25%变为90.00%；这些属于预先列出的描述性排序结果。Score-Z的AUROC、AP、MRR及Hit@1/3/5逐患者与No PR完全相同，sanity check通过。完整9条件×8项排名/分类指标见Table A。',
            '', '## 输出标准化能够解释多少分类增量','',
            '|模型|Full PR − No PR Macro-F1(pp)|Score-Z − No PR(pp)|Full PR − Score-Z(pp)|后者的六项调整区间|','|---|---:|---:|---:|---|']
    for model,name in NAMES.items():
        n=float(a[model,'NO_PR','macro_f1']['mean']);f=float(a[model,'FULL_PR','macro_f1']['mean']);z=float(a[model,'SCORE_Z','macro_f1']['mean']);r=b[model,'FULL_MINUS_SCORE_Z','macro_f1']
        out.append(f'|{name}|{100*(f-n):+.2f}|{100*(z-n):+.2f}|{signed(r["mean"])}|{interval(r,True)}|')
    out += ['', 'SVM的Full PR相对Score-Z在Macro-F1与AUROC上均有正向家族调整区间。DREChanNet中，输出标准化已恢复大部分相对No PR的分类点估计增量，Full PR额外差异未获支持。TimeConv的Score-Z分类点估计高于端到端Full PR，六项校正后差异区间仍跨零。此处比较为同一冻结模型的输出对齐对照，不把点估计增量分解当作因果归因。',
            '', '## B. 患者与来源稳定性','',
            'Table B完整报告正向、无变化和负向患者比例，以及平均配对增益和区间。Table C分别提供Macro-F1、AUROC、AP的四来源结果、CM与Worst。CM为四来源等权均值，Worst逐指标取四来源最小值。来源内方向和区间另见TABLE_B_SOURCE.csv。',
            '', '这些是普通患者外预测的来源分组。已有LOSO/LOCO实验的五臂与本轮六个核心条件不完全匹配，因此本轮不把其合并为匹配的六条件LOSO表，也不将普通来源分组称为LOSO。',
            '', '## C. 固定通道审阅预算','',
            '|模型|审阅5%时EZ覆盖率增量(pp)|审阅10%时增量(pp)|审阅20%时增量(pp)|','|---|---:|---:|---:|']
    cr={(r['model'],float(r['budget'])):r for r in read(pub,'TABLE_D_PAIRED') if r['budget_kind']=='fraction' and r['metric']=='recall'}
    for model,name in NAMES.items():out.append('|'+name+'|'+'|'.join(signed(cr[model,q]['mean']) for q in [.05,.1,.2])+'|')
    out += ['', 'SVM在5%审阅预算下的增量通过九项家族校正；10%/20%区间在校正后跨零。DREChanNet三个预算的差异均未获支持，TimeConv三个预算的覆盖率点估计均下降。全部Recall、Precision、Hit、NEZ数量、Lift及实际审阅通道数见Table D和CSV；固定1/3/5通道结果也完整保留。Lift逐患者按Precision/EZ比例计算后等权平均。',
            '', '覆盖曲线评价已有临床EZ标签的审阅效率，不作手术结局预测。',
            '', '## D. 少发作冻结推断','']
    if done:
        ad=json.loads((pub/'D_ADMISSION.json').read_text(encoding='utf-8'));iv=json.loads((pub/'D_VERIFICATION.json').read_text(encoding='utf-8'))
        out += [f"固定{ad['included_patients']}名至少有3次有效发作的患者，来源人数为HUP 27、LZU 15、多中心9、儿童6。原有{ad['original_channels']:,}个标注通道中，{ad['excluded_channels']}个未在全部候选发作中共同有效，保留{ad['common_channels']:,}个通道；所有K和模型使用同一通道集合。7名单发作和16名双发作患者不进入这项匹配的数量敏感性分析，仍完整保留在A–C。",'',
                '每名患者的K=1和K=2均穷举全部发作组合；本队列最多8次发作，组合数均未超过50。All只有一个独特子集。每次重建生理特征均值/总体SD、原始表征及患者参考；K=1的跨发作SD为零。先对每次子集的指标计算，再在患者内平均。', '',
                f"原始14维特征在全部80人的All重建逐值一致。神经表征的缓存All与原始直接前向表征和logit逐值一致；共{iv['patient_model_cells']}个患者—模型单元、{iv['repetition_rows']}行重复终点已封存并复核。推断使用CPU FP32，权重、FIT预处理器和原验证阈值固定。",'',
                '|模型 / 条件|K=1 Macro-F1 / AUROC|K=2 Macro-F1 / AUROC|All Macro-F1 / AUROC|','|---|---:|---:|---:|']
        e={(r['model'],r['condition'],r['K'],r['metric']):r for r in read(pub,'TABLE_E')}
        for model,name in NAMES.items():
            for arm,label in [('NO_PR','No PR'),('FULL_PR','Full PR')]:out.append('|'+name+' / '+label+'|'+'|'.join(f"{100*float(e[model,arm,K,'macro_f1']['mean']):.2f} / {100*float(e[model,arm,K,'auroc']['mean']):.2f}" for K in ['1','2','All'])+'|')
        out += ['', 'Table E保留全部8项指标及患者有效数；配对差值覆盖两个条件下的K=1/2相对All，以及K=1/2下的Full PR相对No PR。Macro-F1/AUROC按预先固定的36项家族校正，其余指标提供描述性名义区间。', '',
                '|模型|发作数|Full PR − No PR Macro-F1(pp)，36项调整区间|Full PR − No PR AUROC(pp)，36项调整区间|','|---|---:|---:|---:|']
        pairs={(r['model'],r['contrast'],r['metric']):r for r in read(pub,'TABLE_E_PAIRED')}
        for model,name in NAMES.items():
            for k in ['1','2']:
                label='FULL_MINUS_NOPR_K'+k;f,u=pairs[model,label,'macro_f1'],pairs[model,label,'auroc']
                out.append(f"|{name}|{k}|{signed(f['mean'])} {interval(f,True)}|{signed(u['mean'])} {interval(u,True)}|")
        d1,d2=pairs['DRECHANNET','FULL_PR_K1_MINUS_ALL','auroc'],pairs['DRECHANNET','FULL_PR_K2_MINUS_ALL','auroc']
        out += ['', 'SVM和DREChanNet在只有1次或2次发作时仍有明确的Macro-F1收益；两者的对应AUROC差异经36项校正后均未获支持。TimeConv在这两个K下的分类与排序差异均未获支持。', '',
                f"DREChanNet Full PR的AUROC相对All在K=1变化{signed(d1['mean'])}个百分点、K=2变化{signed(d2['mean'])}个百分点，36项调整区间分别为{interval(d1,True)}和{interval(d2,True)}。因此，分类收益仍然存在，同时减少发作输入会降低其排序能力。原始输入改变和跨发作SD缺失均包含在此敏感性中；本实验没有把这些因素分离为独立因果效应。", '',
                '本实验刻画已训练模型在推断时减少发作输入的敏感性，不估计按少发作方案重新训练后的性能；平均点估计接近All不解释为统计等价。']
    else:out += ['尚未完成。已固定57人的共同通道和发作子集，正在冻结模型推断；没有以A–C结果或部分患者结果代替D。']
    out += ['', '## 主文与补充材料的证据边界','',
            '- 主文可直接展示：SVM的患者内排序增益、DREChanNet相对No PR的分类结果、输出标准化对照，以及完整的六条EZ覆盖曲线。Full PR优于Score-Z的明确支持目前来自SVM。',
            '- 患者方向、四来源、固定通道预算和发作数量分析适合完整补充表；对原文结论有决定作用的反例和不支持结果应在主文概括。',
            '- 当前证据不支持“PR对三种架构都普遍提高排序”或“DREChanNet的中间表征PR已证明优于输出后处理”。端到端TimeConv的负向结果及DREChanNet未获支持的排序差异均保留。',
            '- 未根据测试结果选择运行、患者、来源、发作组合或统计校正方式；原JBHI结果未被替换。',
            '', '## 图表与复现','',
            'Table A：排名与分类；Table B：配对增益/方向及输出对齐；Table C：来源、CM、Worst；Table D：审阅预算；Table E：少发作敏感性。每张表均有Markdown和LaTeX版本。',
            '', 'Figure A：AUROC/AP配对增益按来源的分布。密度从固定5个百分点分箱平滑，截断于实际占用分箱范围；圆点与竖线为实际来源均值和名义95%患者bootstrap区间。Figure B：六条覆盖曲线及预先设定的0–20%局部视图，阴影为Full PR名义95%区间。Figure C：同一匹配队列的发作数量曲线。PDF/SVG保留矢量文本，PNG为600 dpi。',
            '', '统计始终以患者为重采样单位，来源内配对bootstrap 10,000次；6、12、9、36项检验家族分别进行Bonferroni校正。单类/无EZ边界保留缺失并报告有效患者数，未将缺失终点补零。',
            '', '可信服务器保留逐患者终点、配对差值、预测和抽样账本。本地/仓库仅包含匿名汇总和图表。科学源码、模型/预测哈希与包SHA见manifest及执行审计。', '']
    (ROOT/'report.md').write_text('\n'.join(out),encoding='utf-8',newline='\n');print('REPORT_BUILT',status['status'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,default=ROOT/'results/public_abc');build(p.parse_args().public)
