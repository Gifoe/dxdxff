"""Build readable deliverables only from authenticated anonymous case tables."""
import csv, hashlib, json
from pathlib import Path
from verify_anonymous import check

ROOT=Path(__file__).resolve().parent
PUBLIC=ROOT/'results'/'public'
OUT=ROOT/'results'/'readable'

def read(name):
    with (PUBLIC/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    check(PUBLIC)
    OUT.mkdir(parents=True,exist_ok=True)
    channels=read('CHANNEL_SCORES_ANONYMOUS.csv')
    metrics=read('CASE_METRICS.csv')
    meta=read('CASE_METADATA.csv')
    audit=json.loads((PUBLIC/'SELECTION_AUDIT.json').read_text())
    manifest=json.loads((PUBLIC/'PUBLIC_MANIFEST.json').read_text())
    for name,entry in manifest['files'].items():
        b=(PUBLIC/name).read_bytes()
        assert hashlib.sha256(b).hexdigest()==entry['sha256'] and len(b)==entry['bytes']
    configs=[('SVM','42')]+[('DRECHANNET',str(s)) for s in (42,43,44)]
    for case in meta:
        cid=case['case_id'];lookup={}
        for r in channels:
            if r['case_id']!=cid:continue
            key=r['channel_id']
            dest=lookup.setdefault(key,dict(case_id=cid,channel_id=key,truth_label=r['truth_label'],truth_ez=r['truth_ez']))
            assert dest['truth_label']==r['truth_label'] and dest['truth_ez']==r['truth_ez']
            prefix=f"{r['model']}_seed{r['seed']}_{r['condition']}"
            for col in ('score_ez','rank_ez','prediction_ez'):dest[prefix+'_'+col]=r[col]
        rows=[lookup[k] for k in sorted(lookup)]
        assert len(rows)==int(case['total_channels'])
        with (OUT/(cid+'_CHANNELS_WIDE.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 冻结 Ictal Outer 预测：PR 改善病例与下降对照','',
      '已完成原始 80 人、7,635 通道的只读重分析。使用 40 个已封存预测单元，不重新训练、推理或选择阈值。所有结果均来自真实冻结分数。','',
      '本次 SVM 为 12 维生理特征 No PR / Legacy PR；DREChanNet 为原三 seed 完整模型 WO_BOTH_PR / FULL（11,425 参数，包含 early FiLM 与 late Gate）。这里不是后续 no-guidance 模型，也不是 LOCO 结果。','',
      '## 两名病例','',
      '|匿名病例|来源|Outer Fold|总通道|EZ|NEZ|DRE 两指标同向 seed 数|',
      '|---|---|---:|---:|---:|---:|---:|']
    for r in meta:
        lines.append('|'+ '|'.join(r[k] for k in ['case_id','center','outer_fold','total_channels','ez_channels','nez_channels','dre_joint_consistent_seeds'])+' / 3|')
    lines+=['','Fold 使用父实验原编号。CH0001 等为病例内通道别名，保持原始通道名称的字典序，确保同分排序可复现；它们不是解剖位置。','',
      '## 实际指标','',
      '表中为 No PR → PR；AUROC、Macro-F1 为 0–1，Top-1 和 Recall 为百分比。DRE 平均指逐 seed 计算患者指标后取平均，未平均概率。Recall@10%/20% 的 K 向上取整；CASE-P01 对应 K=12/23，CASE-N01 对应 K=13/25。','']
    names=['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct']
    for case in meta:
        cid=case['case_id'];lines += [f'### {cid}','','|模型|AUROC|Macro-F1|Top-1 (%)|Recall@10% (%)|Recall@20% (%)|','|---|---:|---:|---:|---:|---:|']
        for model,seed in configs+[('DRECHANNET','MEAN_42_43_44')]:
            rr={r['metric']:r for r in metrics if (r['case_id'],r['model'],r['seed'])==(cid,model,seed)}
            label='SVM' if model=='SVM' else ('DRE 三 seed 平均' if seed.startswith('MEAN') else 'DRE seed'+seed)
            vals=[]
            for m in names:
                fac=1 if m in names[:2] else 100
                fmt='.4f' if fac==1 else '.2f'
                vals.append(f"{float(rr[m]['no_pr'])*fac:{fmt}} → {float(rr[m]['pr'])*fac:{fmt}}")
            lines.append('|'+label+'|'+'|'.join(vals)+'|')
        lines.append('')
    lines += ['CASE-P01：DRE seed43/44 的 AUROC 与 Recall@10% 均改善；seed42 的 AUROC 下降 0.0960、Recall@10% 提高 3.125 个百分点。三个 seed 的 Recall@10% 均改善，但不能称为三个 seed 全指标一致。', '',
      'CASE-N01：SVM 与 DRE 平均的 AUROC、Recall@10%、Recall@20% 均下降。DRE 三个 seed 的 AUROC 均下降，Recall@10% 在 seed43/44 下降、seed42 持平；Macro-F1 三个 seed 均提高。因此这是排序下降对照，不是所有指标都下降的病例。','',
      '## 选择依据与改善分位','',
      f"80 人中 {audit['basic_eligible']} 人满足至少 20 通道、至少 3 EZ 且存在 NEZ；{audit['positive_both_models_before_seed_stability']} 人的 SVM 和 DRE 平均 AUROC / Recall@10% 均改善，进一步要求至少两个 DRE seed 两指标同时改善后剩 {audit['positive_stable_candidates']} 人。",'',
      '选择规则在计算候选结果前提交：将 SVM AUROC、SVM Recall@10%、DRE 平均 AUROC、DRE 平均 Recall@10% 四项变化转换成候选池内百分位，选择距 50 分位平均绝对距离最小者。CASE-P01 的距离为 16.67 个百分位点。选择目标是四指标联合居中，并不保证每个指标的原始增量都接近中位数。','',
      '|选择坐标|改善候选池中位增量|CASE-P01 增量|候选池内分位|全 80 人分位|',
      '|---|---:|---:|---:|---:|']
    coordinates=[('SVM','auroc'),('SVM','recall_at_10pct'),('DRECHANNET','auroc'),('DRECHANNET','recall_at_10pct')]
    for i,(model,m) in enumerate(coordinates):
        r=next(r for r in metrics if r['case_id']=='CASE-P01' and r['model']==model and r['metric']==m and r['seed']==('42' if model=='SVM' else 'MEAN_42_43_44'))
        lines.append(f"|{audit['selection_coordinate_order'][i]}|{audit['positive']['pool_medians'][i]:+.6f}|{float(r['change']):+.6f}|{float(r['percentile_among_case_pool']):.2f}%|{float(r['percentile_among_all80']):.2f}%|")
    lines += ['',
      '这里的百分位按从负到正的变化排序，采用并列中位秩；高百分位表示改善更大。CASE-P01 是改善候选池中的联合居中案例，在全 80 人中仍属于改善较大的部分，不能称为总体典型患者。尤其 SVM AUROC 增量 +0.433689 高于候选池中位数 +0.152872，应保留这个差别。','',
      f"下降对照使用优先级 1：两模型 AUROC / Recall@10% 均下降且至少两个 DRE seed 联合下降，共 {audit['negative']['pool_size']} 名候选；按同一居中规则选择 CASE-N01。两个候选的中位秩距离相同，依次按 seed 一致性、固定哈希打破并列。该池仅两人，居中代表性的分辨率有限。",'',
      '|下降对照坐标|增量|全 80 人分位|下降候选池内分位|','|---|---:|---:|---:|']
    for i,(model,m) in enumerate(coordinates):
        r=next(r for r in metrics if r['case_id']=='CASE-N01' and r['model']==model and r['metric']==m and r['seed']==('42' if model=='SVM' else 'MEAN_42_43_44'))
        lines.append(f"|{audit['selection_coordinate_order'][i]}|{float(r['change']):+.6f}|{float(r['percentile_among_all80']):.2f}%|{float(r['percentile_among_case_pool']):.2f}%|")
    lines += ['', '## 数据文件','',
      '- [全部病例指标与分位](public/CASE_METRICS.csv)：50 行，包含每个 seed 和三 seed 指标平均。',
      '- [改善病例全部通道](readable/CASE-P01_CHANNELS_WIDE.csv)：114 行，真实 EZ/NEZ 标签、SVM 和三个 DRE seed 的两条件实际分数、排名与冻结阈值分类。',
      '- [下降对照全部通道](readable/CASE-N01_CHANNELS_WIDE.csv)：122 行，同上。',
      '- [逐通道长表](public/CHANNEL_SCORES_ANONYMOUS.csv)：1,888 行，236 通道 × 8 模型/条件配置。',
      '- [完整 Recall@K 曲线](public/RECALL_AT_K.csv)：1,904 行，每个模型/seed/条件覆盖 K=0 至全部通道；可直接画真实阶梯曲线。',
      '- [筛选审计](public/SELECTION_AUDIT.json)、[全部 80 人变化分布](public/ALL80_CHANGE_DISTRIBUTION.csv)、[总体指标](public/COHORT_AGGREGATES.csv)。','',
      '服务器私有目录：`D:\\Codex_Research\\experiments\\drechan_pr_outer_representative_cases_v1_runtime_run02\\private`。其中 `SELECTED_CHANNELS_ORIGINAL_NAMES.csv` 保留原始通道名，`IDENTITY_MAP.json` 保留病例身份映射，`ALL80_PATIENT_CHANGES.csv` 保留全部 80 人患者级变化。这些文件未下载、未公开。','',
      '## 核验','',
      '服务器独立核验：40 预测单元、640 患者–条件行、400 患者变化行、3,200 端点值；640 行历史患者指标重放一致，父输入哈希保持不变。本机从去标识化通道分数重新计算两名病例的全部 50 行指标、全部排名和每一个 K 的曲线，全部通过。','',
      '病例选择是回顾性描述，不能用两名病例推断总体 PR 效果。这里不添加病例级置信区间，也不对离散 Recall@K 数据进行改变实测点的曲线调整。','']
    (ROOT/'results'/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    receipt=dict(status='PASS',package_sha256='7920a35660cd4d6b5ce93ae3a8e07307e60ce3793788621dd7c3b5e3b0e689dc',authenticated_payloads=len(manifest['files']),anonymous_channel_rows=len(channels),case_metric_rows=len(metrics),wide_case_channel_counts={r['case_id']:int(r['total_channels']) for r in meta},independent_selected_metric_and_all_k_reconstruction=True)
    (OUT/'LOCAL_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('READABLE_CASE_DELIVERABLES_PASS',json.dumps(receipt))

if __name__=='__main__':main()
