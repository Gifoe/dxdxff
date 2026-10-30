"""Readable TP/FP tradeoff report from authenticated anonymous scores only."""
import csv,hashlib,json,math
from pathlib import Path
from verify_anonymous import check
ROOT=Path(__file__).resolve().parent;P=ROOT/'results'/'public';OUT=ROOT/'results'/'readable'

def read(name):
    with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    check(P)
    manifest=json.loads((P/'PUBLIC_MANIFEST.json').read_text())
    for name,r in manifest['files'].items():
        b=(P/name).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    meta=read('CASE_METADATA.csv')[0];metrics=read('CASE_METRICS.csv');cm=read('CASE_CONFUSION.csv');ch=read('CHANNEL_SCORES_ANONYMOUS.csv')
    audit=json.loads((P/'SELECTION_AUDIT.json').read_text());wide={}
    for r in ch:
        row=wide.setdefault(r['channel_id'],{k:r[k] for k in ['case_id','channel_id','truth_label','truth_ez']})
        assert row['truth_ez']==r['truth_ez']
        prefix=f"{r['model']}_seed{r['seed']}_{r['condition']}_"
        row.update({prefix+k:r[k] for k in ['score_ez','rank_ez','prediction_ez']})
    rr=[wide[k] for k in sorted(wide)];ez=[r for r in rr if r['truth_ez']=='1'];OUT.mkdir(parents=True,exist_ok=True)
    assert len(rr)==int(meta['total_channels']) and len(ez)==int(meta['ez_channels'])
    for name,rows in [('CASE-F01_CHANNELS_WIDE.csv',rr),('CASE-F01_EZ_CHANNELS_ONLY.csv',ez)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rows)
    lines=['# 新增 EZ 检出与假阳性控制：CASE-F01','',
      f"**{meta['center']}，Outer Fold {meta['outer_fold']}，共 {meta['total_channels']} 个通道，其中 {meta['ez_channels']} 个 EZ、{meta['nez_channels']} 个 NEZ。**",'',
      '使用原始冻结验证阈值，不调阈值、不重新训练或推理、不平均预测概率。DRE 为原完整 FULL 对比 WO_BOTH_PR（含 early FiLM / late Gate，11,425 参数），SVM 为 12 维特征 No PR 对比 Legacy PR；不属于 LOCO 或后续 no-guidance 模型。','',
      '## 原阈值下的实际检出与假阳性','',
      'TP：正确检出的 EZ；FP：误报为 EZ 的 NEZ；FN：漏检的 EZ；TN：正确排除的 NEZ。表内均为 No PR → PR。平均行是三个 seed 的患者端点平均，计数可为小数，不表示合并了一个模型。','',
      '|模型|TP|FP|FN|TN|精确率 (%)|敏感度 (%)|','|---|---:|---:|---:|---:|---:|---:|']
    configs=[('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]
    for model,seed in configs:
        r={r['condition']:r for r in cm if (r['model'],r['seed'])==(model,seed)}
        label='SVM' if model=='SVM' else ('DRE 三 seed 平均' if seed.startswith('MEAN') else 'DRE seed'+seed)
        vals=[]
        for k in ['tp','fp','fn','tn','precision','sensitivity']:
            fac=100 if k in ['precision','sensitivity'] else 1;fmt='.2f' if fac==100 or seed.startswith('MEAN') else '.0f'
            vals.append(f"{float(r['NO_PR'][k])*fac:{fmt}} → {float(r['PR'][k])*fac:{fmt}}")
        lines.append('|'+label+'|'+'|'.join(vals)+'|')
    lines+=['',f"所选病例三个 seed 平均 TP 变化 **{audit['mean_tp_change']:+.3f} 个**，FP 变化 **{audit['mean_fp_change']:+.3f} 个**；每个 seed 的最大 FP 增量为 **{audit['maximum_fp_change_across_seeds']} 个**。",'',
      '## 完整排序与分类指标','',
      '|模型|AUROC|Macro-F1|Top-1 (%)|Recall@10% (%)|Recall@20% (%)|','|---|---:|---:|---:|---:|---:|']
    for model,seed in configs:
        r={r['metric']:r for r in metrics if (r['model'],r['seed'])==(model,seed)}
        label='SVM' if model=='SVM' else ('DRE 三 seed 平均' if seed.startswith('MEAN') else 'DRE seed'+seed)
        vals=[]
        for k in ['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct']:
            fac=1 if k in ['auroc','macro_f1'] else 100;fmt='.4f' if fac==1 else '.2f'
            vals.append(f"{float(r[k]['no_pr'])*fac:{fmt}} → {float(r[k]['pr'])*fac:{fmt}}")
        lines.append('|'+label+'|'+'|'.join(vals)+'|')
    n=int(meta['total_channels']);lines+=['',f"固定审阅预算的 K 分别为 ceil(10%×{n})={math.ceil(.1*n)}、ceil(20%×{n})={math.ceil(.2*n)}。这些排序预算结果与原验证阈值下的 TP/FP 是两种不同评价。",'',
      '## 筛选过程与适用解释','',
      '这是按用户提出的“新增 EZ 检出、FP 只小幅增加”寻找的回顾性病例，保留此前三个病例。首轮 V1 额外要求精确率不下降及稳定排序改善，合格病例数为 0，记录保留。随后全体 TP/FP 诊断发现：仅按 TP 增加且各 seed FP 最多增加 2 个，可得到 6 个候选。因此 V2 在匿名病例识别与导出前，明确取消了额外的精确率和排序准入限制。这个调整发生在查看聚合诊断之后，不能称为前瞻性验证。','',
      'V2 要求：至少 20 总通道、同时有 EZ/NEZ、不设 EZ 数量下限、排除前三名病例；DRE 平均 TP 增加，且至少两个 seed 的 TP 增加。先尝试各 seed FP 均不增加；没有候选时允许每个 seed 最多增加 2 个 FP。在首个非空层中选择平均新增 TP 最多者，随后以平均 FP 增量更小、排序增益、固定哈希打破并列。所有其他指标保留原值。','',
      f"实际层级候选数为 **{audit['tier_counts']}**，选择第 **{audit['selected_tier']}** 层，候选 **{audit['candidates']} 人**。选中病例在该集合中平均 TP 增益最高，独立重算确认。",'',
      '这里的“取舍最好”指上述有限 FP 增量约束下 TP 增益最大，不表示临床上 TP 与 FP 的代价已经得到验证，也不表示它代表总体平均效果。三个 seed 的所有变化在表中完整保留。','',
      '## 数据与核验','',
      '- [完整 TP/FP/FN/TN、精确率、敏感度、FPR](public/CASE_CONFUSION.csv)。',
      '- [全部通道：真实标签、原始冻结分数、预测与排名](readable/CASE-F01_CHANNELS_WIDE.csv)。',
      '- [全部 EZ 通道数据](readable/CASE-F01_EZ_CHANNELS_ONLY.csv)。',
      '- [全部指标及改善分位](public/CASE_METRICS.csv)。',
      '- [完整 Recall@K 曲线](public/RECALL_AT_K.csv)。','',
      '独立本机核验重建全部 25 行原指标、10 行混淆矩阵及相关比例、每个通道名次和全部 K 曲线，均通过。所有公开文件均先核验允许清单与 SHA256。','',
      'V2 首次复算在父级 endpoint 函数出现原生访问异常，失败目录保留。工程恢复保持同一冻结 V2 筛选规则与阈值，在独立目录中用纯 Python 端点及混淆矩阵公式复算；准入输入读取仍复用原实现。全部 640 行原患者端点、400 行变化表逐项重放一致。异常根因尚未确定，未通过改阈值或改病例规则恢复。','',
      '身份映射与原始通道名仅保留在可信服务器 `D:\\Codex_Research\\experiments\\drechan_pr_outer_controlled_fp_case_v2_runtime_pure01\\private` 中，未下载。','']
    (ROOT/'results'/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print('CONTROLLED_FP_REPORT_AND_WIDE_TABLES_PASS',len(rr),len(ez))

if __name__=='__main__':main()
