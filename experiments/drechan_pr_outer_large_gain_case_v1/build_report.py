"""Readable tables from authenticated anonymous selected-case results only."""
import csv,hashlib,json
from pathlib import Path
from verify_anonymous import check
ROOT=Path(__file__).resolve().parent
P=ROOT/'results'/'public';OUT=ROOT/'results'/'readable'

def read(name):
    with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    check(P)
    manifest=json.loads((P/'PUBLIC_MANIFEST.json').read_text())
    for name,v in manifest['files'].items():
        b=(P/name).read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    m=read('CASE_METADATA.csv')[0];metrics=read('CASE_METRICS.csv');channels=read('CHANNEL_SCORES_ANONYMOUS.csv')
    wide={}
    for r in channels:
        key=r['channel_id'];row=wide.setdefault(key,{k:r[k] for k in ['case_id','channel_id','truth_label','truth_ez']})
        assert row['truth_label']==r['truth_label'] and row['truth_ez']==r['truth_ez']
        prefix=f"{r['model']}_seed{r['seed']}_{r['condition']}_"
        row.update({prefix+k:r[k] for k in ['score_ez','rank_ez','prediction_ez']})
    rows=[wide[k] for k in sorted(wide)];ez=[r for r in rows if r['truth_ez']=='1']
    assert len(rows)==int(m['total_channels']) and len(ez)==int(m['ez_channels'])
    for name,rr in [('CASE-H01_CHANNELS_WIDE.csv',rows),('CASE-H01_EZ_CHANNELS_ONLY.csv',ez)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rr)
    lines=['# 大幅改善病例：CASE-H01','',
      '在原 80 名 Outer 患者的冻结预测中，按用户确认的“EZ 数量不限、优先改善幅度”筛选。没有新训练、推理或阈值调整；先前两名病例保持不变。','',
      f"匿名病例 **{m['case_id']}**，**{m['center']}**，Outer Fold **{m['outer_fold']}**；共 **{m['total_channels']}** 通道，**{m['ez_channels']} EZ**、**{m['nez_channels']} NEZ**。",'',
      '## 实际指标','',
      'DREChanNet 为原完整三 seed FULL 对比 WO_BOTH_PR（包含 early FiLM / late Gate，11,425 参数）；SVM 为 12 维特征 No PR 对比 Legacy PR。这不是 LOCO 或后续 no-guidance 模型。表中为 No PR → PR；三 seed 平均是患者端点平均，没有平均概率或选择最优 seed。','',
      '|模型|AUROC|Macro-F1|Top-1 (%)|Recall@10% (%)|Recall@20% (%)|','|---|---:|---:|---:|---:|---:|']
    order=[('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]
    names=['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct']
    for model,seed in order:
        rr={r['metric']:r for r in metrics if (r['model'],r['seed'])==(model,seed)}
        label='SVM' if model=='SVM' else ('DRE 三 seed 平均' if seed.startswith('MEAN') else 'DRE seed'+seed)
        cells=[]
        for k in names:
            fac=1 if k in names[:2] else 100;fmt='.4f' if fac==1 else '.2f'
            cells.append(f"{float(rr[k]['no_pr'])*fac:{fmt}} → {float(rr[k]['pr'])*fac:{fmt}}")
        lines.append('|'+label+'|'+'|'.join(cells)+'|')
    lines+=['',
      '在前 10% 审阅预算下实际取前 18 通道：DRE seed42/43/44 从覆盖 3/3/2 个 EZ，增加至 7/8/7 个 EZ。在前 20% 预算下取前 36 通道：从 6/6/7 个 EZ 增加至三个 seed 均覆盖全部 9 个 EZ。三个 seed 的 AUROC 与 Recall@10% 均改善。','',
      '并非所有指标都提高：DRE seed43 的 Macro-F1 从 0.6716 降至 0.6217。SVM 的 AUROC 提高，但 Recall@10% 从 33.33% 降至 22.22%，Macro-F1 从 0.6071 降至 0.4565。因此该病例适合展示 DRE 的患者内排序变化，不能声称各模型、各端点一致改善。','',
      '## 筛选规则与改善分位','',
      '计算前在本地提交并锁定协议与源码：`c42b2237ea11698f42f54c20d4ca9330162b437d`。不设 EZ 数量下限，保留至少 20 总通道且同时存在 EZ/NEZ 的要求，排除原先两名病例。要求 DRE 三 seed 平均 AUROC、Recall@10% 均改善，且至少两个 seed 两端点联合改善。SVM 不参与筛选。','',
      '合格候选共 **15 人**。锁定目标为最大化 `J = (ΔAUROC + ΔRecall@10%)/2`，两项均使用 0–1 原尺度，等权重；并列时依次比较两项改善中的较小值、同向 seed 数、固定哈希。CASE-H01 的 J=0.305167，为合格池最高；独立重算最大值一致。','',
      '|DRE 三 seed 平均端点|实际变化|全 80 人变化分位|','|---|---:|---:|']
    for k in names:
        r=next(r for r in metrics if r['model']=='DRECHANNET' and r['seed']=='MEAN_42_43_44' and r['metric']==k)
        lines.append(f"|{k}|{float(r['change']):+.6f}|{float(r['percentile_among_all80']):.3f}%|")
    lines+=['',
      '联合改善 J 在全 80 人中处于 99.375 分位。分位采用并列中位秩，最高值小于 100%。这是有意按大幅改善挑选的回顾性示例，不是中位改善或总体典型患者；9 个 EZ 中每增减 1 个都会使单 seed Recall 改变约 11.11 个百分点。','',
      '## 九个 EZ 通道的实际名次','',
      '以下为冻结预测分数下的真实名次，均使用 EZ 方向降序，同分按原始通道字典序处理。CH 为患者内匿名通道名，不代表解剖位置。','',
      '|匿名 EZ 通道|SVM No PR → PR|DRE seed42|DRE seed43|DRE seed44|','|---|---:|---:|---:|---:|']
    for r in ez:
        vals=[]
        for model,seed in order[:4]:
            prefix=f'{model}_seed{seed}_';vals.append(r[prefix+'NO_PR_rank_ez']+' → '+r[prefix+'PR_rank_ez'])
        lines.append('|'+r['channel_id']+'|'+'|'.join(vals)+'|')
    lines+=['','## 文件','',
      '- [全部 176 通道的标签、实际分数、排名与预测](readable/CASE-H01_CHANNELS_WIDE.csv)。',
      '- [9 个 EZ 通道的完整数据](readable/CASE-H01_EZ_CHANNELS_ONLY.csv)。',
      '- [五个模型/seed 汇总的全部指标及分位](public/CASE_METRICS.csv)。',
      '- [每个整数 K 的 Recall 曲线数据](public/RECALL_AT_K.csv)。',
      '- [筛选审计](public/SELECTION_AUDIT.json)、[服务器独立核验](public/INDEPENDENT_VERIFICATION.json)。','',
      '原始通道名与身份映射保留于可信服务器 `D:\\Codex_Research\\experiments\\drechan_pr_outer_large_gain_case_v1_runtime\\private` 中的 `SELECTED_CHANNELS_ORIGINAL_NAMES.csv`、`IDENTITY_MAP.json`，未下载。','',
      '服务器重核 40 个冻结预测单元、640 行患者–条件指标、400 行患者变化及 3,200 个端点，历史指标重放全部一致。本机从匿名通道表独立复算全部 25 行指标、每个通道名次与全部 1,416 行 Recall@K，全部通过。匿名包 SHA256：`683efebd83350308bfd56bb951609427bcbb6d29f9bce84c6541570676a5bf4e`。','']
    (ROOT/'results'/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print('HIGH_GAIN_READABLE_REPORT_PASS',len(rows),len(ez))

if __name__=='__main__':main()
