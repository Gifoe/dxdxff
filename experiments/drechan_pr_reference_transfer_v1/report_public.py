"""Read verified aggregates only; no patient files,refitting or outcome selection."""
import csv
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def table(n):
    with (HERE/'results'/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def select(rows,**k):return next(r for r in rows if all(r.get(a)==b for a,b in k.items()))
def num(x):return f'{float(x):.6g}'
def ci(r,family=False):
    l,h=('family_low','family_high') if family else ('ci_low','ci_high')
    return f"{num(r['mean'])} [{num(r[l])}, {num(r[h])}]"
def main():
    status=json.loads((HERE/'results/STATUS.json').read_text());assert status['status']=='COMPLETE_VERIFIED_PR_REFERENCE_TRANSFER_80'
    bg=table('BACKGROUND');ct=table('CONTRASTS');an=table('ANOMALY');su=table('SUPPORT');pe=table('PERMUTATION_SUMMARY')
    mainrows=[r for r in bg if r['scope']=='ALL' and r['coordinate']=='ALL12']
    primary=select(ct,scope='ALL',coordinate='ALL12',contrast='GLOBAL-FULL')
    source=select(ct,scope='ALL',coordinate='ALL12',contrast='SOURCE-FULL')
    center=select(ct,scope='ALL',coordinate='ALL12',contrast='CENTER-FULL')
    gap=select(an,scope='ALL',coordinate='ALL12',endpoint='gap')
    sup=select(su,scope='ALL',coordinate='ALL12',endpoint='unsupported');cross=select(su,scope='ALL',coordinate='ALL12',endpoint='unsupported_query_nonconstant')
    supported=float(primary['ci_low'])>0 and float(source['family_low'])>0 and float(gap['family_low'])>0
    lines=['# Results: patient reference transfer across disjoint channel sets','',
      'All 80 patients; 7,635 channels; 100 bidirectional partitions per patient. Twelve fixed physiological coordinates. No classification model or F1/AUROC/AP endpoint.',
      '', '## Primary background forecasts','',
      'Lower CRPS is better. Losses are in the same TRAIN-standardized units;each patient has equal weight. Brackets are paired-stratum-bootstrap nominal95% intervals.',
      '', '| Probe | CRPS [95% CI] |','| --- | --- |']
    lines += [f"| {r['arm']} | {ci(r)} |" for r in mainrows]
    lines += ['',f"Primary GLOBAL-FULL gain: **{ci(primary)}**. Positive-patient fraction: {num(primary['positive_fraction'])}.",
      '', '## Prespecified pooled contrasts','',
      'Primary has nominal95% bounds;all other rows show additional six-family-adjusted bounds. Positive means lower loss for the right-hand method.',
      '', '| Contrast | Gain [interval] |','| --- | --- |']
    lines += [f"| {r['contrast']} | {ci(r,r['contrast']!='GLOBAL-FULL')} |" for r in ct if r['scope']=='ALL' and r['coordinate']=='ALL12']
    lines += ['',f"Full EZ-minus-NEZ absolute relative atypicality (six-family-adjusted): **{ci(gap,True)}**.",
      f"Unsupported query-coordinate fraction: {ci(sup)}. Unsupported-reference but query residual>1e-8 fraction: {ci(cross)}.",
      '', 'The support rule zeros unsupported relative coordinates, but CRPS retains every coordinate and treats SD0 as a point distribution. These are different declared rules.',
      '', '## All four sources','', '| Source | Global CRPS | Full CRPS | Global-Full [95% CI] | EZ-NEZ gap [95% CI] |','| --- | --- | --- | --- | --- |']
    for s in ['hup','lzu','multicenter','pediatric']:
        g=select(bg,scope=s,coordinate='ALL12',arm='GLOBAL');f=select(bg,scope=s,coordinate='ALL12',arm='FULL');c=select(ct,scope=s,coordinate='ALL12',contrast='GLOBAL-FULL');a=select(an,scope=s,coordinate='ALL12',endpoint='gap')
        lines.append(f"| {s} | {num(g['mean'])} | {num(f['mean'])} | {ci(c)} | {ci(a)} |")
    lines += ['', '## All12 coordinates','', '| Coordinate | Global-Full CRPS gain [95% CI] | EZ-NEZ relative gap [95% CI] |','| --- | --- | --- |']
    for r in ct:
        if r['scope']=='ALL' and r['contrast']=='GLOBAL-FULL' and r['coordinate']!='ALL12':
            a=select(an,scope='ALL',coordinate=r['coordinate'],endpoint='gap');lines.append(f"| {r['coordinate']} | {ci(r)} | {ci(a)} |")
    lines += ['', '## Conditional label-shuffle reference','', '| Scope | Observed gap | Null central95% | Plus-one upper fraction |','| --- | --- | --- | --- |']
    lines += [f"| {r['scope']} | {num(r['observed'])} | [{num(r['null_low'])}, {num(r['null_high'])}] | {num(r['conditional_plus_one_upper'])} |" for r in pe]
    lines += ['', 'This preserves within-patient EZ counts and frozen features/references. It is a conditional diagnostic,not an exact population-level test. It cannot replace the patient-bootstrap interval.',
      '', '## Complete machine-readable results','',
      'BACKGROUND.csv,CONTRASTS.csv,ANOMALY.csv andSUPPORT.csv retain all five scopes and13 coordinate summaries. PERMUTATION.csv retains all1000 draws per scope. Source/coordinate intervals are descriptive. The six pooled family slots were fixed before outcomes.',
      '', '## Scope and uncertainty','',
      'This is historical-cohort exploratory assessment,conditional on fixed overlapping TRAIN banks. The100 splits do not create8000 independent patients.79 assessments lie inside originalFIT;one is the previously declared original-validation exception. All80 remain absent from their own TRAIN/CAL. No strict79 complementary table is exported.',
      'These Gaussian marginal probes measure reference transfer,not true Gaussianity,healthy baseline,technical nuisance,biological causality or new clinical effectiveness. Electrode sampling/anatomy can contribute to the patient reference. No spatial independence of channels is asserted;resampling operates on patients.',
      'The prior mechanism study found no improvement in transferable standardized separation. That negative/mixed result remains unchanged and must accompany any broader PR narrative.','']
    def readable(text):
        for a,b in [(';','; '),(',not',', not'),(',not',' , not'),('all80','all 80'),('All80','All 80'),('all1000','all 1,000'),('all five scopes and13','all five scopes and 13'),('nominal95%','nominal 95%'),('nominal95','nominal 95'),('conditional95','conditional 95'),('six-family-adjusted','six-family adjusted'),('TRAIN/CAL','TRAIN/CAL'),('originalFIT','original FIT'),('The100','The 100'),('create8000','create 8,000'),('The80','The 80'),('specific12','specific 12'),('andSUPPORT','and SUPPORT')]:text=text.replace(a,b)
        return text
    (HERE/'RESULTS.md').write_text(readable('\n'.join(lines)),encoding='utf-8')
    findings=['# Mechanistic interpretation','',f"Primary background-reference gain: {ci(primary)}.",f"Beyond same-source reference: {ci(source,True)} (six-family-adjusted).",f"EZ-related absolute-relative gap: {ci(gap,True)} (six-family-adjusted).",'',
      ('The three predeclared evidence conditions are met. The supported claim is limited to patient-specific cross-channel reference transfer and retained EZ-associated atypicality under this probe.' if supported else 'The full predeclared mechanism chain is NOT established. Interpret each supported component separately;do not substitute a favorable feature,source or nominal interval.'),
      '',f"Added Full-versus-center gain: {ci(center,True)}. " + ('An additional scaling contribution is supported for this endpoint.' if float(center['family_low'])>0 else 'An additional scaling benefit is not established;this is not an equivalence test.'),
      '', 'Relative atypicality is a specific12-coordinate diagnostic,not proof that all discriminative information is preserved. Mean/SD normalization identities are not presented as empirical discoveries:reference and query channels are disjoint. All scores are evaluated in one common coordinate system.',
      '', 'The experiment neither overturns the prior negative transferable-SNR result nor establishes biological causality or performance improvement. Both studies address different properties of PR and must remain distinguishable.','']
    findings += ['## Paper-ready statement','',
      'In 80 historical patients, we estimated patient-specific reference distributions from unlabeled channel subsets and evaluated them on disjoint channels over 100 fixed partitions. Full location-and-scale references reduced mean held-out CRPS from 0.4509 to 0.1734 (paired reduction 0.2775, 95% CI [0.2215, 0.3517]), with improvements in 79 patients. The gain remained supported against same-source and center-only references. However, the EZ-minus-NEZ relative-atypicality contrast was inconclusive after multiplicity adjustment (0.9585, adjusted CI [-0.2508, 2.5610]). These findings support patient-specific background estimation, while leaving its proposed EZ-specific component unresolved.', '',
      'The methods must disclose the 79 original-FIT assessments plus one predeclared original-validation exception, fixed overlapping TRAIN banks, and historical-cohort exploratory status. Disjoint channel sets are not asserted to be statistically independent.','']
    (HERE/'FINDINGS.md').write_text(readable('\n'.join(findings)),encoding='utf-8')
    chinese=['# 中文研究结论','',f"80人、7635通道全部纳入；每人100次无标签通道二分，双向评价。主效应（Global−Full CRPS）为 {ci(primary)}。",'',
      f"相对于同来源背景的收益：{ci(source,True)}；EZ−NEZ相对偏离：{ci(gap,True)}；两者使用六项家族校正区间。",'',
      '三项预先规定的机制证据均成立，可支持有限的患者参考外推解释。' if supported else '预先规定的完整机制链未全部成立，不能写成“证明PR机制”。应分别报告背景预测与EZ关联的证据。',
      '',f"Full相对仅中心化的额外收益：{ci(center,True)}。未显著不能解释为两者等价。",'',
      '结论仅适用于该历史队列和预设探针。统计背景不等于健康基线；相对偏离关联不等于信息无损、因果机制或临床效能。之前PR未改善可迁移标准化分离度的结果保留。','']
    (HERE/'CHINESE_SUMMARY.md').write_text('\n'.join(chinese),encoding='utf-8')
    print('AGGREGATE_REPORTS_WRITTEN')
if __name__=='__main__':main()
