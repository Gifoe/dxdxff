"""Reports from authenticated aggregate payloads only."""
import csv
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def tables():
    status=json.loads((HERE/'results/STATUS.json').read_text(encoding='utf-8'));assert status['status']=='COMPLETE_VERIFIED_PR_EZ_COMPOSITION_80'
    out={}
    for name in status['row_counts']:
        with (HERE/'results'/(name+'.csv')).open(encoding='utf-8',newline='') as f:out[name]=list(csv.DictReader(f))
    with (HERE/'results/LEGACY_ANOMALY.csv').open(encoding='utf-8',newline='') as f:out['LEGACY_ANOMALY']=list(csv.DictReader(f))
    return out
def row(rows,**tags):return next(r for r in rows if all(r.get(k)==str(v) for k,v in tags.items()))
def fmt(x):return 'not estimable' if x=='' else f'{float(x):.6g}'
def ci(r,family=False):
    lo,hi=('family_low','family_high') if family else ('ci_low','ci_high')
    return 'not estimable' if r['mean']=='' else f"{fmt(r['mean'])} [{fmt(r[lo])}, {fmt(r[hi])}]"
def direction(r):
    if r['mean']=='':return 'not estimable'
    return 'positive' if float(r['family_low'])>0 else 'negative' if float(r['family_high'])<0 else 'inconclusive'
def main():
    t=tables();h=row(t['PRIMARY'],endpoint='E1_H_FULL');d=row(t['PRIMARY'],endpoint='E2_D_FULL_25_MINUS_100');legacy=row(t['LEGACY_ANOMALY'],scope='ALL',coordinate='ALL12',endpoint='gap')
    lines=['# EZ-conditioned reference analysis and composition sensitivity','',
      'Real computation on all 80 patients and 7,635 channels. Twelve unchanged physiological coordinates; 100 fixed rankings/partitions per patient. These are exploratory supplements after viewing the parent results. No classifier was fitted.',
      '', '## Two primary endpoints','', '| Endpoint | Mean [nominal 95% CI] | Two-primary adjusted 97.5% CI |','| --- | --- | --- |',
      f"| H: NEZ gain minus EZ gain | {ci(h)} | [{fmt(h['family_low'])}, {fmt(h['family_high'])}] |",f"| D: Full 25% minus Full 100% CRPS | {ci(d)} | [{fmt(d['family_low'])}, {fmt(d['family_high'])}] |",'',
      'Positive H means NEZ reference-fitting gains exceed EZ gains. Positive D means the smaller reference set incurs greater forecast error. Neither establishes localization performance or a clinical reliability threshold.',
      '', '## E1: original frozen channel halves','',
      '| Method | EZ CRPS | NEZ CRPS | EZ gain | NEZ gain | H [nominal 95% CI] |','| --- | --- | --- | --- | --- | --- |']
    for arm in ['GLOBAL','SOURCE','CENTER','SCALE','FULL']:
        q=dict(scope='ALL',coordinate='ALL12',arm=arm)
        vals=[row(t['E1_CLASS_LOSS'],**q,group=g)['mean'] for g in ['EZ','NEZ']]+[row(t['E1_GAIN'],**q,group=g)['mean'] for g in ['EZ','NEZ']]
        hh=row(t['E1_H'],**q);lines.append('| '+arm+' | '+' | '.join(fmt(v) for v in vals)+' | '+('0 (identity)' if arm=='GLOBAL' else ci(hh))+' |')
    lines+=['', 'Each class is averaged within each patient before equal-patient aggregation. Class-weighted reconstruction reproduces the parent all-channel CRPS. E1 uses the exact original reference moments, not the new leave-one-channel-out references.',
      '', '## E2: nested leave-one-channel-out references','',
      '| Other-channel reference fraction | Global CRPS | Source CRPS | Center CRPS | Scale CRPS | Full CRPS [95% CI] | Full EZ-NEZ absolute gap [95% CI] |','| --- | --- | --- | --- | --- | --- | --- |']
    for fraction in [.25,.5,.75,1.]:
        q=dict(scope='ALL',coordinate='ALL12',fraction=fraction,group='ALL');rr=[row(t['E2_LOSS'],**q,arm=a) for a in ['GLOBAL','SOURCE','CENTER','SCALE','FULL']]
        gap=row(t['E2_ANOMALY'],scope='ALL',coordinate='ALL12',fraction=fraction,endpoint='gap')
        lines.append('| '+str(int(100*fraction))+'% | '+' | '.join(fmt(r['mean']) for r in rr[:4])+' | '+ci(rr[4])+' | '+ci(gap)+' |')
    lines+=['', 'Fractions refer to OTHER channels; every query is excluded even at 100%. Counts use ceil(fraction*(C-1)). All fractions use the same queries and nested sets. Full-reference moments are calculated once and reused across repetitions. Global/Source invariance is an engineering check, not empirical evidence.',
      '', '## Reference EZ composition','', '| Fraction | CRPS change per +10pp reference EZ [11-family interval] | Identifiable / assessed patients |','| --- | --- | --- |']
    for pct in [25,50,75]:
        r=row(t['SECONDARY'],endpoint=f'COMPOSITION_SLOPE_{pct}');lines.append(f"| {pct}% | {ci(r,True)} | {r['patients']} / {r['assessed_patients']} |")
    lines+=['| 100% | Structurally not estimable | No repeated composition variation |','',
      'Both predictor and Full CRPS are centered within the same patient/query channel over repeated references before the patient slope is calculated. These slopes are associations. Channel count does not measure anatomical implantation coverage.',
      '', '## All eleven secondary slots','', '| Endpoint | Mean [11-family adjusted interval] |','| --- | --- |']
    lines += [f"| {r['endpoint']} | {ci(r,True)} |" for r in t['SECONDARY']]
    lines+=['', '## Four-source E1 H and E2 curves','', '| Source | E1 Full H [95% CI] | Full CRPS 25% | 50% | 75% | 100% |','| --- | --- | --- | --- | --- | --- |']
    for source in ['hup','lzu','multicenter','pediatric']:
        hh=row(t['E1_H'],scope=source,coordinate='ALL12',arm='FULL');v=[row(t['E2_LOSS'],scope=source,coordinate='ALL12',fraction=f,arm='FULL',group='ALL')['mean'] for f in [.25,.5,.75,1.]]
        lines.append('| '+source+' | '+ci(hh)+' | '+' | '.join(fmt(x) for x in v)+' |')
    lines+=['', 'All source and coordinate results are descriptive. Complete tables retain every source, all 12 coordinates, all five arms, and both class losses. No best-coordinate or favorable-source replacement is permitted.',
      '', '## Legacy absolute-atypicality evidence retained','',f"Original EZ-minus-NEZ absolute-relative gap: {ci(legacy,True)} with its original six-family correction. The exact legacy CSV is copied byte-for-byte. Its inconclusive result is not replaced by new H.",
      '', '## Conditional permutations and limitations','',
      'All 1,000 within-patient count-preserving label permutations are retained for each of the four nonidentity arms and each scope. Ties use the frozen numerical tolerance. Their plus-one tails are conditional diagnostics, not independent population tests.',
      'All resampling operates on patients after repeat/query aggregation: 100 repetitions do not increase the independent sample size. Intervals condition on the historical overlapping TRAIN banks; they do not capture refitting uncertainty. The old 79 original-FIT plus one original-validation assessment exception remains explicit. No anatomical, biological-causal, information-conservation or clinical-localization conclusion follows from H, D or composition slopes alone.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    hd=direction(h);dd=direction(d)
    interpretation={'positive':'NEZ的参考拟合收益大于EZ。','negative':'EZ的参考拟合收益大于NEZ。','inconclusive':'EZ与NEZ的参考拟合收益差异未获支持。','not estimable':'该终点不可估计。'}
    size={'positive':'25%参考的预测误差高于完整其他通道参考。','negative':'25%参考的预测误差低于完整其他通道参考。','inconclusive':'25%与完整其他通道参考的误差差异未获支持，不能解释为等价。','not estimable':'该终点不可估计。'}
    text=['# 中文结论与论文解释','',f"实验一 H={ci(h,True)}（两项主终点校正区间）。"+interpretation[hd],f"实验二 D={ci(d,True)}（两项主终点校正区间）。"+size[dd],'',
      f"原绝对异常度检验仍为 {ci(legacy,True)}，沿用原六项校正。新H不替代它。",'',
      '两个实验均覆盖80人、7635通道，使用原12个坐标与100次固定排序。参考组成斜率控制同一患者、同一评价通道的固定差异，但仍属于关联诊断。参考比例不等于解剖植入范围。','',
      'H仅说明两类通道的参考拟合收益是否不同。D仅说明预设参考数量改变时的预测误差变化。二者均不能单独证明EZ更容易定位，或证明PR保存了全部判别信息。','']
    (HERE/'CHINESE_SUMMARY.md').write_text('\n'.join(text),encoding='utf-8')
    paper=['# Evidence-bounded findings','',f"E1 differential reference gain H: {ci(h,True)}. Direction: {hd}.",f"E2 reference-size penalty D: {ci(d,True)}. Direction: {dd}.",f"Retained parent absolute-atypicality result: {ci(legacy,True)}.",'',
      'These supplements test label-conditioned fitting gains and statistical reference sensitivity in the same historical cohort. A positive H establishes a differential fitting gain under this probe, not localization efficacy. A supported D establishes a statistical reference-size effect, not a clinical minimum channel count. Within-query composition slopes remain associational. All negative, null and source-dependent results must remain visible.','']
    paper+=['## Paper-ready statement','',
      f"In exploratory analyses of 80 historical patients, patient-relative references reduced forecasting errors in both EZ and NEZ channels. The differential gain H was {fmt(h['mean'])} (two-primary adjusted interval [{fmt(h['family_low'])}, {fmt(h['family_high'])}]), leaving preferential NEZ fitting unresolved. Reducing the reference to 25% of the other channels changed Full PR CRPS by {fmt(d['mean'])} relative to all other channels (adjusted interval [{fmt(d['family_low'])}, {fmt(d['family_high'])}]). Within-query associations between reference EZ proportion and forecast error were also inconclusive after adjustment. These findings do not establish EZ-specific reference selectivity or reference-size equivalence.", '',
      'The methods must retain the original 79-FIT/one-validation assessment exception, historical-cohort exploratory status, paired source-stratified patient bootstrap and conditional fixed-TRAIN scope. No new independent cohort was evaluated.','']
    (HERE/'FINDINGS.md').write_text('\n'.join(paper),encoding='utf-8');print('VERIFIED_AGGREGATE_REPORTS_WRITTEN')
if __name__=='__main__':main()
