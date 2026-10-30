"""Export and independently verify the user-requested DREChanNet seed42 view."""
import argparse,csv,hashlib,io,json,math,re
from pathlib import Path
from sklearn.metrics import roc_auc_score,f1_score
FILES={'README_ZH.md','CASE_METADATA.csv','CASE_METRICS.csv','CASE_CONFUSION.csv',
       'CHANNEL_SCORES_ANONYMOUS.csv','RECALL_AT_K.csv',
       'DRECHANNET_SEED42_CHANNELS_WIDE.csv','DRECHANNET_SEED42_EZ_CHANNELS_ONLY.csv',
       'SOURCE_PUBLIC_MANIFEST.json','PROVENANCE.json','VERIFICATION.json'}
METRICS=['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct']

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def writecsv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def writejson(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def verify(root,check_manifest=True):
    root=Path(root)
    if check_manifest:
        assert {p.name for p in root.iterdir()}==FILES|{'PUBLIC_MANIFEST.json'}
        manifest=json.loads((root/'PUBLIC_MANIFEST.json').read_text(encoding='utf-8'))
        assert set(manifest['files'])==FILES
        assert manifest['real_patient_identifiers_included'] is False and manifest['original_contact_names_included'] is False
        for name,v in manifest['files'].items():
            b=(root/name).read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
    for name in FILES:
        if not name.endswith('.csv'):continue
        rr=read(root/name)
        assert not {'patient','patient_id','private_patient_id','channel_name','original_channel_name','identity'}&set(rr[0])
        for r in rr:
            if 'case_id' in r:assert r['case_id']=='CASE-A01'
            if 'model' in r:assert r['model']=='DRECHANNET'
            if 'seed' in r:assert r['seed']=='42'
            if 'channel_id' in r:assert re.fullmatch(r'CH\d{4}',r['channel_id'])
    meta=read(root/'CASE_METADATA.csv');assert len(meta)==1
    assert meta[0]['center']=='HUP' and int(meta[0]['outer_fold'])==4 and int(meta[0]['total_channels'])==49 and int(meta[0]['ez_channels'])==10
    scores=read(root/'CHANNEL_SCORES_ANONYMOUS.csv');metrics=read(root/'CASE_METRICS.csv');cm=read(root/'CASE_CONFUSION.csv');curves=read(root/'RECALL_AT_K.csv')
    assert (len(scores),len(metrics),len(cm),len(curves))==(98,5,2,100)
    values={};lookup={}
    for cond in ['NO_PR','PR']:
        rr=[r for r in scores if r['condition']==cond];assert len(rr)==49
        assert len({r['channel_id'] for r in rr})==49
        y=[int(r['truth_ez']) for r in rr];p=[float(r['score_ez']) for r in rr];pred=[int(r['prediction_ez']) for r in rr]
        assert sum(y)==10 and all(math.isfinite(v) and 0<=v<=1 for v in p)
        assert all(r['truth_label']==('EZ' if int(r['truth_ez']) else 'NEZ') for r in rr)
        order=sorted(range(49),key=lambda i:(-p[i],rr[i]['channel_id']))
        assert [int(rr[i]['rank_ez']) for i in order]==list(range(1,50))
        values[cond]=dict(auroc=float(roc_auc_score(y,p)),macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro')),top1=float(y[order[0]]))
        for frac in [.1,.2]:values[cond][f'recall_at_{round(frac*100)}pct']=sum(y[i] for i in order[:math.ceil(frac*49)])/10
        tp=sum(a==b==1 for a,b in zip(y,pred));fp=sum(a==0 and b==1 for a,b in zip(y,pred));fn=10-tp;tn=39-fp
        c=next(r for r in cm if r['condition']==cond)
        expected=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0,sensitivity=tp/10,fpr=fp/39,predicted_positive=tp+fp)
        for k,v in expected.items():assert abs(float(c[k])-v)<1e-12
        for k in range(50):
            r=next(r for r in curves if r['condition']==cond and int(r['k'])==k);hits=sum(y[i] for i in order[:k])
            assert int(r['ez_recovered'])==hits and int(r['ez_total'])==10 and int(r['total_channels'])==49
            assert abs(float(r['recall'])-hits/10)<1e-12 and abs(float(r['review_fraction'])-k/49)<1e-12
            assert (r['precision']=='' if k==0 else abs(float(r['precision'])-hits/k)<1e-12)
        for r in rr:lookup[r['channel_id'],cond]=r
    assert {r['metric'] for r in metrics}==set(METRICS)
    for r in metrics:
        no,pr=[values[c][r['metric']] for c in ['NO_PR','PR']]
        assert max(abs(no-float(r['no_pr'])),abs(pr-float(r['pr'])),abs(pr-no-float(r['change'])))<1e-12
    wide=read(root/'DRECHANNET_SEED42_CHANNELS_WIDE.csv');ez=read(root/'DRECHANNET_SEED42_EZ_CHANNELS_ONLY.csv')
    assert len(wide)==49 and len(ez)==10 and {r['channel_id'] for r in wide}=={r['channel_id'] for r in scores}
    for r in wide:
        for cond,prefix in [('NO_PR','no_pr'),('PR','pr')]:
            src=lookup[r['channel_id'],cond]
            assert r['truth_ez']==src['truth_ez'] and r['truth_label']==src['truth_label']
            for k in ['score_ez','rank_ez','prediction_ez']:assert r[prefix+'_'+k]==src[k]
    assert ez==[r for r in wide if r['truth_ez']=='1']
    assert (int(next(r for r in cm if r['condition']=='NO_PR')['tp']),int(next(r for r in cm if r['condition']=='PR')['tp']))==(2,5)
    assert (int(next(r for r in cm if r['condition']=='NO_PR')['fp']),int(next(r for r in cm if r['condition']=='PR')['fp']))==(0,1)
    return dict(status='PASS',case_id='CASE-A01',model='DRECHANNET',seed=42,channels=49,ez_channels=10,metric_rows=5,confusion_rows=2,score_rows=98,recall_k_rows=100,all_ranks_metrics_counts_and_curves_independently_recomputed=True)

def export(source,out):
    source=Path(source);out=Path(out);assert not out.exists()
    manifest=json.loads((source/'PUBLIC_MANIFEST.json').read_text(encoding='utf-8'))
    for name,v in manifest['files'].items():
        b=(source/name).read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
    out.mkdir(parents=True)
    meta=read(source/'CASE_METADATA.csv')[0]
    writecsv(out/'CASE_METADATA.csv',[{k:meta[k] for k in ['case_id','center','outer_fold','total_channels','ez_channels','nez_channels']}])
    selected={}
    for name in ['CASE_METRICS.csv','CASE_CONFUSION.csv','CHANNEL_SCORES_ANONYMOUS.csv','RECALL_AT_K.csv']:
        rows=[r for r in read(source/name) if r['model']=='DRECHANNET' and r['seed']=='42']
        assert rows;selected[name]=rows;writecsv(out/name,rows)
    lookup={(r['channel_id'],r['condition']):r for r in selected['CHANNEL_SCORES_ANONYMOUS.csv']}
    wide=[]
    for ch in sorted({k[0] for k in lookup}):
        no=lookup[ch,'NO_PR'];r={k:no[k] for k in ['case_id','channel_id','truth_label','truth_ez']}
        for cond,prefix in [('NO_PR','no_pr'),('PR','pr')]:
            r.update({prefix+'_'+k:lookup[ch,cond][k] for k in ['score_ez','rank_ez','prediction_ez']})
        wide.append(r)
    writecsv(out/'DRECHANNET_SEED42_CHANNELS_WIDE.csv',wide)
    writecsv(out/'DRECHANNET_SEED42_EZ_CHANNELS_ONLY.csv',[r for r in wide if r['truth_ez']=='1'])
    (out/'SOURCE_PUBLIC_MANIFEST.json').write_bytes((source/'PUBLIC_MANIFEST.json').read_bytes())
    writejson(out/'PROVENANCE.json',dict(case_id='CASE-A01',model='DRECHANNET',seed=42,comparison='Original complete FULL vs WO_BOTH_PR; 11425 parameters; not LOCO/no-guidance',original_source_commit='493a7d35ba70d419c788a5e990a7460b0e112f38',source_archive_sha256='5030436c5008cfa5cb8158f1f7c6040ad071a832e4d6741127e50da63bc9b0fc',source_manifest_sha256=hashlib.sha256((source/'PUBLIC_MANIFEST.json').read_bytes()).hexdigest(),case_selection='Retrospectively selected using all three seeds mean TP/FP and AUROC; all four prior cases excluded',display_selection='User requested seed42 after viewing all three seeds; this is a selected case/seed illustration, not an independent validation',original_thresholds_unchanged=True,new_training=0,new_inference=0,other_seed_predictions_in_this_subset=False,original_contact_names_included=False,real_patient_identifiers_included=False))
    rows=['# CASE-A01：DREChanNet seed42','',
          '**HUP，Outer Fold 4，49 个通道：10 EZ、39 NEZ。**','',
          '以下均为冻结原验证阈值下的 WO_BOTH_PR → FULL。原完整 DREChanNet 含 early FiLM / late Gate（11,425 参数）。没有重新训练、推理或调整阈值。','',
          '|指标|No PR|Full PR|','|---|---:|---:|',
          '|TP：检出的 EZ|2|5|','|FP：误报的 NEZ|0|1|','|FN：漏检的 EZ|8|5|','|TN：正确排除的 NEZ|39|38|']
    label={'auroc':'AUROC','macro_f1':'Macro-F1','top1':'Top-1','recall_at_10pct':'Recall@10%','recall_at_20pct':'Recall@20%'}
    for r in selected['CASE_METRICS.csv']:
        fac=1 if r['metric'] in ['auroc','macro_f1'] else 100;suffix='' if fac==1 else '%';fmt='.4f' if fac==1 else '.2f'
        rows.append(f"|{label[r['metric']]}|{float(r['no_pr'])*fac:{fmt}}{suffix}|{float(r['pr'])*fac:{fmt}}{suffix}|")
    rows+=['','精确率：100% → 83.33%；敏感度：20% → 50%。前 10% 审阅预算实际为 5 个通道，前 20% 为 10 个通道。','',
           '## 筛选背景','',
           '病例按三个 seed（42/43/44）的平均 TP 增益、有限 FP 增量及平均 AUROC 提高筛选；用户在查看三 seed 结果后指定导出 seed42。本包仅展示 seed42。原三 seed 数据完整保留于可信服务器及本机。原病例平均 AUROC 0.7521 → 0.7692；两个 seed 提高、seed44 下降。该病例及 seed 的展示选择是回顾性的，不能作为独立确认或替代总体结果。','',
           '## 文件','',
           '- [全部 49 通道的标签、实际分数、预测与排名](DRECHANNET_SEED42_CHANNELS_WIDE.csv)。',
           '- [10 个 EZ 通道](DRECHANNET_SEED42_EZ_CHANNELS_ONLY.csv)。',
           '- [TP/FP/FN/TN、精确率等](CASE_CONFUSION.csv)。',
           '- [全部指标与原筛选分位](CASE_METRICS.csv)。',
           '- [逐个整数 K 的 Recall 曲线](RECALL_AT_K.csv)。','',
           '真实患者身份与原始通道名称未公开。通道别名 CH0001 等保留原同分排序顺序，不代表解剖位置。独立核验覆盖全部指标、混淆矩阵、分数排序、每个 K 和宽表一致性。','']
    (out/'README_ZH.md').write_text('\n'.join(rows),encoding='utf-8')
    receipt=verify(out,check_manifest=False);writejson(out/'VERIFICATION.json',receipt)
    assert {p.name for p in out.iterdir()}==FILES
    writejson(out/'PUBLIC_MANIFEST.json',dict(files={name:dict(bytes=(out/name).stat().st_size,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in sorted(FILES)},real_patient_identifiers_included=False,original_contact_names_included=False))
    print('SEED42_EXPORT_PASS',json.dumps(verify(out)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:print('SEED42_INDEPENDENT_VERIFICATION_PASS',json.dumps(verify(a.verify)))
    else:assert a.source and a.output;export(a.source,a.output)
