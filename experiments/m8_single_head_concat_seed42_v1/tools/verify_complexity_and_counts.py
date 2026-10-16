"""Independent analytical affine MAC and anonymous confusion-count audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from report import *

def main():
    audit=jr(PUB/'COMPLEXITY_AUDIT.json');S,C,W,L=audit['actual_input_shape'];assert L==500
    table=pd.read_csv(PUB/'COMPLEXITY.csv');padded=S*C*W;checks=[]
    for _,r in table.iterrows():
        n=int(r.valid_raw_windows);name=r.model
        if name=='REF_D1':expected=1440*C
        elif name=='REF_RAW_ONLY':expected=73764*n+6208*C
        elif name=='REF_LATE_FUSION':expected=73764*n+7648*C
        elif name in ['REF_M8','REF_M8_NO_VAR']:expected=73764*n+992*padded+7648*C
        else:expected=73764*n+7104*C+(992*padded if name in ['S3','S4'] else 0)
        assert r.actual_affine_MAC==expected,(name,r.actual_affine_MAC,expected)
        checks.append(dict(model=name,expected_actual_affine_MAC=expected,matches_hooks=True))
    cost=pd.read_csv(PUB/'OUTER_FORWARD_COST.csv')
    for model,g in cost.groupby('model'):
        assert g.valid_windows.sum()==1471965 and g.channels.sum()==7635
        required=73764*1471965+7104*7635
        extra=int(g.actual_affine_MAC.sum())-required
        assert extra==0 if model in ['S0','S1','S2'] else extra>0 and extra%992==0
    p=read(ROOT/'OUTER_PATIENT_METRICS_PRIVATE.pkl');rows=[]
    for model,g in p.groupby('model'):
        assert int(g.n.sum())==7635 and int(g.positives.sum())==1743
        for scope in ['all','fold','center']:
            groups=[('all',g)] if scope=='all' else g.groupby(scope)
            for key,h in groups:
                rows.append(dict(model=model,scope=scope,group=str(key),patients=len(h),unit='pooled_counts_not_patientmacro_rates',
                    **{k:int(h[k].sum()) for k in ['TP','TN','FP','FN','n','positives','negatives']}))
    pd.DataFrame(rows).to_csv(PUB/'CONFUSION_COUNTS.csv',index=False)
    write(PUB/'INDEPENDENT_COMPLEXITY_AUDIT.json',dict(status='PASS',actual_matched_FIT_MAC_formulas=checks,
        full80_new_CNN_head_terms_and_Gate_multiple_checked=True,pooled_confusion_rows=len(rows),no_extra_inference_or_training=True,source_sha256=sha(__file__)))
    print('SINGLE_HEAD_MAC_FORMULAS_AND_ANONYMOUS_CONFUSION_COUNTS_PASS',flush=True)

if __name__=='__main__':main()
