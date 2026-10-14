"""Three fixed contrasts; 10,000 paired unique-patient draws, seed42."""
from support import *
CONTRASTS=[('DELTA_effect','A1','A0'),('PR_effect','A2','A1'),('RawCorrection_package_effect','A3','A2')]

def run(a):
    lock,c,core,fac,trainer,records,rec,banks,binding=setup(a);pub=a.runtime/'results'
    assert jread(pub/'OUTER_STATUS.json')['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
    df=pd.DataFrame(read(a.runtime/'OUTER_METRICS_PRIVATE.pkl'));means=df.groupby('arm')[METRICS].mean();increments=[];boot=[]
    for name,l,r in CONTRASTS:
        left=df[df.arm==l];right=df[df.arm==r]
        delta=means.loc[l]-means.loc[r]
        paired=left.set_index(['fold','patient'])[METRICS]-right.set_index(['fold','patient'])[METRICS]
        wins=int((paired.macro_f1>0).sum());folds=int((paired.macro_f1.groupby(level='fold').mean()>0).sum())
        increments.append(dict(increment=name,later=l,earlier=r,**{f'delta_{k}':v for k,v in delta.items()},MacroF1_improved_patients=wins,MacroF1_improved_folds=folds,patients=80,folds=5,package_not_pure_CNN=(l=='A3')))
        boot.extend(dict(increment=name,later=l,earlier=r,**v) for v in BOOT.compare(c,left,right,METRICS))
    pd.DataFrame(increments).to_csv(pub/'PROGRESSIVE_INCREMENT.csv',index=False)
    pd.DataFrame(boot).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
    trend={m:bool(np.all(np.diff(means.loc[['A0','A1','A2','A3'],m])>0)) for m in METRICS[:4]}
    write(pub/'SCIENTIFIC_GATE.json',dict(terminal='PROGRESSIVE_IMPROVEMENT_OBSERVED' if all(trend.values()) else 'PROGRESSIVE_TREND_NOT_UNIFORM',strict_monotonic_order=trend,test_reuse='EXPLORATORY_REPEATED_TEST',unadjusted_uncertainty=True,no_test_driven_iteration=True,RawCorrection_effect='complete M8 package, includes physiology gate and changed training objective',PR_effect='input PatientZ in physiological D1 branch; not separately M8 Raw PatientZ'))
    print('BOOTSTRAP_10000_COMPLETE',flush=True)

if __name__=='__main__':run(args())
