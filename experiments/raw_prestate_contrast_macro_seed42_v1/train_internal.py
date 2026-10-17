"""Bounded pre-state information experiment; no final Development/Outer API."""
from common import *
from admit import evaluate

# The already-tested fit implementation runs under this experiment's common,
# model and admit modules. Its source hash is fixed in setup() and the freeze.
engine=module('prestate_pinned_fresh_fit_engine',FIT_ENGINE)
fit=engine.fit
_write=engine.write
def _progress(path,value):
    if path.name=='RUN_STATUS.json' and value.get('status')=='TRAINING_JOINT_RAW_INTERNAL':
        value=dict(value,status='TRAINING_PRESTATE_RAW_INTERNAL')
    _write(path,value)
engine.write=_progress


def means(rows):return {m:float(np.mean([r[m] for r in rows])) for m in METRICS}
def parameters(arm):return 12529 if arm in ['NATIVE','COMPACT_PRE'] else 13105 if arm=='FROZEN_PRE' else 14577
def penalized(rows):
    values=[np.mean([r['macro_f1'] for r in rows if r['center']==s]) for s in sorted(set(r['center'] for r in rows))]
    return means(rows)['macro_f1']-.1*(float(np.std(values,ddof=1)) if len(values)>1 else 0.)
def choose(results):return min(results,key=lambda r:(-r['penalized_macro'],r['parameters'],ARMS.index(r['arm'])))


def baseline(ctx,arm):
    logit,theta=reference(ctx,arm);rows=held_rows(ctx,logit,theta,arm)
    return dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,parameters=parameters(arm),epochs=0,selected_epoch=0,
        threshold=theta,patients=len(rows),penalized_macro=penalized(rows),**means(rows)),rows


def first_gate(first):
    candidate=choose([r for r in first if r['arm'] in TRAINED]);base=[r for r in first if r['arm'] in REFERENCES]
    assert {r['arm'] for r in base}==set(REFERENCES)
    mg=candidate['macro_f1']-max(r['macro_f1'] for r in base);ag=candidate['auroc']-max(r['auroc'] for r in base)
    return dict(chosen=candidate['arm'],advanced=mg>=.01 and ag>=-.01,Macro_gain=mg,AUROC_gain=ag)


def paired(frame,left,right):
    q=frame[frame.arm==left].merge(frame[frame.arm==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
    assert len(q)==len(frame[frame.arm==left])==len(frame[frame.arm==right])
    return pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in METRICS}))


def selection(frame,race):
    fold=race['fold'];g=frame[frame.fold==fold];scores=[];gates=[];eligible=[]
    for arm in REFERENCES[:2]+(TRAINED if race['advanced'] else []):
        z=g[g.arm==arm]
        assert set(z.screen)==({1,2,3} if race['advanced'] else {1})
        scores.append(dict(arm=arm,parameters=parameters(arm),penalized_macro=penalized(z.to_dict('records'))))
    if race['advanced']:
        first=g[g.screen==1];strong=first[first.arm=='COMPACT_PRE'];assert len(strong)>0
        for arm in TRAINED:
            candidate=g[g.arm==arm];one=first[first.arm==arm]
            passed=bool(one.macro_f1.mean()>=strong.macro_f1.mean()+.01 and one.auroc.mean()>=strong.auroc.mean()-.01)
            contrasts=[]
            for ref in REFERENCES[:2]:
                base=g[g.arm==ref];diff=paired(g[g.arm.isin([arm,ref])],arm,ref)
                ci=next(r for r in parent.cluster_interval(diff.to_dict('records')) if r['metric']=='macro_f1')
                wins=sum(candidate[candidate.screen==s].macro_f1.mean()>base[base.screen==s].macro_f1.mean() for s in [1,2,3])
                macro_gain=float(candidate.macro_f1.mean()-base.macro_f1.mean());auc_gain=float(candidate.auroc.mean()-base.auroc.mean())
                ok=macro_gain>=.015 and float(ci['low'])>0 and wins>=2 and auc_gain>=-.01;passed=passed and ok
                contrasts.append(dict(reference=ref,Macro_gain=macro_gain,AUROC_gain=auc_gain,paired_Macro_low=float(ci['low']),
                    paired_Macro_high=float(ci['high']),path_wins=int(wins),passed=bool(ok)))
            gates.append(dict(fold=fold,arm=arm,eligible=bool(passed),contrasts=contrasts))
            if passed:eligible.append(arm)
    allowed=eligible if eligible else REFERENCES[:2]
    selected=choose([r for r in scores if r['arm'] in allowed])['arm']
    return dict(fold=fold,advanced=bool(race['advanced']),first_candidate=race['chosen'],selected_arm=selected,
                fresh_eligible=bool(eligible),eligible_arms=eligible,scores=scores),gates


def main():
    binding=setup();admission=jr(PUB/'ADMISSION.json');assert admission['status']=='PASS' and admission['input_binding_sha256']==digest(binding)
    cc={(r['fold'],r['screen']):r for r in contexts()};store=RawStore();summaries=[];rows=[];histories=[];seals=[];races=[]
    def record(ctx,arm):
        if arm in REFERENCES:summary,part=baseline(ctx,arm);history=[]
        else:
            summary,part,history=fit(ctx,arm,binding,store)
            path=ROOT/'models_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm/'HELD_PRIVATE.pkl'
            seals.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,sha256=sha(path)))
        summaries.append(summary);rows.extend(part)
        histories.extend([dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**r) for r in history])
        pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False)
        pd.DataFrame(histories).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
        return summary
    for fold in range(1,6):
        first=[record(cc[(fold,1)],arm) for arm in ARMS]
        race=dict(fold=fold,**first_gate(first));race['promoted']=TRAINED if race['advanced'] else []
        freeze(ROOT/'races_private'/f'fold{fold}.json',race);races.append(race)
        if race['advanced']:
            for screen in [2,3]:
                ctx=cc[(fold,screen)]
                for arm in reference_names(ctx)+TRAINED:record(ctx,arm)
        write(PUB/'RACING_DECISIONS.json',dict(status='PRESTATE_RAW_INTERNAL_RACING',decisions=races))
    assert 15<=len(seals)<=45;save(ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl',rows)
    freeze(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json',dict(input=binding,cells=seals,races=races))
    write(PUB/'RUN_STATUS.json',dict(status='ALL_REGISTERED_PRESTATE_RAW_INTERNAL_COMPLETE',actual_neural_fits=len(seals),logical_cells=len(summaries),
        epochs=sum(r['epochs'] for r in summaries),score_seal_sha256=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),new_Outer_passes=0,goal_established=False))
    print('PRESTATE_RAW_ALL_INTERNAL_COMPLETE',len(seals),flush=True)


if __name__=='__main__':main()
