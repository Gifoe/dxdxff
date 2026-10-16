"""Independent scalar selection, sealed-score and sklearn metric verification."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
import common as c
from decoders import predict,candidates,rows,oracle_macro


def independent_fit(family,x,y_nez,patient,original):
    configs=candidates(family,x);grid=c.np.stack([predict(q,x,patient) for q in configs],1);yy=1-y_nez
    a=[];b=[];ch=[];baseline=predict(original,x,patient)
    for pid in c.np.unique(patient):
        j=patient==pid;y=yy[j,None];p=grid[j]
        tp=(p&(y==1)).sum(0);tn=(~p&(y==0)).sum(0);err=(p!=(y==1)).sum(0)
        ez=2*tp/(2*tp+err);nez=2*tn/(2*tn+err)
        a.append((ez+nez)/2);b.append(ez);ch.append((p!=baseline[j,None]).mean(0))
    a,b,ch=c.np.mean(a,0),c.np.mean(b,0),c.np.mean(ch,0)
    def key(i):
        v=configs[i]['value'];dist=abs(v-(.5 if family=='TOP_FRACTION' else 0)) if v is not None else float('inf')
        return float(a[i]),float(b[i]),-float(ch[i]),-dist,-i
    return configs[max(range(len(configs)),key=key)]


def independent_metrics(y,x,p,ch):
    order=c.np.lexsort((ch,x));first=int(c.np.flatnonzero(y[order])[0])
    return dict(macro_f1=f1_score(y,p,average='macro',zero_division=0),f1=f1_score(y,p,zero_division=0),
        nez_f1=f1_score(y,p,pos_label=0,zero_division=0),accuracy=accuracy_score(y,p),auroc=roc_auc_score(y,-x),
        auprc=average_precision_score(y,-x),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(p.mean()-y.mean()))


def main():
    binding=c.setup();assert c.jr(c.PUB/'RUN_STATUS.json')['status']=='ALL120_DECODER_CELLS_SEALED'
    seal=c.jr(c.ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json');assert seal['input']==binding and seal['all_scores_before_HELD_metrics']
    configs=c.jr(c.ROOT/'CAL_CONFIGS_PRIVATE.json');assert seal['configuration_seal']==c.sha(c.ROOT/'CAL_CONFIGS_PRIVATE.json')
    frame=[];precision=[];oracles=[];checks=[];maximum=0.
    for ctx in c.contexts():
        d=ctx['data'];cal,held=ctx['roles']['cal'],ctx['roles']['held'];f,s=ctx['fold'],ctx['screen']
        for body in c.BODIES:
            source=c.body_scores(ctx,body);original=dict(family='ORIGINAL',value=float(source['theta']))
            for decoder in c.DECODERS:
                procedure=body+'__'+decoder;record=next(r for r in configs['configs'] if (r['fold'],r['screen'],r['procedure'])==(f,s,procedure))
                expected=original if decoder=='ORIGINAL' else independent_fit(decoder,source['cal'],d['y'][cal],d['patient'][cal],original)
                assert expected==record['config'] and source['source']==record['source']
                path=c.ROOT/'held_scores_private'/f'{f}_{s}_{procedure}.pkl'
                sh=next(r['sha256'] for r in seal['cells'] if (r['fold'],r['screen'],r['procedure'])==(f,s,procedure))
                assert c.sha(path)==sh;q=c.read(path);logit=q['logit_nez'];pred=q['pred_ez']
                assert c.np.array_equal(logit,source['held']) and c.np.array_equal(pred,predict(expected,logit,d['patient'][held]))
                part=rows(d,held,logit,pred)
                for r in part:
                    j=d['patient'][held]==r['patient'];y=1-d['y'][held][j]
                    metric=independent_metrics(y,logit[j],pred[j],d['channel'][held][j])
                    for k,v in metric.items():maximum=max(maximum,abs(r[k]-v));assert abs(r[k]-v)<1e-12
                frame.extend([dict(fold=f,screen=s,procedure=procedure,**r) for r in part])
            # Binary ORIGINAL replay is exact; legacy probability ranking is a separate precision audit.
            old=c.wave.parent.patient_rows(d,held,source['held'],source['theta'])
            current=rows(d,held,source['held'],predict(original,source['held'],d['patient'][held]))
            for a,b in zip(old,current):
                assert a['patient']==b['patient']
                for k in ['macro_f1','f1','nez_f1','accuracy','ez_fraction_bias']:assert abs(a[k]-b[k])<1e-12
                j=d['patient'][held]==a['patient'];x=source['held'][j];pn=c.probability(x)
                precision.append(dict(fold=f,screen=s,body=body,patient=a['patient'],center=a['center'],channels=len(x),
                     saturated_nez=int(((pn==0)|(pn==1)).sum()),probability_distinct=len(c.np.unique(pn)),margin_distinct=len(c.np.unique(x)),
                     **{k+'_legacy':a[k] for k in ['auroc','auprc','mrr','hit_at_1']},
                     **{k+'_margin':b[k] for k in ['auroc','auprc','mrr','hit_at_1']}))
                oracles.append(dict(fold=f,screen=s,body=body,patient=a['patient'],center=a['center'],
                                    oracle_macro_f1=oracle_macro(1-d['y'][held][j],x),original_macro_f1=b['macro_f1']))
            checks.append(dict(fold=f,screen=s,body=body,CAL_refit_identical=True,HELD_scores_exact=True,original_binary_replay_exact=True))
        print('DECISION_PATH_INDEPENDENTLY_VERIFIED',f,s,flush=True)
    frame=c.pd.DataFrame(frame);assert len(frame)==2040
    old=c.pd.DataFrame(c.read(c.ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl'));keys=['fold','screen','procedure','patient']
    a,b=frame.sort_values(keys).reset_index(drop=True),old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and c.np.max(c.np.abs(a[c.METRICS].to_numpy()-b[c.METRICS].to_numpy()))<1e-12
    for body,expected in [('NATIVE',.6090840707517672),('PRECOMPAND_RESIDUAL',.6091492406139017)]:
        assert abs(frame[frame.procedure==body+'__ORIGINAL'].macro_f1.mean()-expected)<1e-12
        ranks=frame[frame.procedure.str.startswith(body+'__')].groupby(['fold','screen','patient'])[['auroc','auprc','mrr','hit_at_1']].std(ddof=0)
        assert c.np.max(ranks.to_numpy())<1e-12
    frame.groupby('procedure')[c.METRICS].mean().reset_index().to_csv(c.PUB/'INTERNAL_MAIN_RESULTS.csv',index=False)
    frame.groupby(['procedure','fold'])[c.METRICS].mean().reset_index().to_csv(c.PUB/'INTERNAL_FOLD_RESULTS.csv',index=False)
    sources=[];absolute=[];paired=[];choices=[]
    for (procedure,center),g in frame.groupby(['procedure','center']):
        n=g.patient.nunique();sources.append(dict(procedure=procedure,center=center,appearances=len(g),unique_patients=n,suppressed=n<5,
                          **{m:float(g[m].mean()) if n>=5 else c.np.nan for m in c.METRICS}))
    for procedure,g in frame.groupby('procedure'):absolute.extend([dict(procedure=procedure,**r) for r in c.interval(g.to_dict('records'))])
    pairs=[(p,p.split('__')[0]+'__ORIGINAL') for p in c.PROCEDURES if not p.endswith('__ORIGINAL')]
    pairs.append(('PRECOMPAND_RESIDUAL__ORIGINAL','NATIVE__ORIGINAL'))
    for left,right in pairs:
        m=frame[frame.procedure==left].merge(frame[frame.procedure==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
        diff=c.pd.DataFrame(dict(patient=m.patient,**{k:m[k+'_l']-m[k+'_r'] for k in c.METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in c.interval(diff.to_dict('records'))])
    for fold in range(1,6):
        scores=[]
        for p in c.PROCEDURES:
            g=frame[(frame.fold==fold)&(frame.procedure==p)];ss=g.groupby('center').macro_f1.mean()
            score=float(g.macro_f1.mean())-.1*(float(ss.std(ddof=1)) if len(ss)>1 else 0.)
            scores.append(dict(procedure=p,score=score,parameters=(576 if p.startswith('PRECOMPAND') else 0)+(0 if p.endswith('__ORIGINAL') else 1)))
        high=max(r['score'] for r in scores)
        selected=min([r for r in scores if r['score']>=high-.005],key=lambda r:(r['parameters'],-r['score'],c.PROCEDURES.index(r['procedure'])))
        choices.append(dict(fold=fold,selected_procedure=selected['procedure'],score=selected['score'],all_scores=scores))
    c.pd.DataFrame(sources).to_csv(c.PUB/'INTERNAL_SOURCE_RESULTS.csv',index=False)
    c.pd.DataFrame(absolute).to_csv(c.PUB/'INTERNAL_ABSOLUTE_CI.csv',index=False)
    c.pd.DataFrame(paired).to_csv(c.PUB/'INTERNAL_PAIRED_CI.csv',index=False)
    pr=c.pd.DataFrame(precision);cols=[n for n in pr if n not in ['patient','center','fold','screen','body']]
    pr.groupby('body')[cols].mean().reset_index().to_csv(c.PUB/'RANKING_PRECISION_AUDIT.csv',index=False)
    oracle=c.pd.DataFrame(oracles);oracle.groupby('body')[['oracle_macro_f1','original_macro_f1']].mean().reset_index().to_csv(c.PUB/'INTERNAL_ORACLE_DIAGNOSTIC.csv',index=False)
    c.pd.DataFrame([{k:v for k,v in r.items() if k!='all_scores'} for r in choices]).to_csv(c.PUB/'OWN_FIT_SELECTION.csv',index=False)
    c.freeze(c.ROOT/'OWN_FIT_SELECTION_PRIVATE.json',dict(input=binding,score_seal=c.sha(c.ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json'),choices=choices))
    c.write(c.PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checks=checks,patient_metric_rows=len(frame),maximum_metric_error=maximum,
        independent_sklearn_metrics=True,ranking_invariant_within_body=True,original_Macro_reproduced_exactly=True,
        ownFIT_selection_sha256=c.sha(c.ROOT/'OWN_FIT_SELECTION_PRIVATE.json'),selection_uses_ownFIT_only=True,new_Outer_passes=0))
    c.write(c.PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_PATIENT_DECODERS_INTERNAL',neural_fits=0,scalar_fits=90,logical_cells=120,
        held_appearances_per_procedure=255,unique_HELD_patients=79,original_Development_Outer_used=False,goal_established=False,
        oracle_is_non_deployable_not_goal_evidence=True,independent_confirmation=False))
    print('DECISION_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
