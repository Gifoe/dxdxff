"""Frozen two-body margin recovery; 40 fixed cells plus five selected aliases."""
import argparse
import gc
import sys
import common as c
from decoders import fit,predict,patient_metrics
from summarize import independent_fit

FINAL=c.ROOT/'final_stage_v1';PUBLIC=FINAL/'public'
PROCEDURES=c.PROCEDURES+['SELECTED']
OLD=c.WAVE/'final_stage_v1'


def setup():
    initial=c.setup();verification=c.jr(c.PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json')
    assert verification['status']=='PASS' and c.jr(c.PUB/'INTERNAL_TERMINAL.json')['status']=='COMPLETE_VERIFIED_PATIENT_DECODERS_INTERNAL'
    selection=c.ROOT/'OWN_FIT_SELECTION_PRIVATE.json';assert c.sha(selection)==verification['ownFIT_selection_sha256']
    selected=c.jr(selection);assert selected['input']==initial and selected['score_seal']==c.sha(c.ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json')
    choices={r['fold']:r['selected_procedure'] for r in selected['choices']};assert len(choices)==5
    previous=c.read(OLD/'MODEL_FREEZE_PRIVATE.pkl')
    models=[r for r in previous['rows'] if r['procedure'] in c.BODIES];assert len(models)==10
    for r in models:
        assert c.sha(r['body'])==r['body_sha256']
        if r['matrix']:assert c.sha(r['matrix'])==r['matrix_sha256']
    history=c.wave.parent.HISTORY;hf=c.read(history/'MODEL_FREEZE_PRIVATE.pkl')
    assert c.sha(history/'MODEL_FREEZE_PRIVATE.pkl')==previous['input']['body_freeze']
    refs=sorted([r for r in hf['selections'] if r['family']=='Concat' and r['recipe']=='RAW_ONLY'],key=lambda r:r['fold'])
    assert [r['fold'] for r in refs]==list(range(1,6))
    for r in refs:
        native=next(q for q in models if q['fold']==r['fold'] and q['procedure']=='NATIVE')
        assert native['body_sha256']==r['checkpoint_sha256'] and native['threshold']==r['threshold']
        assert c.sha(r['validation'])==r['validation_sha256']
    binding=dict(internal_binding=c.digest(initial),internal_selection=c.sha(selection),previous_model_freeze=c.sha(OLD/'MODEL_FREEZE_PRIVATE.pkl'),
        previous_score_seal=c.sha(OLD/'OUTER_SCORE_SEAL_PRIVATE.json'),body_inputs=[{k:r[k] for k in ['fold','procedure','body_sha256','matrix_sha256','threshold']} for r in models],
        source={n:c.sha(c.HERE/n) for n in ['final_stage.py','verify_final.py','common.py','decoders.py','summarize.py']},
        waveform_model_source=c.sha(c.WAVE_CODE/'model.py'),raw_registry=c.io.EXPECTED_REG,reconstruction=c.io.EXPECTED_REC)
    FINAL.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True);c.freeze(FINAL/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,choices,models,refs


def calibrate():
    binding,choices,models,refs=setup();_,banks=c.io.identity_and_banks(labels=True);result=[];public=[]
    for ref in refs:
        fold=ref['fold'];d=banks[fold-1];va=d['val'];byprocedure={}
        for body in c.BODIES:
            model=next(r for r in models if r['fold']==fold and r['procedure']==body)
            if body=='NATIVE':
                saved=c.read(ref['validation']);assert c.np.array_equal(saved['archive_indices'],d['archive_indices'][va]);logit=saved['logit']
                source=dict(validation_sha256=ref['validation_sha256'])
            else:
                p=OLD/'matrices_private'/f'fold{fold}_screenfinal'/body/'HELD_PRIVATE.pkl';saved=c.read(p)
                state=c.torch.load(model['matrix'],map_location='cpu',weights_only=False)
                assert saved['binding']==state['binding'] and saved['binding']['input']==c.read(OLD/'MODEL_FREEZE_PRIVATE.pkl')['input']
                assert c.np.array_equal(saved['cal_archive_indices'],d['archive_indices'][va])
                assert saved['summary']['threshold']==model['threshold'];logit=saved['cal_logit_nez'];source=dict(validation_sha256=c.sha(p))
            assert len(logit)==len(va) and c.np.isfinite(logit).all()
            original=dict(family='ORIGINAL',value=float(model['threshold']))
            for decoder in c.DECODERS:
                config,summary=(original,{}) if decoder=='ORIGINAL' else fit(decoder,logit,d['y'][va],d['patient'][va],original)
                expected=original if decoder=='ORIGINAL' else independent_fit(decoder,logit,d['y'][va],d['patient'][va],original)
                assert config==expected
                procedure=body+'__'+decoder
                row=dict(fold=fold,procedure=procedure,body=body,decoder=decoder,config=config,model=model,
                         development_indices=d['archive_indices'][va].tolist(),development_source=source)
                result.append(row);byprocedure[procedure]=row
                public.append(dict(fold=fold,procedure=procedure,body=body,decoder=decoder,parameter=config['value'],mode=config.get('mode','scalar'),
                                   neural_updates=0,**summary))
        result.append(dict(byprocedure[choices[fold]],procedure='SELECTED',selected_procedure=choices[fold]))
    assert len(result)==45
    c.freeze(FINAL/'DECISION_FREEZE_PRIVATE.json',dict(input=binding,rows=result,scalar_fits=30,neural_updates=0))
    c.pd.DataFrame(public).to_csv(PUBLIC/'FINAL_CALIBRATION.csv',index=False)
    c.write(PUBLIC/'CALIBRATION_STATUS.json',dict(status='ALL45_DECISIONS_FROZEN',logical_cells=45,new_scalar_fits=30,neural_updates=0,
        all_choices_before_original_Development=True,decision_freeze_sha256=c.sha(FINAL/'DECISION_FREEZE_PRIVATE.json'),new_Outer_passes=0))
    print('RAW_DECISION_ALL45_FROZEN',flush=True)


def load_model_class():
    # Old model's module globals must resolve to the pinned OLD common module.
    prior=sys.modules['common'];sys.modules['common']=c.wave
    try:old=c.module('raw_decision_frozen_wave_model',c.WAVE_CODE/'model.py')
    finally:sys.modules['common']=prior
    return old.Model


def test():
    binding,choices,models,refs=setup();status=c.jr(PUBLIC/'CALIBRATION_STATUS.json');assert status['status']=='ALL45_DECISIONS_FROZEN'
    assert status['decision_freeze_sha256']==c.sha(FINAL/'DECISION_FREEZE_PRIVATE.json')
    fr=c.jr(FINAL/'DECISION_FREEZE_PRIVATE.json');assert fr['input']==binding
    c.freeze(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json',dict(input=binding,decisions=c.sha(FINAL/'DECISION_FREEZE_PRIVATE.json'),
        all_choices_before_Development=True,all_decisions_before_new_margin_inference=True))
    fh=c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');oldseal=c.jr(OLD/'OUTER_SCORE_SEAL_PRIVATE.json')
    meta,banks=c.io.identity_and_banks(labels=False);store=c.wave.RawStore();Model=load_model_class();cells={};parity=[]
    for ref in refs:
        fold=ref['fold'];d=banks[fold-1];ix=c.np.flatnonzero(c.np.isin(meta['patient'],sorted(d['roles']['test'])))
        assert set(meta['patient'][ix]).isdisjoint(d['roles']['fit']|d['roles']['validation'])
        foldrows=[r for r in fr['rows'] if r['fold']==fold];loaded={};previous={};logits={b:c.np.empty(len(ix),c.np.float32) for b in c.BODIES}
        for row in [r for r in models if r['fold']==fold]:
            body=row['procedure'];p=OLD/'outer_scores_private'/f'{body}_{fold}.pkl'
            assert c.sha(p)==oldseal['cells'][f'{body}_{fold}'];q=c.read(p)
            assert c.np.array_equal(q['indices'],ix) and q['body_sha256']==row['body_sha256'] and q['matrix_sha256']==row['matrix_sha256']
            assert q['threshold']==row['threshold'];previous[body]=q
            cp=c.torch.load(row['body'],map_location='cpu',weights_only=False);m=Model(body,fold,cp['model']).cuda().eval()
            if row['matrix']:
                state=c.torch.load(row['matrix'],map_location='cpu',weights_only=False)
                assert state['selection']['threshold']==row['threshold'] and state['binding']['body_checkpoint_sha256']==row['body_sha256']
                with c.torch.no_grad():m.matrix.copy_(state['matrix'])
            loaded[body]=m
        maximum={b:0. for b in c.BODIES};windows=0
        for pid in sorted(set(meta['patient'][ix])):
            jj=c.np.flatnonzero(meta['patient'][ix]==pid);ai=ix[jj]
            path=FINAL/'margin_patients_private'/f'fold{fold}'/(c.io.private_key(pid)+'.pkl')
            pb=dict(freeze=fh,fold=fold,archive_indices=ai.tolist(),waveform_sha256=store.plans[str(pid)]['sha256'])
            if path.exists():q=c.read(path);assert q['binding']==pb
            else:
                raw,mask,times=store.get(pid,meta['channel'][ai]);features,audit=loaded['NATIVE'].encode(raw,mask,times)
                with c.torch.inference_mode():score={b:m.forward_stats(features).cpu().numpy() for b,m in loaded.items()}
                errors={}
                for b in c.BODIES:
                    pn=c.probability(score[b]);old=previous[b]['score_nez'][jj]
                    errors[b]=float(c.np.max(c.np.abs(pn-old)));assert errors[b]<=1e-6
                    assert c.np.array_equal(pn<previous[b]['threshold'],old<previous[b]['threshold'])
                q=dict(binding=pb,logits=score,probability_errors=errors,original_decisions_exact=True,
                       valid_windows=int(mask.sum()),normalization_audit=audit)
                c.save(path,q)
            assert q['original_decisions_exact'];windows+=q['valid_windows']
            for b in c.BODIES:
                logits[b][jj]=q['logits'][b];maximum[b]=max(maximum[b],q['probability_errors'][b])
            c.write(PUBLIC/'INFERENCE_PROGRESS.json',dict(status='RECOVERING_FROZEN_MARGINS_NO_OUTER_TRUTH',fold=fold,
                completed_patients_in_fold=len(list((FINAL/'margin_patients_private'/f'fold{fold}').glob('*.pkl')))))
        for row in foldrows:
            x=logits[row['body']];pred=predict(row['config'],x,meta['patient'][ix]);key=f"{row['procedure']}_{fold}"
            path=FINAL/'outer_scores_private'/(key+'.pkl');result=dict(freeze=fh,configuration=row,indices=ix,logit_nez=x,pred_ez=pred)
            if path.exists():
                old=c.read(path);assert old['freeze']==fh and old['configuration']==row
                assert c.np.array_equal(old['logit_nez'],x) and c.np.array_equal(old['pred_ez'],pred)
            else:c.save(path,result)
            cells[key]=c.sha(path)
        parity.append(dict(fold=fold,patients=len(set(meta['patient'][ix])),channels=len(ix),valid_windows=windows,
                           maximum_probability_error=maximum,all_original_binary_decisions_exact=True))
        print('RAW_DECISION_OUTER_FOLD_SEALED',fold,len(cells),flush=True)
        del loaded;gc.collect();c.torch.cuda.empty_cache()
    assert len(cells)==45 and sum(r['valid_windows'] for r in parity)==1471965
    c.freeze(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,cells=cells,all45_scores_before_full_Outer_truth=True))
    c.write(PUBLIC/'OUTER_STATUS.json',dict(status='ALL45_FROZEN_MARGIN_DECISION_CELLS_SEALED',cells=45,parity=parity,
        patients_per_procedure=80,channels_per_procedure=7635,valid_windows=1471965,
        score_seal_sha256=c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),fullOuter_truth_read=False))


def labels_after_seal():
    status=c.jr(PUBLIC/'OUTER_STATUS.json');assert status['status']=='ALL45_FROZEN_MARGIN_DECISION_CELLS_SEALED'
    assert status['score_seal_sha256']==c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    seal=c.jr(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');assert len(seal['cells'])==45 and seal['all45_scores_before_full_Outer_truth']
    assert seal['freeze']==c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    meta,banks=c.io.identity_and_banks(labels=False)
    archive=c.Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert c.sha(archive)==c.read(c.io.BASE/'RECONSTRUCTION_PRIVATE.pkl')['binding']['archive']
    with c.np.load(archive,allow_pickle=True) as z:
        assert c.np.array_equal(z['patient'],meta['patient']) and c.np.array_equal(z['channel'],meta['channel']);y=1-z['y']
    assert len(y)==7635 and y.sum()==1743
    return seal,meta,banks,y


def report():
    setup();seal,meta,banks,y=labels_after_seal();rows=[];pooled=[];precision=[]
    for procedure in PROCEDURES:
        indices=[];truth=[];margins=[];predictions=[]
        for fold in range(1,6):
            path=FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl';assert c.sha(path)==seal['cells'][f'{procedure}_{fold}']
            q=c.read(path);ix=q['indices'];x=q['logit_nez'];pred=q['pred_ez'];assert q['freeze']==seal['freeze']
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test']
            indices.extend(ix);truth.extend(y[ix]);margins.extend(x);predictions.extend(pred)
            for pid in sorted(set(meta['patient'][ix])):
                j=meta['patient'][ix]==pid;ai=ix[j]
                metric=patient_metrics(y[ai],x[j],pred[j],meta['channel'][ai])
                rows.append(dict(procedure=procedure,fold=fold,patient=str(pid),center=str(meta['center'][ai][0]),**metric))
                if procedure.endswith('__ORIGINAL'):
                    pn=c.probability(x[j]);pp=1-pn.astype(float);legacy=dict(**c.wave.parent.ev.classification(y[ai],pp,pred[j]),**c.wave.parent.ev.ranking(y[ai],pp,meta['channel'][ai]))
                    precision.append(dict(body=q['configuration']['body'],fold=fold,patient=str(pid),channels=len(ai),saturated=int(((pn==0)|(pn==1)).sum()),
                        **{k+'_legacy':legacy[k] for k in ['auroc','auprc','mrr','hit_at_1']},**{k+'_margin':metric[k] for k in ['auroc','auprc','mrr','hit_at_1']}))
        assert len(indices)==len(set(indices))==7635 and set(indices)==set(range(7635))
        pm=patient_metrics(c.np.array(truth),c.np.array(margins),c.np.array(predictions),c.np.arange(7635).astype(str))
        pooled.append(dict(procedure=procedure,**{k:pm[k] for k in ['macro_f1','f1','nez_f1','accuracy','auroc','auprc','ez_fraction_bias']}))
    frame=c.pd.DataFrame(rows);assert len(frame)==720 and all(frame[frame.procedure==p].patient.nunique()==80 for p in PROCEDURES)
    for body,expected in [('NATIVE',.6304930529317453),('PRECOMPAND_RESIDUAL',.6314324049383006)]:
        assert abs(frame[frame.procedure==body+'__ORIGINAL'].macro_f1.mean()-expected)<1e-12
        ranks=frame[frame.procedure.str.startswith(body+'__')].groupby(['fold','patient'])[['auroc','auprc','mrr','hit_at_1']].std(ddof=0)
        assert c.np.max(ranks.to_numpy())<1e-12
    c.save(FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl',rows)
    frame.groupby('procedure')[c.METRICS].mean().reset_index().to_csv(PUBLIC/'OUTER_PATIENT_MEAN.csv',index=False)
    c.pd.DataFrame(pooled).to_csv(PUBLIC/'OUTER_POOLED_MIXED_FOLD_MARGINS.csv',index=False)
    fold=frame.groupby(['procedure','fold'])[c.METRICS].mean().reset_index();fold.to_csv(PUBLIC/'OUTER_FOLDS.csv',index=False)
    fold.groupby('procedure')[c.METRICS].std(ddof=1).reset_index().to_csv(PUBLIC/'OUTER_FOLD_SD.csv',index=False)
    absolute=[];paired=[];sources=[];coverage=[]
    for procedure,g in frame.groupby('procedure'):
        absolute.extend([dict(procedure=procedure,**r) for r in c.interval(g.to_dict('records'))])
        coverage.extend([dict(procedure=procedure,metric=m,finite_patients=int(c.np.isfinite(g[m]).sum()),total_patients=80) for m in c.METRICS])
    for (procedure,center),g in frame.groupby(['procedure','center']):
        n=g.patient.nunique();sources.append(dict(procedure=procedure,center=center,patients=n,suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else c.np.nan for m in c.METRICS}))
    pairs=[(p,'NATIVE__ORIGINAL') for p in PROCEDURES if p!='NATIVE__ORIGINAL']
    pairs.extend([(p,'PRECOMPAND_RESIDUAL__ORIGINAL') for p in c.PROCEDURES if p.startswith('PRECOMPAND') and not p.endswith('__ORIGINAL')])
    for left,right in pairs:
        m=frame[frame.procedure==left].merge(frame[frame.procedure==right],on=['patient','center','fold'],suffixes=('_l','_r'),validate='one_to_one')
        diff=c.pd.DataFrame(dict(patient=m.patient,**{k:m[k+'_l']-m[k+'_r'] for k in c.METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in c.interval(diff.to_dict('records'))])
    c.pd.DataFrame(absolute).to_csv(PUBLIC/'OUTER_ABSOLUTE_CI.csv',index=False);c.pd.DataFrame(paired).to_csv(PUBLIC/'OUTER_PAIRED_CI.csv',index=False)
    c.pd.DataFrame(sources).to_csv(PUBLIC/'OUTER_SOURCE.csv',index=False);c.pd.DataFrame(coverage).to_csv(PUBLIC/'OUTER_METRIC_COVERAGE.csv',index=False)
    pr=c.pd.DataFrame(precision);cols=[n for n in pr if n not in ['patient','fold','body']]
    pr.groupby('body')[cols].mean().reset_index().to_csv(PUBLIC/'OUTER_RANKING_PRECISION_AUDIT.csv',index=False)
    points={p:float(frame[frame.procedure==p].macro_f1.mean()) for p in PROCEDURES}
    c.write(PUBLIC/'FINAL_TERMINAL.json',dict(status='COMPLETE_NINE_RAW_DECISION_PROCEDURES',patient_equal_macro_f1=points,
        point_over_065={p:v>.65 for p,v in points.items()},goal_point_established=any(v>.65 for v in points.values()),
        all_predeclared_procedures_reported=True,neural_updates=0,postOuter_tuning=False,outer_oracle=False,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False))
    print('RAW_DECISION_FINAL_COMPLETE',points,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['calibrate','test','report']);args=parser.parse_args()
    dict(calibrate=calibrate,test=test,report=report)[args.action]()
