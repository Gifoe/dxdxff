"""Freeze all scalar CAL fits and all HELD scores before any HELD metrics."""
import subprocess
import sys
import common as c
from decoders import fit,predict,rows


def main():
    subprocess.run([sys.executable,'-m','pytest','-q',str(c.HERE/'test_decoders.py')],check=True)
    binding=c.setup();contexts=c.contexts();assert len(contexts)==15
    configs=[];loaded={};public=[]
    for ctx in contexts:
        d=ctx['data'];cal=ctx['roles']['cal'];f,s=ctx['fold'],ctx['screen']
        for body in c.BODIES:
            q=c.body_scores(ctx,body);loaded[(f,s,body)]=q
            original=dict(family='ORIGINAL',value=float(q['theta']))
            for decoder in c.DECODERS:
                config,summary=(original,{}) if decoder=='ORIGINAL' else fit(decoder,q['cal'],d['y'][cal],d['patient'][cal],original)
                row=dict(fold=f,screen=s,body=body,decoder=decoder,procedure=body+'__'+decoder,
                         config=config,source=q['source'],cal_archive_indices=d['archive_indices'][cal].tolist())
                configs.append(row);public.append(dict(fold=f,screen=s,body=body,decoder=decoder,
                    parameter=config['value'],mode=config.get('mode','scalar'),**summary))
    assert len(configs)==120
    c.freeze(c.ROOT/'CAL_CONFIGS_PRIVATE.json',dict(input=binding,configs=configs,new_scalar_fits=90,neural_updates=0))
    cells=[]
    for ctx in contexts:
        d=ctx['data'];held=ctx['roles']['held'];f,s=ctx['fold'],ctx['screen']
        for row in [r for r in configs if (r['fold'],r['screen'])==(f,s)]:
            logit=loaded[(f,s,row['body'])]['held'];pred=predict(row['config'],logit,d['patient'][held])
            path=c.ROOT/'held_scores_private'/f"{f}_{s}_{row['procedure']}.pkl"
            q=dict(input=binding,config_seal=c.sha(c.ROOT/'CAL_CONFIGS_PRIVATE.json'),configuration=row,
                   archive_indices=d['archive_indices'][held],logit_nez=logit,pred_ez=pred)
            if path.exists():
                old=c.read(path);assert old['input']==binding and old['configuration']==row
                assert c.np.array_equal(old['logit_nez'],logit) and c.np.array_equal(old['pred_ez'],pred)
            else:c.save(path,q)
            cells.append(dict(fold=f,screen=s,procedure=row['procedure'],sha256=c.sha(path)))
    c.freeze(c.ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json',dict(input=binding,configuration_seal=c.sha(c.ROOT/'CAL_CONFIGS_PRIVATE.json'),
            cells=cells,all_scores_before_HELD_metrics=True,oracle_not_yet_computed=True))
    frame=[]
    for ctx in contexts:
        for row in [r for r in configs if (r['fold'],r['screen'])==(ctx['fold'],ctx['screen'])]:
            path=c.ROOT/'held_scores_private'/f"{ctx['fold']}_{ctx['screen']}_{row['procedure']}.pkl";q=c.read(path)
            frame.extend([dict(fold=ctx['fold'],screen=ctx['screen'],procedure=row['procedure'],
                              **r) for r in rows(ctx['data'],ctx['roles']['held'],q['logit_nez'],q['pred_ez'])])
        print('DECISION_HELD_PATH_SCORED',ctx['fold'],ctx['screen'],flush=True)
    assert len(frame)==2040
    c.save(c.ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl',frame)
    c.pd.DataFrame(public).to_csv(c.PUB/'CALIBRATION_AUDIT.csv',index=False)
    c.write(c.PUB/'ADMISSION.json',dict(status='PASS',critical_path_tests=4,neural_updates=0,source_gate='PASS'))
    c.write(c.PUB/'RUN_STATUS.json',dict(status='ALL120_DECODER_CELLS_SEALED',new_scalar_fits=90,neural_fits=0,
             score_seal_sha256=c.sha(c.ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json'),patient_metric_rows=len(frame),new_Outer_passes=0))
    print('DECISION_INTERNAL120_SCORED',flush=True)


if __name__=='__main__':main()

