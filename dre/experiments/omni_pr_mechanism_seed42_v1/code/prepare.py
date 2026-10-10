"""Full historical reproduction and immutable input audit, no new head training."""
from source import *
from transforms import transform
def main():
    torch.set_num_threads(2);ROOT.mkdir(exist_ok=True);(ROOT/'public').mkdir(exist_ok=True)
    pins={'BANK_GATE_PRIVATE.json':'historical_bank_gate_sha256','INPUTS_PRIVATE.json':'historical_inputs_sha256','C0_ARCHIVE_PRIVATE.json':'historical_archive_sha256','FINAL_OUTER_LEDGER_PRIVATE.pt':'historical_ledger_sha256','public/COMPLETION_VERIFICATION.json':'historical_completion_sha256'}
    snapshots={str(PROTOCOL):sha(PROTOCOL)}
    for path,pin in pins.items():assert sha(HIST/path)==cfg()[pin];snapshots[str(HIST/path)]=sha(HIST/path)
    gate=json.loads((HIST/'BANK_GATE_PRIVATE.json').read_text());assert gate['status']=='PASS' and len(gate['banks'])==620
    for p,h in gate['banks'].items():assert sha(p)==h;snapshots[p]=h
    inputs=json.loads((HIST/'INPUTS_PRIVATE.json').read_text())
    for p,h in inputs['original_hashes'].items():assert sha(p)==h;snapshots[p]=h
    assert len({p for paths in inputs['records'].values() for p in paths})==137
    mf=manifest();assert len(mf)==124 and mf.center.value_counts().to_dict()==cfg()['cohort']['sources']
    assert mf.n_pathological_channels.sum()==1113 and mf.n_normal_channels.sum()==9089 and mf.n_labeled_channels.sum()==10202
    old=torch.load(HIST/'FINAL_OUTER_LEDGER_PRIVATE.pt',map_location='cpu',weights_only=False)
    for p,h in old['prediction_hashes'].items():assert sha(p)==h;snapshots[p]=h
    totals=0;rows=[];error=0.;metricerror=0.
    for fold in range(1,6):
        assignment=plan(mf,fold);saved=torch.load(HIST/f'fold_{fold}/ASSIGNMENT_PRIVATE.pt',map_location='cpu',weights_only=False);assert assignment==saved
        head=pr_head(fold);hp=HIST/f'fold_{fold}/C1/SELECTED_HEAD_PRIVATE.pt';snapshots[str(hp)]=sha(hp)
        cnn=historical.OLD/f'checkpoints/fold_{fold}/baseline/last.pt';assert sha(cnn)==historical.lock()['backbone_checkpoints'][str(fold)]
        data=bank(fold,assignment['held']);prdata=transform(data,assignment['held'],'C1-PR',fold,'HELD')
        replay=predict(head,prdata,assignment['held'],assignment['real'])
        for p in assignment['held']:
            labelpath=HIST/'labels'/f'{key(p)}.npz';snapshots[str(labelpath)]=sha(labelpath)
            with np.load(labelpath,allow_pickle=False) as v:
                assert np.array_equal(v['channels'],data[p]['channels']);y=v['labels'];valid=y>=0
            orig=historical.OLD/f'oof_patient_cache/fold_{fold}/baseline/{key(p)}.npz'
            with np.load(orig,allow_pickle=False) as v:
                assert np.array_equal(v['labels'],y) and np.array_equal(v['channels'],data[p]['channels'])
                assert np.array_equal(sigmoid(data[p]['base']),v['scores'])
            totals+=int(valid.sum())
            for arm,ll in [('C0',data[p]['base']),('C1-PR',replay[p])]:
                oldarm='C0' if arm=='C0' else 'C1';cache=HIST/f'outer_predictions/{oldarm}/{key(p)}.npz'
                with np.load(cache,allow_pickle=False) as v:
                    assert np.array_equal(v['channels'],data[p]['channels']);archived=v['logits'].copy()
                error=max(error,float(abs(ll-archived).max()));assert np.array_equal(ll,archived)
                tau=selected(fold,arm)['threshold'];q=metric(y[valid],sigmoid(ll[valid]),tau);fixed=classification(y[valid],sigmoid(ll[valid]),.5)
                target=next(r for r in old['patient_rows'] if r['patient']==p and r['method']==oldarm)
                for m in METRICS:metricerror=max(metricerror,abs(q[m]-target[m]))
                assert tau==target['threshold'] and metricerror<=1e-12
                rows.append({'patient':p,'fold':fold,'center':str(mf.loc[mf.patient_name==p,'center'].iloc[0]),'method':arm,'threshold':tau,'units':int(valid.sum()),**q,**{m+'_0_5':fixed[m] for m in CLASS}})
    assert totals==10202 and error==0
    expected={'C0':(.7621630659358641,.47725499096051377,.6103517057230773),'C1-PR':(.8118233539960643,.5235271351033821,.6558193618934671)}
    summaries=aggregate(pd.DataFrame(rows),['method'],METRICS+[m+'_0_5' for m in CLASS])
    for r in summaries:
        for m,t in zip(['auroc','ap','macro_f1'],expected[r['method']]):assert abs(r[m]-t)<=1e-12
    save_torch(ROOT/'HISTORICAL_ROWS_PRIVATE.pt',{'rows':rows,'historical_ledger':sha(HIST/'FINAL_OUTER_LEDGER_PRIVATE.pt')})
    write_json(ROOT/'INPUTS_VALIDATED_PRIVATE.json',{'status':'PASS','protocol':sha(PROTOCOL),'code':code_hashes(),'original_hashes':snapshots,'historic_rows_sha256':sha(ROOT/'HISTORICAL_ROWS_PRIVATE.pt')})
    write_json(ROOT/'public/HISTORICAL_REPRODUCTION.json',{'status':'PASS','max_logit_error':error,'max_patient_metric_error':metricerror,'references':summaries,'historical_checkpoint_and_thresholds_unchanged':True,'no_PR_retraining':True})
    write_json(ROOT/'public/COHORT_AND_SPLIT_AUDIT.json',{'status':'PASS',**cfg()['cohort'],'manifest_sha256':sha(historical.OLD/'PATIENT_5FOLD_MANIFEST_PRIVATE.csv'),'five_original_folds':True,'inner_split_exact_private_assignment_parity':True,'label_orientation':'normal0 pathological1 unknown-1; ordered good/outcome/resection/soz rule unchanged'})
    write_json(ROOT/'public/EMBEDDING_BANK_AUDIT.json',{'status':'PASS','cells':620,'dimension':32,'original_bank_gate_sha256':sha(HIST/'BANK_GATE_PRIVATE.json'),'all_cell_hashes_verified':True,'raw_reextraction':False,'CNN_training':False,'correct_outer_fold_CNN_only':True})
    print('HISTORICAL_C0_PR_EXACT_REPLAY_AND620_BANK_HASH_PASS',flush=True)
if __name__=='__main__':main()
