"""Post-run immutable provenance, population, selection and publication gates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from source import *
from transforms import *

def main():
    torch.set_num_threads(2)
    validated=json.loads((ROOT/'INPUTS_VALIDATED_PRIVATE.json').read_text())
    assert validated['status']=='PASS' and validated['protocol']==sha(PROTOCOL)
    for path,h in validated['original_hashes'].items():assert sha(path)==h,'Historical input mutation'
    freeze_path=ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json';freeze=json.loads(freeze_path.read_text())
    assert freeze['status']=='ALL_CONTROLS_FROZEN_BEFORE_OUTER' and freeze['protocol']==sha(PROTOCOL) and freeze['code']==code_hashes()
    assert freeze['historical_freeze']==cfg()['historical_freeze_sha256']
    assert freeze['historical_replay_sha256']==sha(ROOT/'HISTORICAL_ROWS_PRIVATE.pt')
    assert len(freeze['choices'])==10 and len(freeze['transform_hashes'])==640
    for path,h in freeze['transform_hashes'].items():assert sha(path)==h
    mf=manifest();inner_runs=refits=0
    for choice in freeze['choices']:
        k=choice['fold'];arm=choice['method'];a=plan(mf,k)
        ck=ROOT/f'fold_{k}/{arm}/SELECTED_HEAD_PRIVATE.pt'
        assert sha(ck)==choice['checkpoint_sha256'] and not choice['outer_labels_used']
        st=torch.load(ck,map_location='cpu',weights_only=False)
        assert st['threshold']==choice['threshold'] and st['selected_epoch']==choice['selected_epoch']
        model=Head('C1');model.load_state_dict(st['model']);assert sum(t.numel() for t in model.parameters())==35
        for inner in range(3):
            last=torch.load(ROOT/f'fold_{k}/{arm}/inner_{inner}/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            train=[p for p in a['fit'] if a['inner'][p]!=inner];val=[p for p in a['fit'] if a['inner'][p]==inner]
            assert last['epoch']==50 and len(last['history'])==50 and set(last['oof'])==set(range(1,51))
            assert all(set(pred)==set(val) for pred in last['oof'].values())
            assert last['binding']['training_ids']==digest(train) and last['binding']['inner_ids']==digest(val)
            assert not set(train)&set(a['held']) and not set(val)&set(a['held']);inner_runs+=1
            if arm=='C1-ABS':
                path=ROOT/f'transforms/fold_{k}/ABS_inner_{inner}.pt';sc=torch.load(path,map_location='cpu',weights_only=False)
                assert sc['fit_patient_digest']==digest(train) and sc['fit_patients']==len(train)
        refit=torch.load(ROOT/f'fold_{k}/{arm}/refit/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert refit['epoch']==choice['selected_epoch'] and state_hash(refit['model'])==state_hash(st['model'])
        assert refit['binding']['training_ids']==digest(a['fit']);refits+=1
        if arm=='C1-ABS':
            path=ROOT/f'transforms/fold_{k}/ABS_full_FIT.pt';assert sha(path)==choice['scaler_sha256']
            sc=torch.load(path,map_location='cpu',weights_only=False);assert sc['fit_patient_digest']==digest(a['fit'])
    assert inner_runs==30 and refits==10
    for k in range(1,6):
        a=plan(mf,k)
        for role in ['FIT','HELD']:
            for p in a['fit' if role=='FIT' else 'held']:
                with np.load(ROOT/f'transforms/fold_{k}/{role}/{key(p)}.npz',allow_pickle=False) as v:
                    ix=v['permutation'];assert np.array_equal(ix,permutation(k,role,key(p),len(ix)))
                    assert not np.any(ix==np.arange(len(ix))) and sorted(ix.tolist())==list(range(len(ix)))
    ledger=torch.load(ROOT/'FINAL_LEDGER_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert ledger['freeze']==sha(freeze_path) and len(ledger['prediction_hashes'])==496
    for path,h in ledger['prediction_hashes'].items():assert sha(path)==h
    frame=pd.DataFrame(ledger['rows']);diag=pd.DataFrame(ledger['diagnostics'])
    assert len(frame)==496 and len(diag)==248 and set(frame.method)=={'C0','C1-PR','C1-ABS','C1-SHUFFLE'}
    for _,g in frame.groupby('method'):
        assert g.patient.nunique()==124 and set(g.patient)==set(mf.patient_name) and g.units.sum()==10202
        assert g.center.value_counts().to_dict()==cfg()['cohort']['sources']
    original=pd.DataFrame(torch.load(ROOT/'HISTORICAL_ROWS_PRIVATE.pt',map_location='cpu',weights_only=False)['rows'])
    frozen_rows=frame[frame.method.isin(['C0','C1-PR'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(original,frozen_rows,check_exact=True)
    public=ROOT/'public';boot=pd.read_csv(public/'PAIRED_PATIENT_BOOTSTRAP.csv')
    assert len(boot)==95 and boot.contrast.nunique()==5 and (boot.draws==10000).all() and (boot.seed==42).all() and (boot.patients==124).all()
    assert len(pd.read_csv(public/'INNER_EPOCH_METRICS.csv'))==500
    assert len(pd.read_csv(public/'INNER_SELECTION.csv'))==10
    scale=json.loads((public/'SCALE_ONLY_INVARIANCE_AUDIT.json').read_text())
    assert scale['status']=='PASS' and scale['max_float64_constant_identity_error']<1e-10 and scale['formal_max_math_rank_error']<=1e-12
    report=(public/'FINAL_REPORT.md').read_text()
    for n in range(1,27):assert f'{n}.' in report
    status=json.loads((public/'RUN_STATUS.json').read_text());gate=json.loads((public/'VALIDATION_GATE.json').read_text())
    assert status['status']=='COMPLETE' and status['terminal']==gate['terminal'] and status['private_ledger_sha256']==sha(ROOT/'FINAL_LEDGER_PRIVATE.pt')
    assert gate['protocol_sha256']==sha(PROTOCOL) and gate['PR_mechanism_gate']['integrity']
    forbidden={'patient','patient_name','patient_id','channel','channels','label','y','logit','logits','score','scores','embedding'}
    files=list(public.glob('*'))
    for f in files:
        if f.suffix=='.csv':
            table=pd.read_csv(f);assert not set(table.columns)&forbidden
            # Explicit non-applicable metadata is legal; metric NaNs are not.
            allowed_null=set()
            if f.name=='INNER_SELECTION.csv':
                assert table.scaler_sha256.isna().equals(table.method=='C1-SHUFFLE')
                allowed_null.add('scaler_sha256')
            if f.name in {'INFERENCE_SHUFFLE_INTERVENTION.csv','SCALE_ONLY_CLASSIFICATION.csv'}:
                assert table.center.isna().equals(table.patients==124)
                allowed_null.add('center')
            assert not table.drop(columns=list(allowed_null)).isna().any().any(),f.name
            assert np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all(),f.name
        if f.suffix in {'.csv','.json','.md'}:
            text=f.read_text()
            assert not any(p in text or key(p) in text for p in mf.patient_name),'Individual identifier leaked'
    # Check again after final aggregates to certify the historical inputs stayed byte-identical.
    for path,h in validated['original_hashes'].items():assert sha(path)==h
    write_json(public/'COMPLETION_VERIFICATION.json',{'status':'PASS','protocol_sha256':sha(PROTOCOL),'code_sha256':code_hashes(),'new_freeze_sha256':sha(freeze_path),'historical_immutable_files':len(validated['original_hashes']),'historical_bank_cells':620,'formal_inner_runs':30,'formal_refits':10,'patients_per_arm':124,'labeled_units_per_arm':10202,'new_prediction_cells':248,'diagnostic_cells':248,'registered_bootstrap_rows':95,'paired_draws':10000,'private_predictions_never_published':True,'aggregate_sha256':{p.name:sha(p) for p in files if p.name!='COMPLETION_VERIFICATION.json'}})
    print('OMNI_PR_COMPLETION_AND_AGGREGATE_PRIVACY_PASS',flush=True)
if __name__=='__main__':main()
