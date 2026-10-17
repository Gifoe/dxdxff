"""Read-only checkpoint adapter replay; no labels or development metrics."""
from common import *

def main():
    binding,core,fac,records,rec,banks,old=setup();models,studies=catalogue(old);inputs=store();checks=[]
    for r in models:
        d=bank(r['fold'],banks);va=d['val'];pid=sorted(set(d['patient'][va]))[0]
        jj=np.flatnonzero(d['patient'][va]==pid);ix=va[jj];ss=scale(r['fold'])
        m,v=make(r,core,fac,records,old)
        assert np.array_equal(v['archive_indices'],d['archive_indices'][va])
        before=state_hash(m.state_dict())
        out=base.predict(m,d,ix,inputs,ss);error=float(np.max(np.abs(out-v['logit'][jj])))
        assert error<=1e-6
        assert np.array_equal(probability(out)<r['threshold'],probability(v['logit'][jj])<r['threshold'])
        assert state_hash(m.state_dict())==before
        checks.append(dict(arm=r['arm'],fold=r['fold'],maximum_logit_error=error,threshold_decisions_exact=True,one_head=True,parameters=sum(p.numel() for p in m.parameters())))
        print('FROZEN_ADAPTER_REPLAY_PASS',r['arm'],r['fold'],error,flush=True)
        del m;gc.collect();torch.cuda.empty_cache()
    admission=dict(status='PASS',sources=sources(),checks=checks,no_labels_or_development_metrics_used=True,no_new_fits=True)
    freeze(ROOT/'ADMISSION_PRIVATE.json',admission)
    frozen=dict(status='ALL55_CELLS_FROZEN_BEFORE_NEW_OUTER',input=binding,models=models,studies=studies,sources=sources(),
        preprocessing=preprocessing(banks),old_reference_seal=sha(BASE_SEAL),old_reference_files=jr(BASE_SEAL)['files'],
        admission_sha256=sha(ROOT/'ADMISSION_PRIVATE.json'),loaded_source_closure=loaded_closure(),candidate_selection='NONE_ALL10_INCLUDED',
        development_improvement_gates_unchanged=True,scientific_status='EXPLORATORY_REPEATED_TEST_MULTI_CANDIDATE_AUDIT')
    freeze(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',frozen)
    csv_out('ADAPTER_REPLAY_CHECKS',checks)
    csv_out('FROZEN_MODELS',[dict(arm=r['arm'],fold=r['fold'],epoch=r['epoch'],threshold=r['threshold'],checkpoint_sha256=r['checkpoint_sha256'],validation_sha256=r['validation_sha256'],study=r['study']) for r in models])
    write(PUBLIC/'ADMISSION.json',dict(status='PASS',cells=55,maximum_logit_error=max(r['maximum_logit_error'] for r in checks),
        freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),no_labels_or_development_metrics_used=True,no_new_fits=True,old_development_gates_unchanged=True,
        sources=sources(),historical_study_hashes=studies))
    print('ALL55_CHECKPOINTS_AND_THRESHOLDS_FROZEN_BEFORE_NEW_OUTER',flush=True)

if __name__=='__main__':main()
