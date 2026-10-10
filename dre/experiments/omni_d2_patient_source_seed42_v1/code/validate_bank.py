"""Fresh C0 parity hard gate; normalization and full label-free bank geometry."""
from common import *
def main():
    cfg=lock();mf=manifest();rows=[];maxdiff=0.;digests={};unknown=0
    for fold in range(1,6):
        for row in mf.itertuples():
            p=str(row.patient_name);path=ROOT/f'embeddings/fold_{fold}/{key(p)}.npz';seal=json.loads(path.with_suffix('.json').read_text())
            assert seal['sha256']==sha(path) and seal['binding']['protocol']==sha(PROTOCOL)
            assert seal['binding']['checkpoints']==cfg['backbone_checkpoints'];digests[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as d:
                assert str(d['checkpoint_sha256'])==cfg['backbone_checkpoints'][str(fold)] and bool(d['outer_held'])==(fold==row.fold)
                names=d['channels'];h=d['embedding'];l=d['logits'];assert h.shape==(len(names),32) and np.isfinite(l).all() and np.isfinite(h).all()
                assert np.array_equal(d['segment_count'],d['edf_count']*30)
                with np.load(ROOT/'labels'/f'{key(p)}.npz',allow_pickle=False) as vault:assert np.array_equal(names,vault['channels'])
                z=normalized(h);assert z.shape==h.shape
                if fold!=row.fold:continue
                original=OLD/f'oof_patient_cache/fold_{fold}/baseline/{key(p)}.npz'
                with np.load(original,allow_pickle=False) as old:
                    assert np.array_equal(names,old['channels']);y=old['labels'];before=old['scores'];v=y>=0
                    diff=float(abs(sigmoid(l)-before).max());maxdiff=max(maxdiff,diff)
                    assert diff<=cfg['baseline_gate']['maximum_score_error']
                    now=metric(y[v],sigmoid(l[v]));past=metric(y[v],before[v])
                    assert all(abs(now[m]-past[m])<=1e-12 for m in METRICS),'Fresh C0 metric/ranking mismatch: engineering block, never tune'
                    rows.append(now);unknown+=int((y<0).sum())
    means={m:float(np.mean([r[m] for r in rows])) for m in METRICS}
    for m,expected in cfg['baseline_gate'].items():
        if m in means:assert abs(means[m]-expected)<1e-12
    write_json(ROOT/'BANK_GATE_PRIVATE.json',{'status':'PASS','protocol':sha(PROTOCOL),'banks':digests})
    write_json(ROOT/'public/BASELINE_REPRODUCTION.json',{'status':'PASS','fresh_metrics':means,'archived_score_max_abs_error':maxdiff,'all_patient_metric_errors_le1e12':True,'no_frozen_score_substitution':True,'CNN_retrained':False})
    write_json(ROOT/'public/PATIENT_RELATIVE_AUDIT.json',{'status':'PASS','patient_fold_cells':620,'unknown_good_canonical_OOF_channels':unknown,'normalization_labelblind':True,'all_unknown_good_retained':True,'zero_variance_rule':'population variance +1e-5','finite':True})
    print('FRESH_C0_AND620_REPRESENTATION_GATE_PASS',flush=True)
if __name__=='__main__':main()
