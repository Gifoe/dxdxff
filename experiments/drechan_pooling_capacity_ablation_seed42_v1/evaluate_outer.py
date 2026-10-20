"""Only after all25 selections freeze; seal all scores before pooled reporting."""
from common import *

COUNTS = ['valid_windows', 'relative_active_coordinates', 'relative_used_coordinates',
          'global_fallback_coordinates', 'relative_singleton_valid_windows',
          'mismatch_changed_valid_windows', 'mismatch_eligible_valid_windows', 'mismatch_singleton_valid_windows']


def reference_coverage(r, root):
    path = root/'public/RAW_COVERAGE.csv'
    with path.open(encoding='utf-8', newline='') as f:
        old = next(q for q in csv.DictReader(f) if q['arm'] == r['source_arm'] and int(q['fold']) == r['fold'])
    counts = {k:int(old.get(k, 0)) for k in COUNTS}
    counts['relative_singleton_valid_windows'] = int(old.get('relative_singleton_valid_windows', old.get('singleton_valid_windows', 0)))
    return counts, sha(path)


def main():
    binding, core, fac, records, rec, banks = setup(); fr = frozen(); assert fr['input'] == binding
    fh = sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'); inputs, seal, coverage, meta = store(), {}, [], rec['meta']
    for r in fr['cells']:
        arm, seed, fold = r['arm'], r['seed'], r['fold']; key = key_for(arm, seed, fold)
        d, ss = bank(fold, banks, arm), scale(fold)
        ix = np.flatnonzero(np.isin(meta['patient'], list(d['roles']['test'])))
        assert preprocessor(d, ss) == r['preprocessor']
        path = ROOT/'scores_private'/(key+'.pkl'); reused, coverage_sha = 0, ''
        if r['reused']:
            q, source_sha, root = reference_score(r)
            assert np.array_equal(q['indices'], ix)
            result = dict(freeze=fh, indices=ix, score_nez=q['score_nez'], threshold=r['threshold'],
                          checkpoint=r['checkpoint_sha256'], reused_source_sha256=source_sha)
            counts, coverage_sha = reference_coverage(r, root)
        else:
            x = query_features(rec['R14'][ix], meta['patient'][ix], d['pre'], arm, core)
            assert x.shape == (len(ix), 14) and np.isfinite(x).all()
            td = dict(x=x, patient=meta['patient'][ix], channel=meta['channel'][ix])
            m = make(core, fac, records, fold, arm, seed)
            m.load_state_dict(torch.load(r['checkpoint'], weights_only=False, map_location='cpu')['model'])
            m.eval(); m.collect_guidance_audit = True; before = state_hash(m.state_dict())
            out, counts = np.empty(len(ix), np.float32), dict.fromkeys(COUNTS, 0)
            for n, pid in enumerate(sorted(d['roles']['test']), 1):
                jj = np.flatnonzero(td['patient'] == pid)
                cb = dict(freeze=fh, checkpoint=r['checkpoint_sha256'], indices=legacy.array_sha(ix[jj]),
                          input=legacy.array_sha(x[jj]), scale=legacy.array_sha(ss.cpu().numpy()))
                cache = ROOT/'patient_scores_private'/arm/f'seed{seed}'/f'fold{fold}'/(hashlib.sha256(str(pid).encode()).hexdigest()+'.pkl')
                if cache.exists():
                    q = read(cache); assert q['binding'] == cb; reused += 1
                else:
                    with torch.inference_mode():
                        logit, info = forward(m, td, jj, inputs, ss, fold, binding)
                        p = probability(logit.cpu().numpy())
                    q = dict(binding=cb, score_nez=p, info=info); save(cache, q)
                assert q['score_nez'].shape == (len(jj),) and np.isfinite(q['score_nez']).all()
                out[jj] = q['score_nez']
                for k in counts: counts[k] += q['info'].get(k, 0)
                write(PUB/'PROGRESS.json', dict(status='FROZEN_OUTER_INFERENCE', arm=arm, seed=seed, fold=fold,
                      patient_number=n, fold_patients=len(d['roles']['test']), sealed_cells=len(seal), total_cells=25))
            assert state_hash(m.state_dict()) == before
            result = dict(freeze=fh, indices=ix, score_nez=out, threshold=r['threshold'], checkpoint=r['checkpoint_sha256'])
            del m; gc.collect(); torch.cuda.empty_cache()
        coverage.append(dict(arm=arm, seed=seed, fold=fold, patients=len(d['roles']['test']), channels=len(ix),
            raw_consumed=arm != 'PHYSIOLOGY_ONLY', reused_scientific_cell=r['reused'], reused_patient_caches=reused,
            coverage_source_sha256=coverage_sha, **counts))
        if path.exists():
            q = read(path)
            for k, v in result.items():
                assert np.array_equal(q[k], v) if isinstance(v, np.ndarray) else q[k] == v
        else: save(path, result)
        seal[key] = sha(path)
        print('PCA_OUTER_CELL_SEALED', arm, seed, fold, len(seal), 25, flush=True)
    assert len(seal) == len(coverage) == 25
    for arm in ARMS:
        for seed in SEEDS:
            z = [r for r in coverage if r['arm'] == arm and r['seed'] == seed]
            assert len(z) == 5 and sum(r['patients'] for r in z) == 80 and sum(r['channels'] for r in z) == 7635
            assert sum(r['valid_windows'] for r in z) == (0 if arm == 'PHYSIOLOGY_ONLY' else 1471965)
    metadata = ROOT/'METADATA_PRIVATE.pkl'
    if not metadata.exists(): save(metadata, dict(meta=meta, archive_sha256=rec['binding']['archive']))
    else: assert read(metadata)['archive_sha256'] == rec['binding']['archive']
    freeze(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json', dict(freeze=fh, files=seal, all25_cells_sealed_before_metric_reporting=True))
    csv_out('RAW_COVERAGE', coverage)
    n = sum(r['reused'] for r in fr['cells'])
    write(PUB/'PREDICTION_STATUS.json', dict(status='ALL25_OUTER_CELLS_SEALED', freeze_sha256=fh,
        score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json'), new_prediction_cells=25-n, reused_cells=n,
        patients_each=80, channels_each=7635))
    print('PCA_ALL25_OUTER_SCORES_SEALED', flush=True)


if __name__ == '__main__': main()
