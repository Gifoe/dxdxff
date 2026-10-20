"""Forty new frozen cells and five exact historical references, sealed before metrics."""
from common import *

COUNTS = ['valid_windows', 'relative_active_coordinates', 'relative_used_coordinates',
          'global_fallback_coordinates', 'singleton_valid_windows']


def reference_coverage(fold):
    path = PREVROOT/'public/RAW_COVERAGE.csv'
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    r = next(r for r in rows if r['arm'] == 'FULL_F0' and int(r['fold']) == fold)
    return dict.fromkeys(COUNTS, 0) | dict(valid_windows=int(r['valid_windows'])), sha(path)


def main():
    binding, core, fac, records, rec, banks = setup()
    fr = frozen()
    assert fr['input'] == binding
    fh = sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    inputs, seal, coverage, meta = store(), {}, [], rec['meta']
    oldseal = jr(PREVROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    for r in fr['cells']:
        arm, fold = r['arm'], r['fold']
        d, ss = bank(fold, banks, arm), scale(fold)
        ix = np.flatnonzero(np.isin(meta['patient'], list(d['roles']['test'])))
        assert len(set(meta['patient'][ix])) == [16, 16, 17, 15, 16][fold-1]
        assert preprocessor(d, ss) == r['preprocessor']
        path = ROOT/'scores_private'/f'{arm}_{fold}.pkl'
        reused, coverage_sha = 0, ''
        if r['reused']:
            key = f'FULL_F0_{fold}'
            src = PREVROOT/'scores_private'/(key+'.pkl')
            assert sha(src) == oldseal['files'][key]
            q = read(src)
            assert np.array_equal(q['indices'], ix) and q['checkpoint'] == r['checkpoint_sha256'] and q['threshold'] == r['threshold']
            result = dict(freeze=fh, indices=ix, score_nez=q['score_nez'], threshold=r['threshold'],
                          checkpoint=r['checkpoint_sha256'], reused_source_sha256=sha(src))
            counts, coverage_sha = reference_coverage(fold)
        else:
            pp = d['pre']
            if arm in ['WO_FEATURE_PR', 'WO_BOTH_PR']:
                z = rec['R14'][ix]
                x = ((np.where(np.isnan(z), pp['imputer_statistics'], z)-pp['mean'])/pp['scale']).astype(np.float32)
            else:
                x = query_feature_pr(rec['R14'][ix], meta['patient'][ix], pp)
            assert x.shape == (len(ix), 14) and np.isfinite(x).all()
            td = dict(x=x, patient=meta['patient'][ix], channel=meta['channel'][ix])
            m = make(core, fac, records, fold, arm)
            m.load_state_dict(torch.load(r['checkpoint'], weights_only=False, map_location='cpu')['model'])
            m.eval()
            m.collect_guidance_audit = True
            before = state_hash(m.state_dict())
            out, counts = np.empty(len(ix), np.float32), dict.fromkeys(COUNTS, 0)
            for n, pid in enumerate(sorted(d['roles']['test']), 1):
                jj = np.flatnonzero(td['patient'] == pid)
                ii = ix[jj]
                cb = dict(freeze=fh, checkpoint=r['checkpoint_sha256'], indices=legacy.array_sha(ii),
                          input=legacy.array_sha(x[jj]), scale=legacy.array_sha(ss.cpu().numpy()))
                cache = ROOT/'patient_scores_private'/arm/f'fold{fold}'/(hashlib.sha256(str(pid).encode()).hexdigest()+'.pkl')
                if cache.exists():
                    q = read(cache)
                    assert q['binding'] == cb
                    reused += 1
                else:
                    with torch.inference_mode():
                        logit, info = forward(m, td, jj, inputs, ss, fold, binding)
                        p = probability(logit.cpu().numpy())
                    q = dict(binding=cb, score_nez=p, info=info)
                    save(cache, q)
                assert q['score_nez'].shape == (len(jj),) and np.isfinite(q['score_nez']).all()
                out[jj] = q['score_nez']
                for k in counts:
                    counts[k] += q['info'].get(k, 0)
                write(PUB/'PROGRESS.json', dict(status='FROZEN_OUTER_INFERENCE', arm=arm, fold=fold,
                      patient_number=n, fold_patients=len(d['roles']['test']), sealed_cells=len(seal), total_cells=45))
            assert state_hash(m.state_dict()) == before
            result = dict(freeze=fh, indices=ix, score_nez=out, threshold=r['threshold'], checkpoint=r['checkpoint_sha256'])
            del m
            gc.collect()
            torch.cuda.empty_cache()
        coverage.append(dict(arm=arm, fold=fold, patients=len(d['roles']['test']), channels=len(ix), raw_consumed=True,
                        reused_scientific_cell=r['reused'], reused_patient_caches=reused,
                        coverage_source_sha256=coverage_sha, **counts))
        if path.exists():
            q = read(path)
            for k, v in result.items():
                assert np.array_equal(q[k], v) if isinstance(v, np.ndarray) else q[k] == v
        else:
            save(path, result)
        seal[f'{arm}_{fold}'] = sha(path)
        print('RGF0_OUTER_CELL_SEALED', arm, fold, len(seal), 45, flush=True)
    assert len(seal) == len(coverage) == 45
    for arm in ARMS:
        z = [r for r in coverage if r['arm'] == arm]
        assert len(z) == 5 and sum(r['patients'] for r in z) == 80 and sum(r['channels'] for r in z) == 7635
        assert sum(r['valid_windows'] for r in z) == 1471965
    metadata = ROOT/'METADATA_PRIVATE.pkl'
    if not metadata.exists():
        save(metadata, dict(meta=meta, archive_sha256=rec['binding']['archive']))
    else:
        assert read(metadata)['archive_sha256'] == rec['binding']['archive']
    freeze(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json', dict(freeze=fh, files=seal, all45_cells_sealed_before_metric_reporting=True))
    csv_out('RAW_COVERAGE', coverage)
    write(PUB/'PREDICTION_STATUS.json', dict(status='ALL45_OUTER_CELLS_SEALED', freeze_sha256=fh,
          score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json'), new_prediction_cells=40,
          reused_cells=5, patients_each=80, channels_each=7635))
    print('RGF0_ALL45_OUTER_SCORES_SEALED', flush=True)


if __name__ == '__main__':
    main()
