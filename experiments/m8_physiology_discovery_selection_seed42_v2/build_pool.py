"""Unlabeled complete-window pool; legacy28 audit and old7 numerical parity."""
import gc
import pickle
import warnings

from common import *


def nanmean(a, axis):
    good = np.isfinite(a)
    return np.divide(np.nansum(a, axis=axis), good.sum(axis=axis), out=np.full(np.nansum(a, axis=axis).shape, np.nan), where=good.sum(axis=axis) > 0)


def finite_delta(values, mask, times):
    out = np.zeros_like(values)
    S, C, W, F = values.shape
    seizure = np.full((S, C, F), np.nan, np.float32)
    for s in range(S):
        for c in range(C):
            valid = mask[s, c]
            if not valid.any():
                continue
            pre = valid & (times[s] < 0)
            baseline = nanmean(values[s, c, pre], 0)
            out[s, c, valid] = values[s, c, valid] - baseline
            seizure[s, c] = nanmean(out[s, c, valid], 0)
    mu = nanmean(seizure, 0)
    sd = np.sqrt(nanmean((seizure - mu[None]) ** 2, 0))
    return out, np.stack([mu, sd], -1).reshape(C, 2 * F)


def graph_for(pid, ledger, rec, channels, mask, times, graph_files):
    result = np.full((*mask.shape, 7), np.nan, np.float32)
    own = next(q for q in rec['ledgers'] if q['patient'] == pid)
    for s, item in enumerate(own['records']):
        candidates = [(i, r) for i, r in enumerate(ledger) if (r['patient'], r['run'], r['sample']) == tuple(item['compound_key'][:3])]
        assert len(candidates) == 1, 'Nonunique graph/raw run identity'
        i, row = candidates[0]
        path = GRAPH / 'run_graph_private' / f'run_{i:03d}.npz'
        meta = jr(path.with_suffix('.json'))
        assert sha(path) == meta['output_sha256']
        with np.load(path) as data:
            v = data['metrics']; t = data['times']; valid = data['valid']
            assert np.array_equal(t, times[s, :len(t)])
            assert v.shape == (len(t), len(row['names']), 7)
            lookup = {norm(ch): j for j, ch in enumerate(channels)}
            ix = [lookup[norm(ch)] for ch in row['names']]
            assert np.array_equal(valid, mask[s, ix, :len(t)])
            result[s, np.asarray(ix)[:, None], np.arange(len(t))[None, :]] = v.transpose(1, 0, 2)
        graph_files.append(dict(sha256=sha(path), metadata_sha256=sha(path.with_suffix('.json'))))
    return result


def historical_audit(rec):
    path = Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl')
    assert sha(path) == rec['binding']['window']
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    sys.path.insert(0, str(SOURCE))
    from neuroez_c.dual_view_data import _record_key
    mapping = {_record_key(r): r for r in payload['run_records']}
    stat = {}; records = 0
    for led in rec['ledgers']:
        for item in led['records']:
            value = mapping[item['compound_key']]
            sample = value['sample']; names = list(map(str, sample['window_feature_names']))
            assert len(names) == len(set(names)) == 28
            x = np.asarray(sample['window_features']); mask = item['mask']
            assert x.shape[:2] == mask.shape and x.shape[2] == 28
            assert tuple(map(norm, value['channel_names_norm'])) == tuple(map(norm, item['names']))
            times = np.asarray(item['times']); records += 1
            for j, name in enumerate(names):
                a = x[..., j]; observed = a[mask]
                r = stat.setdefault(name, dict(feature=name, valid_windows=0, finite_windows=0, nonzero_windows=0, record_channel_units=0, temporally_varying_units=0))
                r['valid_windows'] += len(observed); r['finite_windows'] += int(np.isfinite(observed).sum()); r['nonzero_windows'] += int((observed != 0).sum())
                for c in range(a.shape[1]):
                    v = a[mask[:, c], c]
                    if len(v):
                        r['record_channel_units'] += 1
                        r['temporally_varying_units'] += int(np.nanmax(v) != np.nanmin(v))
    assert records == 256 and len(stat) == 28
    for name, r in stat.items():
        assert r['valid_windows'] == 1471965
        if name in ['log_bp_delta', 'log_bp_theta']:
            reason = 'CACHE_CONSTANT_ZERO_AND_LF_SOURCE_GATE_FAILED_NOT_BIOLOGICAL_INEFFECTIVENESS'
        elif name in [n.replace('rebuilt_', '') for n in GRAPH7]:
            reason = 'LEGACY_GRAPH_SOURCE_MISMATCH_USE_SEPARATE_VERIFIED_RAW_REBUILT_CANDIDATE'
        elif name in ['early_high_gamma_slope', 'early_line_length_slope', 'onset_latency_high_gamma', 'onset_latency_line_length', 'onset_rank_high_gamma', 'onset_rank_line_length', 'high_gamma_top20pct_mean', 'line_length_top20pct_mean']:
            reason = 'NONLOCAL_HISTORICAL_ONSET_BURST_DESCRIPTOR_SOURCE_AND_MASK_REPLAY_UNPROVEN_NOT_A_VALIDATED_WINDOW_DELTA_MEASUREMENT'
        else:
            reason = 'PINNED_FORMULA_RECOMPUTED_ON_MATCHED_VALID_RAW_WINDOWS_BAND_AND_NUMERICAL_LIMITS_RETAINED'
        r['eligibility_reason'] = reason
        r['physiology_invalid_claim'] = False
    pd.DataFrame(list(stat.values())).to_csv(PUB / 'HISTORICAL28_ELIGIBILITY.csv', index=False)
    del payload, mapping
    gc.collect()


def main():
    ROOT.mkdir(exist_ok=True); PUB.mkdir(exist_ok=True)
    rec = read(BASE / 'RECONSTRUCTION_PRIVATE.pkl')
    raw = Store(BASE)
    assert len(raw.registry['plans']) == 80 and rec['R14'].shape == (7635, 14)
    extractor = load('v2_pinned_raw_physiology', SOURCE / 'raw_delta_encoder_seed42_v1/code/raw_physiological_features.py').Extractor(SOURCE, '18ec360e9d9ec7e5b56b05208c0ef6575c13769ab6866ef241e4d65e8a554c72')
    candidate = load('v2_original_candidates', Path(r'D:\m8_physiology_feature_discovery_seed42_v1\features.py'))
    oldreg = read(Path(r'D:\dgm_net_seed42_v1_runtime\CONDITIONING_REGISTRY_PRIVATE.pkl'))
    newreg = read(PRIOR / 'CANDIDATE_REGISTRY_PRIVATE.pkl')
    numreg = read(PRIOR / 'NUMERICAL_CONTROL_REGISTRY_PRIVATE.pkl')
    graph_gate = jr(GRAPH / 'public/GRAPH_ALIGNMENT_AUDIT.json')
    assert graph_gate['status'] == 'PASS' and graph_gate['binding']['raw'] == rec['binding']['raw']
    with (GRAPH / 'GRAPH_LEDGER_PRIVATE.pkl').open('rb') as stream:
        graph_ledger = pickle.load(stream)
    assert len(graph_ledger) == 256
    binding = dict(source={n:sha(HERE/n) for n in ['PROTOCOL_LOCK.json', 'common.py', 'build_pool.py']}, reconstruction=sha(BASE / 'RECONSTRUCTION_PRIVATE.pkl'), raw_registry=sha(BASE / 'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),
        original7_registry=sha(Path(r'D:\dgm_net_seed42_v1_runtime\CONDITIONING_REGISTRY_PRIVATE.pkl')), candidate_registry=sha(PRIOR / 'CANDIDATE_REGISTRY_PRIVATE.pkl'), numeric_registry=sha(PRIOR / 'NUMERICAL_CONTROL_REGISTRY_PRIVATE.pkl'),
        graph_ledger=sha(GRAPH / 'GRAPH_LEDGER_PRIVATE.pkl'), graph_gate=sha(GRAPH / 'public/GRAPH_ALIGNMENT_AUDIT.json'),
        ez_features=sha(SOURCE / 'ez_features.py'), candidate_source=sha(Path(candidate.__file__)), names=NAMES, labels_used=False)
    freeze(ROOT / 'POOL_INPUT_FREEZE_PRIVATE.json', binding)
    if not (PUB / 'HISTORICAL28_ELIGIBILITY.csv').exists():
        historical_audit(rec)
    summary = np.full((7635, 64), np.nan, np.float32); registry = {}; counts = []; graph_files = []
    for number, pid in enumerate(sorted(raw.registry['plans']), 1):
        plan = raw.registry['plans'][pid]
        target = ROOT / 'pool_private' / f'{number:03d}.pkl'
        if target.exists():
            q = read(target); assert q['binding'] == binding and q['raw_bank_sha256'] == plan['sha256']
        else:
            old = read(oldreg['files'][pid]['path']); prior = read(newreg['files'][pid]['path']); numeric = read(numreg['files'][pid]['path'])
            for registry_source, value in [(oldreg, old), (newreg, prior), (numreg, numeric)]:
                assert sha(registry_source['files'][pid]['path']) == registry_source['files'][pid]['sha256']
                assert value['raw_bank_sha256'] == plan['sha256'] and value['channels'] == plan['channels']
            wave, mask, times, _ = raw.get(pid, plan['channels'], 'cpu')
            wave, mask, times = wave.numpy(), mask.numpy(), times.numpy()
            values = np.full((*mask.shape, 25), np.nan, np.float32)
            ix = np.flatnonzero(mask.ravel()); flat = wave.reshape(-1, 500); out = values.reshape(-1, 25)
            for start in range(0, len(ix), 2048):
                take = ix[start:start + 2048]
                base = extractor.all(flat[take]); new, _ = candidate.extract(flat[take])
                log = np.log(np.maximum(np.expm1(base[:, extractor.indices[:3]].astype(float)), 1e-30)).astype(np.float32)
                out[take] = np.concatenate([base[:, extractor.indices], new, log, base[:, [2, 6, 11, 12]]], -1)
            d, su = candidate.delta_summary(values, mask, times)
            # Original source/runtime and arithmetic, not a tolerance-adjusted replacement.
            assert np.array_equal(d[..., :7], old['delta'])
            assert np.array_equal(su[:, :14], old['seven14'])
            assert np.array_equal(d[..., 7:18], prior['delta'])
            assert np.array_equal(d[..., 18:21], numeric['delta'])
            graphs = graph_for(pid, graph_ledger, rec, plan['channels'], mask, times, graph_files)
            gd, gs = finite_delta(graphs, mask, times)
            allvalues = np.concatenate([values, graphs], -1)
            delta = np.concatenate([d, gd], -1); su = np.concatenate([su, gs], -1)
            phase = np.full((len(mask), mask.shape[1], 3, 32), np.nan, np.float32)
            for s in range(len(mask)):
                for k, select in enumerate([times[s] < 0, (times[s] >= 0) & (times[s] < 8), (times[s] >= 8) & (times[s] < 20)]):
                    observed = allvalues[s].copy(); observed[~(mask[s] & select[None])] = np.nan
                    phase[s, :, k] = nanmean(observed, 1)
            q = dict(binding=binding, raw_bank_sha256=plan['sha256'], channels=plan['channels'], archive_indices=old['archive_indices'],
                delta=delta, summary=su, phase=phase, mask=mask, times=times, values=allvalues,
                windows=int(mask.sum()), seizures=len(mask), finite_counts=np.isfinite(allvalues[mask]).sum(0),
                old7_parity_exact=True, previous14_new_delta_parity_exact=True)
            save(target, q)
            del wave, mask, times, values, graphs, delta, flat, out, allvalues, phase
        summary[q['archive_indices']] = q['summary']
        registry[pid] = dict(path=str(target), sha256=sha(target), archive_indices=q['archive_indices'])
        counts.append(dict(patients=1, channels=len(q['archive_indices']), windows=q['windows'], seizures=q['seizures'], finite=q['finite_counts']))
        write(PUB / 'POOL_STATUS.json', dict(status='BUILDING_UNLABELED_POOL', patients=len(registry), total_patients=80))
        print('UNLABELED32_POOL_PATIENT_COMPLETE', number, 'of80', flush=True)
        raw.cache.clear(); gc.collect()
    assert sum(r['windows'] for r in counts) == 1471965 and sum(r['seizures'] for r in counts) == 256
    assert np.array_equal(summary[:, :14], rec['R14'])
    save(ROOT / 'POOL_PRIVATE.pkl', dict(binding=binding, summary=summary, names=NAMES, meta=rec['meta']))
    save(ROOT / 'POOL_REGISTRY_PRIVATE.pkl', dict(binding=binding, files=registry))
    finite = np.stack([r['finite'] for r in counts]).sum(0)
    quality = [dict(feature=n, group=j, valid_windows=1471965, finite_fraction=float(finite[j]/1471965),
                    potentially_eligible=bool(finite[j]/1471965 >= .99), source_family='original7' if j<7 else 'existing_candidate11' if j<18 else 'numeric_sameband3' if j<21 else 'legacy_spectral_recomputed4' if j<25 else 'verified_rebuilt_graph7',
                    labels_used=False, input_source_operationally_verified=True, independent_acquisition_onset_not_proven=True) for j,n in enumerate(NAMES)]
    pd.DataFrame(quality).to_csv(PUB / 'POOL_QUALITY.csv', index=False)
    write(PUB / 'POOL_STATUS.json', dict(status='COMPLETE_UNLABELED32_POOL', patients=80, channels=7635, seizures=256, valid_windows=1471965,
        original7_summary_and_delta_exact=True, previous14_candidate_delta_exact=True, names=NAMES, labels_used=False,
        pool_sha256=sha(ROOT / 'POOL_PRIVATE.pkl'), registry_sha256=sha(ROOT / 'POOL_REGISTRY_PRIVATE.pkl'), input_freeze_sha256=sha(ROOT / 'POOL_INPUT_FREEZE_PRIVATE.json')))
    print('UNLABELED32_POOL_COMPLETE_NO_LABELS', flush=True)


if __name__ == '__main__':
    main()
