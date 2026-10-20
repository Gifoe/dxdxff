"""Outcome-free admission and strict per-cell historical reuse gates."""
import copy
import subprocess
import re
from common import *


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def check_reference(arm, fold, core, fac, records, banks, inputs, binding):
    r, state, v = reference(arm, 42, fold, optional=True)
    d, ss = bank(fold, banks, arm), scale(fold)
    m = make(core, fac, records, fold, arm, 42)
    assert state_hash(m.state_dict()) == v['binding']['initial'], 'INITIALIZATION_NOT_IDENTICAL'
    assert preprocessor(d, ss) == r['preprocessor'] == v['binding']['preprocessor'], 'PREPROCESSOR_NOT_IDENTICAL'
    assert len(v['history']) >= 6 and len(v['history']) <= 30
    winner = max(v['history'], key=lambda x: (x['macro_f1'], x['ez_f1'], -x['epoch']))
    assert winner['epoch'] == r['epoch'] == state['epoch'] and r['threshold'] == state['threshold']
    m.load_state_dict(state['model'], strict=True)
    before = state_hash(m.state_dict())
    out = predict(m, d, d['val'], inputs, ss, fold, binding)
    assert np.array_equal(out, v['logit']), 'VALIDATION_PREDICTION_NOT_IDENTICAL'
    assert state_hash(m.state_dict()) == before and np.array_equal(v['archive_indices'], d['archive_indices'][d['val']])
    selected = core.select_threshold(d['y'][d['val']], probability(out), d['patient'][d['val']])
    assert all(abs(float(z)-float(state[k])) < 1e-12 for k, z in selected.items())
    q, score_sha, _ = reference_score(r)
    test_ix = np.flatnonzero(np.isin(CONTEXT['rec']['meta']['patient'], list(d['roles']['test'])))
    assert np.array_equal(q['indices'], test_ix) and np.isfinite(q['score_nez']).all()
    del m, state, v
    gc.collect(); torch.cuda.empty_cache()
    return dict(arm=arm, seed=42, fold=fold, eligible=True, initial_state_exact=True,
                validation_logits_bit_exact=True, preprocessor_exact=True, threshold_and_epoch_exact=True,
                stored_prediction_hash_verified=True, checkpoint_sha256=r['checkpoint_sha256'],
                validation_sha256=r['validation_sha256'], prediction_sha256=score_sha,
                reason='ALL_STRICT_GATES_PASS')


def main():
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-v'],
                          cwd=HERE, capture_output=True, text=True)
    print(test.stdout, end=''); print(test.stderr, end='', file=sys.stderr)
    assert test.returncode == 0
    ntests = int(re.search(r'Ran (\d+) tests', test.stderr)[1])
    binding, core, fac, records, rec, banks = setup()
    inputs, roles, reuses, prep, raw_gate = store(), [], [], [], {}
    for fold in range(1, 6):
        d = bank(fold, banks, 'FULL')
        original = current.bank(fold, banks, 'FULL')
        assert np.array_equal(d['x'], original['x'])
        assert np.array_equal(query_features(rec['R14'][d['archive_indices']], d['patient'], d['pre'], 'FULL', core), d['x'])
        counts = [len(set(d['patient'][d['train']])), len(set(d['patient'][d['val']])), len(d['roles']['test'])]
        assert counts == [[51,51,50,52,51][fold-1], 13, [16,16,17,15,16][fold-1]]
        assert set(d['patient'][d['train']]) == d['roles']['fit']
        assert set(d['patient'][d['val']]).isdisjoint(d['roles']['fit'] | d['roles']['test'])
        roles.append(dict(fold=fold, fit_patients=counts[0], validation_patients=counts[1], test_patients=counts[2],
                          fit_channels=len(d['train']), validation_channels=len(d['val']), original_roles_exact=True))
        for arm in MODEL.DROP:
            ab = bank(fold, banks, arm); j = MODEL.DROP[arm]
            keep = [k for k in range(14) if k not in [2*j, 2*j+1]]
            assert np.array_equal(ab['x'][:, keep], d['x'][:, keep])
            assert np.array_equal(query_features(rec['R14'][ab['archive_indices']], ab['patient'], ab['pre'], arm, core), ab['x'])
        prep.append(dict(fold=fold, Full_input_bit_exact=True, all_six_masked_banks_legal=True,
                         unchanged_coordinates_bit_exact=True, all_original_rows_retained=True))
        for arm in current.ARMS:
            reuses.append(check_reference(arm, fold, core, fac, records, banks, inputs, binding))
        try:
            row = check_reference('RAW_ONLY', fold, core, fac, records, banks, inputs, binding)
        except AssertionError as error:
            # Optional reuse only: any strict mismatch becomes a fresh fit, never
            # a relaxed tolerance or a change to the inherited model/protocol.
            row = dict(arm='RAW_ONLY', seed=42, fold=fold, eligible=False, initial_state_exact=False,
                validation_logits_bit_exact=False, preprocessor_exact=False, threshold_and_epoch_exact=False,
                stored_prediction_hash_verified=False, checkpoint_sha256='', validation_sha256='', prediction_sha256='',
                reason=str(error) or 'STRICT_GATE_FAILED_FRESH_FIT_REQUIRED')
        raw_gate[str(fold)] = row['eligible']; reuses.append(row)
        print('CFA3_REUSE_FOLD_CHECKED', fold, 'optional_raw', row['eligible'], flush=True)
    assert len(reuses) == 50 and sum(r['eligible'] for r in reuses if r['arm'] != 'RAW_ONLY') == 45
    csv_out('ROLE_AUDIT', roles); csv_out('PREPROCESSING_ADMISSION', prep); csv_out('ADMISSION_REUSE', reuses)

    # All285 initializations, including all three independent training seeds.
    initchecks = []
    for seed in SEEDS:
        for fold in range(1, 6):
            base = make(core, fac, records, fold, 'FULL', seed)
            initial = {k: p.detach().cpu().clone() for k, p in base.named_parameters()}
            for arm in ARMS:
                m = make(core, fac, records, fold, arm, seed)
                for name, p in m.named_parameters():
                    key = name if not name.startswith('cnn.') or m.early else name.replace('cnn.', 'cnn.stem.', 1)
                    expected = initial[key]
                    if name == 'head.0.weight' and m.branch != 'joint':
                        expected = expected[:, :96] if m.branch == 'raw' else expected[:, 96:]
                    assert torch.equal(p.detach().cpu(), expected), (arm, seed, fold, name)
                assert sum(p.numel() for p in m.parameters()) == MODEL.PARAMETERS[arm]
                initchecks.append(dict(arm=arm, seed=seed, fold=fold, shared_tensors_exact=True,
                                       parameters=MODEL.PARAMETERS[arm], initial_sha256=state_hash(m.state_dict())))
                del m
            del base
    csv_out('INITIALIZATION_AUDIT', initchecks)

    fold = 2; d = bank(fold, banks, 'WO_BOTH_PR'); candidates = []
    for pid in sorted(d['roles']['fit']):
        plan = inputs.raw.registry['plans'][str(pid)]; assert sha(plan['path']) == plan['sha256']
        pack = torch.load(plan['path'], weights_only=False, map_location='cpu')
        candidates.append((int(pack['mask'][0].sum()), str(pid))); del pack
    _, pid = max(candidates); ix = d['train'][d['patient'][d['train']] == pid]; ss = scale(fold)
    for seed in SEEDS:
        source = source_for(core, fac, records, fold, seed)
        old = MODEL.BASE.Ablation(source, 'WO_BOTH_PR').cuda(); del source
        new = make(core, fac, records, fold, 'WO_BOTH_PR', seed)
        assert state_hash(old.state_dict()) == state_hash(new.state_dict())
        rng = core.rng_state(); outcomes = []
        for m, fn in [(old, legacy_parent.forward), (new, forward)]:
            core.restore_rng(rng); m.train(); opt = torch.optim.AdamW(m.parameters(), lr=.0003, weight_decay=.001)
            y = torch.as_tensor(d['y'][ix], device='cuda', dtype=torch.float32)
            pos = torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()), device='cuda')
            logits, _ = fn(m, d, ix, inputs, ss, fold, binding); loss = MODEL.objective(logits, y, pos); loss.backward()
            gradients = {n:p.grad.detach().clone() for n,p in m.named_parameters()}
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
            outcomes.append(dict(logits=logits.detach().clone(), loss=loss.detach().clone(), gradients=gradients,
                model=copy.deepcopy(m.state_dict()), optimizer=copy.deepcopy(opt.state_dict()), rng=core.rng_state()))
        assert equal_tree(outcomes[0], outcomes[1]), ('UNCHANGED_NO_PR_PARITY', seed)
        del old, new, outcomes, opt, gradients, logits, loss, y, pos
        gc.collect(); torch.cuda.empty_cache()
    audits = []
    for arm in ARMS:
        d = bank(fold, banks, arm); ix = d['train'][d['patient'][d['train']] == pid]
        m = make(core, fac, records, fold, arm, 42); opt = torch.optim.AdamW(m.parameters(), lr=.0003, weight_decay=.001)
        y = torch.as_tensor(d['y'][ix], device='cuda', dtype=torch.float32)
        pos = torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()), device='cuda')
        for step in range(2):
            opt.zero_grad(set_to_none=True); logit, info = forward(m, d, ix, inputs, ss, fold, binding)
            loss = MODEL.objective(logit, y, pos); assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            assert m.head[2].weight.grad.norm() > 0 and m.head[0].weight.grad.norm() > 0
            if m.branch != 'phys':
                assert dict(m.named_parameters())['cnn.stem.0.weight' if m.early else 'cnn.0.weight'].grad.norm() > 0
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
        audits.append(dict(arm=arm, seed=42, fold=fold, complete_patient_channels=len(ix), valid_windows=info['valid_windows'],
                           discarded_admission_steps=2, all_gradients_finite=True, retained_paths_effective=True))
        del m, opt, logit, loss, y, pos; gc.collect(); torch.cuda.empty_cache()
        print('CFA3_COMPLETE_PATIENT_PATH_PASS', arm, flush=True)
    nraw = sum(raw_gate.values())
    write(PUB/'ADMISSION.json', dict(status='PASS', source=sources(), unique_math_model_reporting_tests=ntests,
        mandatory_current42_reuses=45, raw_reuse_by_fold=raw_gate, optional_raw_reuses=nraw,
        expected_fresh_fits=240-nraw, expected_reused_cells=45+nraw, initialization_cells_checked=285,
        unchanged_noPR_forward_loss_gradients_Adam_rng_bit_exact_all3seeds=True,
        fresh_path_admission=audits, new_formal_fits=0, no_new_Outer=True, no_performance_eligibility_gate=True))
    print('CFA3_ADMISSION_PASS_NO_FORMAL_FITS', flush=True)


if __name__ == '__main__':
    main()
