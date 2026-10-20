"""Original-protocol identities and unique changed executable-path admission."""
import copy
import subprocess
from common import *


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-v'], cwd=HERE)
    assert result.returncode == 0
    binding, core, fac, records, rec, banks = setup()
    inputs = store()
    roles, reuses, preprocessing = [], [], []
    for fold in range(1, 6):
        old = parent.bank(fold, banks, 'FULL_F0')
        no_pr = bank(fold, banks, 'WO_BOTH_PR')
        revised = bank(fold, banks, 'FULL')
        protocol = jr(HERE/'PROTOCOL_LOCK.json')
        counts = [len(set(old['patient'][old['train']])), len(set(old['patient'][old['val']])), len(old['roles']['test'])]
        assert counts == [protocol['roles'][name][fold-1] for name in ['train_patients', 'validation_patients', 'test_patients']]
        assert set(old['patient'][old['train']]) == old['roles']['fit']
        assert set(old['patient'][old['val']]).isdisjoint(old['roles']['fit'] | old['roles']['test'])
        roles.append(dict(fold=fold, fit_patients=counts[0], validation_patients=counts[1], test_patients=counts[2],
                          fit_channels=len(old['train']), validation_channels=len(old['val']), original_roles_exact=True))
        raw = rec['R14'][old['archive_indices']]
        assert np.array_equal(no_pr['x'], parent.bank(fold, banks, 'WO_BOTH_PR')['x'])
        assert np.array_equal(query_feature_pr(raw, old['patient'], revised['pre']), revised['x'])
        for phase, ix in [('FIT', old['train']), ('VALIDATION', old['val'])]:
            active = [j for j in range(14) if j not in [8, 9]]
            preprocessing.append(dict(fold=fold, phase=phase, patients=len(set(old['patient'][ix])), channels=len(ix),
                corrected_active_abs_max=float(np.abs(revised['x'][ix][:, active]).max()),
                legacy_active_abs_max=float(np.abs(old['x'][ix][:, active]).max()),
                NoFeaturePR_matrix_bit_unchanged=True, all_original_rows_retained=True,
                unsupported_FIT_unit_coordinates=int((revised['pre']['FIT_unit_variance'] == 0).sum())))
        row, state, validation = reference('B0_ORIGINAL', fold)
        model = make(core, fac, records, fold, 'B0_ORIGINAL')
        assert state_hash(model.state_dict()) == validation['binding']['initial']
        assert preprocessor(old, scale(fold)) == row['preprocessor']
        model.load_state_dict(state['model'])
        before = state_hash(model.state_dict())
        logits = predict(model, old, old['val'], inputs, scale(fold), fold, binding)
        assert np.array_equal(logits, validation['logit']) and state_hash(model.state_dict()) == before
        assert np.array_equal(validation['archive_indices'], old['archive_indices'][old['val']])
        reuses.append(dict(arm='B0_ORIGINAL', fold=fold, initial_state_exact=True, validation_logits_bit_exact=True,
            parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=row['checkpoint_sha256'],
            validation_sha256=row['validation_sha256'], selected_epoch=row['epoch'], threshold=row['threshold']))
        del model, state, validation
        gc.collect()
        torch.cuda.empty_cache()
        print('RGF0_ORIGINAL_REFERENCE_PASS', fold, flush=True)
    csv_out('ROLE_AUDIT', roles)
    csv_out('PREPROCESSING_ADMISSION', preprocessing)
    csv_out('ADMISSION_REUSE', reuses)

    # Largest genuine waveform support within one original legal FIT, no labels
    # used to choose the admission workload and no rows removed from any fit.
    fold = 2
    d = bank(fold, banks, 'WO_BOTH_PR')
    candidates = []
    for pid in sorted(d['roles']['fit']):
        plan = inputs.raw.registry['plans'][str(pid)]
        assert sha(plan['path']) == plan['sha256']
        pack = torch.load(plan['path'], weights_only=False, map_location='cpu')
        candidates.append((int(pack['mask'][0].sum()), str(pid)))
        del pack
    _, pid = max(candidates)
    ix = d['train'][d['patient'][d['train']] == pid]
    ss = scale(fold)
    raw, mask, times, delta, _ = inputs.get(pid, d['channel'][ix].astype(str).tolist(), 'cuda')
    old_model = parent.make(core, fac, records, fold, 'WO_BOTH_PR')
    assert torch.equal(old_model.raw_feature_function.__globals__['transform'](delta, mask, times, 'full', None), delta)
    del raw, mask, times, delta
    revised_model = make(core, fac, records, fold, 'WO_BOTH_PR')
    assert state_hash(old_model.state_dict()) == state_hash(revised_model.state_dict())
    rng = core.rng_state()
    outcomes = []
    for model, fn in [(old_model, parent.forward), (revised_model, forward)]:
        core.restore_rng(rng)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
        y = torch.as_tensor(d['y'][ix], device='cuda', dtype=torch.float32)
        pos = torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()), device='cuda')
        logits, _ = fn(model, d, ix, inputs, ss, fold, binding)
        loss = MODEL.objective(logits, y, pos)
        loss.backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        outcomes.append(dict(logits=logits.detach().clone(), loss=loss.detach().clone(), gradients=gradients,
            model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()), rng=core.rng_state()))
    assert equal_tree(outcomes[0], outcomes[1]), 'Unchanged noPR path differs from the original procedure'
    del old_model, revised_model, outcomes, optimizer, logits, loss, gradients, y, pos
    gc.collect()
    torch.cuda.empty_cache()

    audits = []
    for arm in NEW:
        d = bank(fold, banks, arm)
        ix = d['train'][d['patient'][d['train']] == pid]
        model = make(core, fac, records, fold, arm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
        y = torch.as_tensor(d['y'][ix], device='cuda', dtype=torch.float32)
        pos = torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()), device='cuda')
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits, info = forward(model, d, ix, inputs, ss, fold, binding)
            loss = MODEL.objective(logits, y, pos)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            norms = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
            assert norms['head.0.weight'] > 0 and norms['head.2.weight'] > 0
            assert norms['cnn.stem.0.weight' if model.early else 'cnn.0.weight'] > 0
            if model.early:
                assert norms['cnn.conditioner.2.weight'] > 0
                if step:
                    assert norms['cnn.conditioner.0.weight'] > 0
            if model.late:
                assert norms['gate.network.2.weight'] > 0
                if step:
                    assert norms['gate.network.0.weight'] > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        audits.append(dict(arm=arm, fold=fold, complete_patient_channels=len(ix), valid_windows=info['valid_windows'],
            parameters=sum(p.numel() for p in model.parameters()), discarded_admission_steps=2,
            all_gradients_finite=True, retained_paths_have_effective_gradients=True))
        del model, optimizer, logits, loss, y, pos
        gc.collect()
        torch.cuda.empty_cache()
        print('RGF0_FULL_PATIENT_BACKWARD_PASS', arm, flush=True)
    write(PUB/'ADMISSION.json', dict(status='PASS', source=sources(), unique_math_model_reporting_tests=9,
        exact_original_reference_replays=5, unchanged_noPR_logits_loss_gradients_oneAdam_rng_bit_exact=True,
        discarded_parity_updates=2, fresh_path_admission=audits, new_formal_fits=0,
        original_five_roles_exact=True, no_Development_eligibility_gate=True, no_new_Outer=True))
    print('RGF0_ADMISSION_PASS_NO_FORMAL_FITS', flush=True)


if __name__ == '__main__':
    main()
