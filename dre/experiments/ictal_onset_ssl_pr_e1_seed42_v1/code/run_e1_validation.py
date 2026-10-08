"""E1-only controlled training; frozen E3 export/helpers, no SSL or outer test."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


METRICS = ('macro_f1', 'ez_f1', 'ez_ap', 'ez_auroc', 'mrr', 'top1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def train_e1(fold, loader, fit_ids, val_ids, output, epochs, seed, binding, notify):
    from model import IctalLocalization, patient_bce
    from run_experiment import set_seed, score_patient, pool, fit_threshold, sha_file, metric_one
    from run_e3_validation import atomic_torch, load_resume, restore_rng, rng_state, cpu_state
    set_seed(seed)
    model = IctalLocalization(use_pr=True, use_attention=False).to(loader.device)
    # Deliberately NO pretrained-state load: this is the supplied E1 switch.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    path = output / f'fold_{fold}_E1_last_private.pt'
    saved = load_resume(path, binding, loader.device)
    history, best, first = [], None, 1
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        history, best, first = saved['history'], saved['best'], saved['epoch'] + 1
    for epoch in range(first, epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for pid in np.random.permutation(fit_ids):
            pair, present, labels = loader.get(pid, with_label=True)
            optimizer.zero_grad(set_to_none=True)
            logits, mask = model(pair, present)
            loss = patient_bce(logits, labels, mask)
            if not torch.isfinite(loss):
                raise RuntimeError('NONFINITE_SUPERVISED_LOSS')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError('NO_FIT_PATIENTS')
        validation = [score_patient(model, loader, pid) for pid in val_ids]
        metrics = pool(validation, .5)
        primary = metrics['ez_ap'] if np.isfinite(metrics['ez_ap']) else -1.
        key = (primary, metrics['macro_f1'], -epoch)
        history.append({'fold': fold, 'variant': 'E1', 'epoch': epoch,
                        'fit_bce': float(np.mean(losses)), 'val_ap': metrics['ez_ap'],
                        'val_f1_0p5': metrics['macro_f1'], 'seconds': time.perf_counter() - started})
        if best is None or key > best[0]:
            best = (key, cpu_state(model), epoch)
        atomic_torch(path, {'binding': binding, 'epoch': epoch, 'model': model.state_dict(),
                           'optimizer': optimizer.state_dict(), 'rng': rng_state(),
                           'history': history, 'best': best})
        notify(fold, epoch, epochs, history[-1])
    model.load_state_dict(best[1])
    model.eval()
    validation = [score_patient(model, loader, pid) for pid in val_ids]
    threshold = fit_threshold(validation)
    metrics = pool(validation, threshold)
    final = output / f'fold_{fold}_E1.pt'
    checkpoint = {'state_dict': model.state_dict(), 'variant': 'E1', 'fold': fold,
                  'selected_epoch': best[2], 'ez_threshold': threshold,
                  'selection': 'validation mean patient EZ-AP, tie F1@0.5',
                  'val_metrics': metrics, 'binding': binding}
    if not final.exists():
        atomic_torch(final, checkpoint)
    else:
        existing = torch.load(final, map_location=loader.device, weights_only=True)
        if (existing['binding'] != binding or existing['selected_epoch'] != best[2]
                or existing['ez_threshold'] != threshold or existing['val_metrics'] != metrics):
            raise RuntimeError('FINAL_E1_SELECTION_CHANGED')
        if any(not torch.equal(v, existing['state_dict'][k]) for k, v in model.state_dict().items()):
            raise RuntimeError('FINAL_E1_WEIGHTS_CHANGED')
    selection = {'fold': fold, 'variant': 'E1', 'checkpoint': str(final),
                 'checkpoint_sha256': sha_file(final), 'selected_epoch': best[2],
                 'threshold': threshold, 'val_metrics': metrics, 'ssl_checkpoint_loaded': False}
    private_rows = [{'patient': r['id'], 'fold': fold, 'center': r['center'],
                     **metric_one(r['labels_ez'], r['scores_ez'], threshold)} for r in validation]
    return selection, history, private_rows


def summarize(rows, selections, e3_output, output):
    from run_experiment import write_csv, gated_attention_allowed
    from run_e3_validation import atomic_json
    write_csv(output / 'validation_by_fold.csv', [{'fold': s['fold'], 'variant': 'E1',
              'selected_epoch': s['selected_epoch'], 'threshold': s['threshold'],
              'n_patients': 13, **s['val_metrics']} for s in selections])
    centers = []
    for center in sorted({r['center'] for r in rows}):
        group = [r for r in rows if r['center'] == center]
        centers.append({'center': center, 'n_patient_fold_cells': len(group),
                        **{m: float(np.mean([r[m] for r in group])) for m in METRICS}})
    write_csv(output / 'validation_by_center.csv', centers)
    ids = sorted({r['patient'] for r in rows})
    values = np.array([[np.mean([r[m] for r in rows if r['patient'] == pid])
                        for m in METRICS] for pid in ids])
    draws = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    bootstrap = []
    for index, metric in enumerate(METRICS):
        distribution = values[draws, index].mean(1)
        lower, upper = np.quantile(distribution, [.025, .975])
        bootstrap.append({'metric': metric, 'mean_unique_validation_patient': float(values[:, index].mean()),
                          'bootstrap_95_lower': float(lower), 'bootstrap_95_upper': float(upper),
                          'n_unique_patients': len(ids), 'draws': 10000, 'seed': 42,
                          'interpretation': 'descriptive post-selection validation; not outer-test CI'})
    write_csv(output / 'validation_patient_bootstrap.csv', bootstrap)
    summary = {'variant': 'E1', 'n_patient_fold_cells': len(rows), 'n_unique_validation_patients': len(ids),
               'fold_mean': {m: float(np.mean([s['val_metrics'][m] for s in selections])) for m in METRICS},
               'unique_patient_mean': dict(zip(METRICS, values.mean(0).tolist())),
               'interpretation': 'Post-selection source-only validation; no outer test'}
    atomic_json(output / 'validation_summary.json', summary)
    with (e3_output / 'validation_per_patient_private.csv').open(encoding='utf-8', newline='') as handle:
        e3_rows = list(csv.DictReader(handle))
    e1 = {(r['patient'], int(r['fold'])): r for r in rows}
    e3 = {(r['patient'], int(r['fold'])): r for r in e3_rows}
    if len(e1) != 65 or set(e1) != set(e3):
        raise RuntimeError('E1_E3_VALIDATION_MEMBERSHIP_MISMATCH')
    for key in e1:
        if any(int(e1[key][m]) != int(e3[key][m]) for m in ('n_channels', 'ez_count')):
            raise RuntimeError('E1_E3_CLASSIFICATION_UNIT_MISMATCH')
    counts = np.array([sum(key[0] == pid for key in e1) for pid in ids])
    paired = []
    for metric in METRICS:
        sums = np.array([sum(float(e3[key][metric]) - e1[key][metric] for key in e1 if key[0] == pid)
                         for pid in ids])
        distribution = sums[draws].sum(1) / counts[draws].sum(1)
        lower, upper = np.quantile(distribution, [.025, .975])
        paired.append({'contrast': 'E3_minus_E1', 'metric': metric,
                       'delta_patient_fold_cell_mean': float(sums.sum() / counts.sum()),
                       'cluster_bootstrap_95_lower': float(lower), 'cluster_bootstrap_95_upper': float(upper),
                       'n_patient_fold_cells': 65, 'n_unique_patients': len(ids), 'draws': 10000, 'seed': 42,
                       'interpretation': 'paired post-selection validation; not independent SSL confirmation'})
    write_csv(output / 'paired_bootstrap_E3_minus_E1.csv', paired)
    e3_selections = read(e3_output / 'validation_selections.json')
    gate = gated_attention_allowed(selections + e3_selections)
    gate.update({'status': 'DEVELOPMENT_GATE_ONLY', 'E4_E5_run': False,
                 'optional_controls_authorized': False, 'outer_test_accessed': False})
    atomic_json(output / 'validation_gate.json', gate)
    comparison = []
    for variant, selected in [('E1', selections), ('E3', e3_selections)]:
        comparison.append({'variant': variant, 'n_patient_fold_cells': 65, 'n_unique_patients': len(ids),
                           **{m: float(np.mean([s['val_metrics'][m] for s in selected])) for m in METRICS}})
    write_csv(output / 'E1_E3_COMPARISON.csv', comparison)
    return summary, comparison, paired, gate


def main():
    parser = argparse.ArgumentParser()
    for name in ('package', 'export', 'output', 'protocol', 'e3-output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    sys.path.insert(0, args.package)
    from run_experiment import sha_file, write_csv
    from run_e3_validation import CachedPatientFiles, atomic_json
    import run_e3_validation
    protocol = read(args.protocol)
    source, output, e3_output = Path(args.export), Path(args.output), Path(args.e3_output)
    if (protocol['variant'], protocol['supervised_epochs'], protocol['ssl_epochs']) != ('E1', 30, 0):
        raise RuntimeError('E1_PROTOCOL_SCOPE_CHANGED')
    if protocol['outer_test_authorized'] or protocol['optional_E4_E5_authorized']:
        raise RuntimeError('UNAUTHORIZED_E1_SCOPE')
    if np.__version__ != protocol['numpy_runtime'] or torch.__version__ != protocol['torch_runtime']:
        raise RuntimeError('UNEXPECTED_NUMPY_OR_TORCH_RUNTIME')
    paths = {'export_audit_sha256': source / 'audit.json', 'export_manifest_sha256': source / 'manifest.json',
             'model_sha256': Path(args.package) / 'model.py',
             'original_runner_sha256': Path(args.package) / 'run_experiment.py',
             'e3_common_helper_sha256': Path(run_e3_validation.__file__)}
    if any(sha_file(path) != protocol[key] for key, path in paths.items()):
        raise RuntimeError('FROZEN_E3_DATA_OR_SOURCE_CHANGED')
    e3_status = read(e3_output / 'RUN_STATUS.json')
    if e3_status['status'] != 'VALIDATION_COMPLETE_TEST_NOT_ACCESSED' or e3_status['outer_test_accessed']:
        raise RuntimeError('E3_REFERENCE_NOT_FROZEN_VALIDATION')
    audit, manifest = read(source / 'audit.json'), read(source / 'manifest.json')
    if (audit['n_patients'], audit['n_seizures'], audit['n_unique_channels'], audit['authorized_excluded_records']) != (80, 255, 7635, 1):
        raise RuntimeError('E1_AMENDED_DATA_GATE_FAILED')
    reference = read(e3_output / 'validation_selections.json')
    for selected in reference:
        if sha_file(selected['checkpoint']) != selected['checkpoint_sha256']:
            raise RuntimeError('FROZEN_E3_CHECKPOINT_CHANGED')
    binding = {'protocol': sha_file(args.protocol), 'runner': sha_file(__file__),
               **{key: sha_file(path) for key, path in paths.items()},
               'e3_selection_lock': sha_file(e3_output / 'score_selection_lock.json'),
               'e3_private_validation_ledger': sha_file(e3_output / 'validation_per_patient_private.csv'),
               'variant': 'E1', 'seed': 42, 'ssl_epochs': 0, 'supervised_epochs': 30}
    output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    guard = (output / 'runner.lock').open('a+b')
    guard.seek(0)
    if guard.read(1) == b'':
        guard.write(b'0')
        guard.flush()
    guard.seek(0)
    msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
    lock = output / 'TRAINING_BINDING.json'
    if lock.exists() and read(lock) != binding:
        raise RuntimeError('E1_RESUME_BINDING_CHANGED')
    if not lock.exists():
        atomic_json(lock, binding)
    torch.set_num_threads(2)
    def notify(fold, epoch, epochs, row):
        state = {'status': 'RUNNING', 'variant': 'E1', 'pid': os.getpid(), 'fold': fold,
                 'epoch': epoch, 'supervised_epochs': epochs, 'ssl_epochs': 0,
                 'completed_folds': fold - 1, 'epoch_seconds': row['seconds'], 'outer_test_accessed': False}
        atomic_json(output / 'RUN_STATUS.json', state)
        print(json.dumps({**state, 'diagnostics': row}), flush=True)
    selections, all_history, rows = [], [], []
    for entry in sorted(manifest['folds'], key=lambda f: f['fold_idx']):
        fold = entry['fold_idx']
        fit, val, test = map(set, (entry['fit_subjects'], entry['validation_subjects'], entry['test_subjects']))
        if fit & val or fit & test or val & test or len(val) != 13:
            raise RuntimeError('FOLD_ROLE_GATE_FAILED')
        print(json.dumps({'stage': 'LOAD_FIT_VAL_ONLY', 'fold': fold}), flush=True)
        loader = CachedPatientFiles(source, manifest, args.device, entry['fit_subjects'] + entry['validation_subjects'])
        selection, history, private_rows = train_e1(fold, loader, entry['fit_subjects'],
            entry['validation_subjects'], output, 30, 42 + fold * 100, binding, notify)
        selections.append(selection)
        all_history.extend(history)
        rows.extend(private_rows)
        atomic_json(output / f'fold_{fold}_selection.json', selection)
        write_csv(output / 'supervised_history.csv', all_history)
        print(json.dumps({'stage': 'FOLD_COMPLETE', 'fold': fold, 'metrics': selection['val_metrics']}), flush=True)
        del loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(all_history) != 150:
        raise RuntimeError('E1_EPOCH_COVERAGE_FAILED')
    atomic_json(output / 'validation_selections.json', selections)
    atomic_json(output / 'score_selection_lock.json', {'binding': binding, 'selections': selections,
                                                       'outer_test_accessed': False})
    write_csv(output / 'validation_per_patient_private.csv', rows)
    summary, comparison, paired, gate = summarize(rows, selections, e3_output, output)
    table = '\n'.join('| ' + r['variant'] + ' | ' + ' | '.join(f'{r[m]:.6f}' for m in METRICS) + ' |' for r in comparison)
    (output / 'FINAL_REPORT.md').write_text(
        '# E1 vs frozen E3: source-only development validation\n\n'
        'Status: E1_VALIDATION_COMPLETE_TEST_NOT_ACCESSED.\n\n'
        '| Arm | Macro-F1 | EZ-F1 | EZ-AUPRC | EZ-AUROC | MRR | Top1 |\n'
        '|---|---:|---:|---:|---:|---:|---:|\n' + table + '\n\n'
        'Primary values average five patient-equal validation folds (65 patient-fold cells, 47 unique patients). '
        'Same 80-patient, 255-seizure, 7,635-channel amended export and original fold roles as E3. '
        'Exactly one previously approved padded seizure is excluded; source caches are unchanged.\n\n'
        'E1 uses the same 44,401-parameter PR model, 30 supervised epochs, seeds, AdamW settings, '
        'patient-equal EZ:NEZ 2:1 BCE and validation checkpoint/threshold rules, but no SSL pretraining '
        'or pretrained-state loading. E3 checkpoints and outcomes are not modified or rerun.\n\n'
        'Paired comparison resamples 47 patient IDs 10,000 times, seed42, including all of each sampled '
        "patient's validation cells. Point estimates retain the 65-cell primary weighting. These are "
        'post-selection validation intervals, not independent test confirmation. The separate single-arm '
        'bootstrap first averages repeated appearances and uses unique-patient weighting.\n\n'
        'E3 trained with NumPy2.2.6 and finalized with isolated1.26.4; E1 uses isolated1.26.4 throughout. '
        'PyTorch2.8.0+cu128 and model/data/metric code remain unchanged. Prior NumPy metric, sampling and '
        'real validation forward parity passed; E1 original-loop/resume parity is separately audited. '
        'This engineering runtime difference must be disclosed, not hidden as an identical binary environment.\n\n'
        f"Predeclared E3-vs-E1 development gate passed: {gate['passed']}. "
        'E4/E5 are not authorized or run even if eligible. No outer test or independently confirmed EDF '
        'onset replay was performed. Neither these selected validation values nor their paired intervals '
        'establish independent clinical generalization or a benchmark Macro-F1>=0.70.\n', encoding='utf-8')
    atomic_json(output / 'RUN_STATUS.json', {'status': 'E1_VALIDATION_COMPLETE_TEST_NOT_ACCESSED',
        'variant': 'E1', 'folds': 5, 'ssl_epochs': 0, 'supervised_epochs_completed': 150,
        'source_only': True, 'outer_test_accessed': False, 'patients': 80, 'seizures': 255,
        'unique_channels': 7635, 'n_validation_cells': 65, 'n_unique_validation_patients': 47})
    print('E1_VALIDATION_COMPLETE_TEST_NOT_ACCESSED', flush=True)


if __name__ == '__main__':
    main()
