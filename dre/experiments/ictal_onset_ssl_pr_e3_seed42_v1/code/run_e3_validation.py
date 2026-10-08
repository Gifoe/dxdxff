"""E3 only: supplied model/protocol with atomic epoch resume and cached I/O.

No outer patient is scored here. Private caches/checkpoints/individual validation
records remain in the private runtime. Public outputs contain aggregates only.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


METRICS = ('macro_f1', 'ez_f1', 'ez_ap', 'ez_auroc', 'mrr', 'top1')


def atomic_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def atomic_torch(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].detach().cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.detach().cpu() for s in state['cuda']])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class CachedPatientFiles:
    """Exact input values from the package loader, with fold-specific access gate."""
    def __init__(self, folder, manifest, device, allowed, verify=True):
        from run_experiment import sha_file
        self.folder = Path(folder)
        self.meta = manifest['cohort']
        self.device = torch.device(device)
        self.allowed = set(allowed)
        self.cache = {}
        for pid in allowed:
            path = self.folder / self.meta[pid]['file']
            if verify and sha_file(path) != manifest['patient_file_sha256'][pid]:
                raise RuntimeError('PATIENT_FILE_HASH_MISMATCH')
            with np.load(path, allow_pickle=False) as data:
                pair = torch.tensor(data['pair'], device=self.device, dtype=torch.float32)
                present = torch.tensor(data['present'], device=self.device, dtype=torch.bool)
                labels = torch.tensor(data['labels_ez'], device=self.device, dtype=torch.float32)
            if pair.ndim != 4 or pair.shape[:2] != present.shape or pair.shape[2:] != (2, 2500):
                raise RuntimeError('PATIENT_TENSOR_CONTRACT_FAILED')
            if not present.any(0).all() or labels.shape != (pair.shape[1],):
                raise RuntimeError('CHANNEL_COVERAGE_FAILED')
            self.cache[pid] = (pair, present, labels)

    def get(self, pid, with_label=True):
        if pid not in self.allowed:
            raise RuntimeError('OUTER_TEST_ACCESS_FORBIDDEN')
        pair, present, labels = self.cache[pid]
        return pair, present, labels if with_label else None


def load_resume(path, binding, device):
    if not path.exists():
        return None
    result = torch.load(path, map_location=device, weights_only=False)
    if result['binding'] != binding:
        raise RuntimeError('RESUME_PROTOCOL_OR_SOURCE_CHANGED')
    return result


def ssl_stage(fold, loader, fit_ids, output, epochs, seed, binding, notify):
    from model import IctalLocalization, vicreg
    from run_experiment import set_seed, sha_file
    set_seed(seed)
    anchor = IctalLocalization(use_pr=False)
    encoder = anchor.encoder.to(loader.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)
    path = output / f'fold_{fold}_ssl_last_private.pt'
    saved = load_resume(path, binding, loader.device)
    history = []
    first = 1
    if saved:
        encoder.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        history = saved['history']
        first = saved['epoch'] + 1
    for epoch in range(first, epochs + 1):
        started = time.perf_counter()
        encoder.train()
        rows = []
        order = np.random.permutation(fit_ids)
        for offset in range(0, len(order), 2):
            all_v1, all_v2 = [], []
            for pid in order[offset:offset + 2]:
                pair, present, _ = loader.get(pid, with_label=False)
                eligible = torch.nonzero(present.any(0), as_tuple=False).flatten()
                if len(eligible) > 48:
                    eligible = eligible[torch.randperm(len(eligible), device=pair.device)[:48]]
                    pair = pair[:, eligible]
                    present = present[:, eligible]
                if int(present.sum()) < 2:
                    continue
                all_v1.append(encoder(pair, present, augment=True)[present])
                all_v2.append(encoder(pair, present, augment=True)[present])
            if not all_v1:
                continue
            view1, view2 = torch.cat(all_v1, 0), torch.cat(all_v2, 0)
            if len(view1) < 4:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, parts = vicreg(view1, view2)
            if not torch.isfinite(loss):
                raise RuntimeError('NONFINITE_SSL_LOSS')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
            optimizer.step()
            rows.append([float(loss.detach()), *parts])
        if not rows:
            raise RuntimeError('NO_VALID_SSL_PATIENTS')
        means = np.mean(rows, axis=0)
        history.append({'fold': fold, 'epoch': epoch, 'total': float(means[0]),
                        'inv': float(means[1]), 'var': float(means[2]), 'cov': float(means[3]),
                        'seconds': time.perf_counter() - started})
        atomic_torch(path, {'binding': binding, 'epoch': epoch, 'model': encoder.state_dict(),
                           'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'history': history})
        notify('SSL', fold, epoch, epochs, history[-1])
    final = output / f'fold_{fold}_ssl_true_encoder.pt'
    if not final.exists():
        atomic_torch(final, encoder.state_dict())
    else:
        existing = torch.load(final, map_location=loader.device, weights_only=True)
        if any(not torch.equal(v, existing[k]) for k, v in encoder.state_dict().items()):
            raise RuntimeError('FINAL_SSL_CHECKPOINT_CHANGED')
    return final, history, sha_file(final)


def supervised_stage(fold, loader, fit_ids, val_ids, output, epochs, seed, ssl_file, binding, notify):
    from model import IctalLocalization, patient_bce
    from run_experiment import set_seed, score_patient, pool, fit_threshold, sha_file, metric_one
    set_seed(seed)
    model = IctalLocalization(use_pr=True, use_attention=False).to(loader.device)
    model.encoder.load_state_dict(torch.load(ssl_file, map_location=loader.device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    path = output / f'fold_{fold}_E3_last_private.pt'
    saved = load_resume(path, binding, loader.device)
    history, best = [], None
    first = 1
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        history, best = saved['history'], saved['best']
        first = saved['epoch'] + 1
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
        history.append({'fold': fold, 'variant': 'E3', 'epoch': epoch,
                        'fit_bce': float(np.mean(losses)), 'val_ap': metrics['ez_ap'],
                        'val_f1_0p5': metrics['macro_f1'], 'seconds': time.perf_counter() - started})
        if best is None or key > best[0]:
            best = (key, cpu_state(model), epoch)
        atomic_torch(path, {'binding': binding, 'epoch': epoch, 'model': model.state_dict(),
                           'optimizer': optimizer.state_dict(), 'rng': rng_state(),
                           'history': history, 'best': best})
        notify('SUPERVISED', fold, epoch, epochs, history[-1])
    model.load_state_dict(best[1])
    model.eval()
    validation = [score_patient(model, loader, pid) for pid in val_ids]
    threshold = fit_threshold(validation)
    metrics = pool(validation, threshold)
    final = output / f'fold_{fold}_E3.pt'
    checkpoint = {'state_dict': model.state_dict(), 'variant': 'E3', 'fold': fold,
                  'selected_epoch': best[2], 'ez_threshold': threshold,
                  'selection': 'validation mean patient EZ-AP, tie F1@0.5',
                  'val_metrics': metrics, 'binding': binding}
    if not final.exists():
        atomic_torch(final, checkpoint)
    else:
        existing = torch.load(final, map_location=loader.device, weights_only=True)
        if existing['binding'] != binding or existing['selected_epoch'] != best[2] or existing['ez_threshold'] != threshold:
            raise RuntimeError('FINAL_SUPERVISED_CHECKPOINT_CHANGED')
        if any(not torch.equal(v, existing['state_dict'][k]) for k, v in model.state_dict().items()):
            raise RuntimeError('FINAL_SUPERVISED_WEIGHTS_CHANGED')
    selection = {'fold': fold, 'variant': 'E3', 'checkpoint': str(final),
                 'checkpoint_sha256': sha_file(final), 'selected_epoch': best[2],
                 'threshold': threshold, 'val_metrics': metrics}
    private_rows = [{'patient': r['id'], 'fold': fold, 'center': r['center'],
                     **metric_one(r['labels_ez'], r['scores_ez'], threshold)} for r in validation]
    return selection, history, private_rows


def summarize_validation(rows, selections, output):
    from run_experiment import write_csv
    write_csv(output / 'validation_by_fold.csv', [{'fold': s['fold'], 'variant': 'E3',
              'selected_epoch': s['selected_epoch'], 'threshold': s['threshold'],
              'n_patients': 13, **s['val_metrics']} for s in selections])
    by_center = []
    for center in sorted({r['center'] for r in rows}):
        group = [r for r in rows if r['center'] == center]
        by_center.append({'center': center, 'n_patient_fold_cells': len(group),
                          **{m: float(np.nanmean([r[m] for r in group])) for m in METRICS}})
    write_csv(output / 'validation_by_center.csv', by_center)
    ids = sorted({r['patient'] for r in rows})
    values = np.array([[np.nanmean([r[m] for r in rows if r['patient'] == pid])
                        for m in METRICS] for pid in ids])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    bootstrap = []
    for j, metric in enumerate(METRICS):
        distribution = np.nanmean(values[draws, j], axis=1)
        lower, upper = np.quantile(distribution, [.025, .975])
        bootstrap.append({'metric': metric, 'mean_unique_validation_patient': float(np.nanmean(values[:, j])),
                          'bootstrap_95_lower': float(lower), 'bootstrap_95_upper': float(upper),
                          'n_unique_patients': len(ids), 'draws': 10000, 'seed': 42,
                          'interpretation': 'descriptive post-selection validation; not outer-test CI'})
    write_csv(output / 'validation_patient_bootstrap.csv', bootstrap)
    atomic_json(output / 'validation_summary.json', {
        'variant': 'E3', 'n_patient_fold_cells': len(rows), 'n_unique_validation_patients': len(ids),
        'fold_mean': {m: float(np.mean([s['val_metrics'][m] for s in selections])) for m in METRICS},
        'unique_patient_mean': dict(zip(METRICS, np.nanmean(values, axis=0).tolist())),
        'interpretation': 'Development validation after checkpoint and threshold selection; no outer test.'})
    return bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--export', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--amendment', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package)))
    from run_experiment import sha_file, write_csv
    torch.set_num_threads(2)
    output, source = Path(args.output), Path(args.export)
    output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    guard = (output / 'runner.lock').open('a+b')
    guard.seek(0)
    if guard.read(1) == b'':
        guard.write(b'0')
        guard.flush()
    guard.seek(0)
    msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
    audit = json.loads((source / 'audit.json').read_text(encoding='utf-8'))
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    amendment = json.loads(Path(args.amendment).read_text(encoding='utf-8'))
    if (audit['status'], audit['n_patients'], audit['n_seizures'], audit['n_unique_channels'],
        audit['authorized_excluded_records']) != ('PASS_SOURCE_CODE_ALIGNMENT_ONLY', 80, 255, 7635, 1):
        raise RuntimeError('AMENDED_DATA_GATE_FAILED')
    if amendment['approval'] != 'USER_APPROVED' or amendment['outer_evaluation_authorized_in_this_amendment']:
        raise RuntimeError('VALIDATION_SCOPE_MISMATCH')
    binding = {'audit': sha_file(source / 'audit.json'), 'manifest': sha_file(source / 'manifest.json'),
               'model': sha_file(Path(args.package) / 'model.py'),
               'original_runner': sha_file(Path(args.package) / 'run_experiment.py'),
               'e3_runner': sha_file(__file__), 'amendment': sha_file(args.amendment),
               'variant': 'E3', 'seed': 42, 'ssl_epochs': 25, 'supervised_epochs': 30}
    lock = output / 'TRAINING_PROTOCOL_LOCK.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) != binding:
        raise RuntimeError('EXISTING_PROTOCOL_LOCK_CHANGED')
    if not lock.exists():
        atomic_json(lock, binding)
    def notify(stage, fold, epoch, total, row):
        state = {'status': 'RUNNING', 'pid': os.getpid(), 'stage': stage, 'fold': fold,
                 'epoch': epoch, 'epochs_in_stage': total, 'epoch_seconds': row.get('seconds'),
                 'completed_folds': fold - 1, 'outer_test_accessed': False}
        atomic_json(output / 'RUN_STATUS.json', state)
        print(json.dumps({**state, 'diagnostics': row}), flush=True)
    selections, ssl_history, supervised_history, validation_rows = [], [], [], []
    for entry in sorted(manifest['folds'], key=lambda f: f['fold_idx']):
        fold = entry['fold_idx']
        print(json.dumps({'stage': 'LOAD_FOLD_FIT_VAL', 'fold': fold,
                          'fit': len(entry['fit_subjects']), 'val': len(entry['validation_subjects'])}), flush=True)
        loader = CachedPatientFiles(source, manifest, args.device,
                                    entry['fit_subjects'] + entry['validation_subjects'])
        ssl_file, history, ssl_sha = ssl_stage(fold, loader, entry['fit_subjects'], output,
                                               25, 42 + fold, binding, notify)
        ssl_history.extend(history)
        selection, history, private_rows = supervised_stage(fold, loader, entry['fit_subjects'],
            entry['validation_subjects'], output, 30, 42 + fold * 100, ssl_file, binding, notify)
        selection['ssl_checkpoint_sha256'] = ssl_sha
        selections.append(selection)
        supervised_history.extend(history)
        validation_rows.extend(private_rows)
        atomic_json(output / f'fold_{fold}_selection.json', selection)
        write_csv(output / 'ssl_history.csv', ssl_history)
        write_csv(output / 'supervised_history.csv', supervised_history)
        del loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({'stage': 'FOLD_COMPLETE', 'fold': fold, 'selection': selection}), flush=True)
    atomic_json(output / 'validation_selections.json', selections)
    atomic_json(output / 'score_selection_lock.json', {'binding': binding, 'selections': selections,
                                                       'outer_test_accessed': False})
    atomic_json(output / 'validation_gate.json', {'status': 'NOT_ESTIMABLE_E3_ONLY',
        'reason': 'E1 control not requested; optional attention/sham not authorized or run.'})
    write_csv(output / 'validation_per_patient_private.csv', validation_rows)
    bootstrap = summarize_validation(validation_rows, selections, output)
    summary = json.loads((output / 'validation_summary.json').read_text(encoding='utf-8'))
    table = '\n'.join('| ' + r['metric'] + ' | ' + f"{r['mean_unique_validation_patient']:.6f}" +
        ' | ' + f"[{r['bootstrap_95_lower']:.6f}, {r['bootstrap_95_upper']:.6f}]" + ' |' for r in bootstrap)
    (output / 'FINAL_REPORT.md').write_text(
        '# E3 source-only five-fold validation\n\n'
        'Status: VALIDATION_COMPLETE_TEST_NOT_ACCESSED.\n\n'
        'The user approved excluding the single audited padded seizure. The model uses 255 seizures, '
        'all 80 patients and 7,635 channels, with original FIT/VAL/TEST membership. '
        'The source caches are untouched. Independent EDF onset confirmation remains 0/256.\n\n'
        f"Validation contains {summary['n_patient_fold_cells']} patient-fold cells and "
        f"{summary['n_unique_validation_patients']} unique patients. Checkpoints use validation EZ-AP "
        '(tie F1@0.5, earlier epoch); a single global threshold per fold uses validation Macro-F1. '
        'No outer patient was scored in its held-out fold. These post-selection validation metrics '
        'are optimistic and are not an 80-patient outer benchmark or an independent test.\n\n'
        '| Metric | Mean across unique validation patients | Descriptive patient-bootstrap 95% CI |\n'
        '|---|---:|---:|\n' + table + '\n\n'
        'All five folds used 25 SSL epochs and 30 supervised epochs, the supplied E3 topology, '
        'VICReg, patient-equal EZ:NEZ 2:1 BCE, seeds and learning rates. '
        'E0/E1/E2/E4/E5 were not trained. The contribution of SSL versus E1 cannot be inferred. '
        'No claim of benchmark Macro-F1 >=0.70 is made.\n\n'
        'Numerical resume tests and package tests are reported separately. Private checkpoints, '
        'patient records, waveforms and runtime logs must remain on the private server.\n', encoding='utf-8')
    atomic_json(output / 'RUN_STATUS.json', {'status': 'VALIDATION_COMPLETE_TEST_NOT_ACCESSED',
        'folds': 5, 'variant': 'E3', 'source_only': True, 'outer_test_accessed': False,
        'patients': 80, 'seizures': 255, 'unique_channels': 7635,
        'n_validation_cells': len(validation_rows),
        'n_unique_validation_patients': summary['n_unique_validation_patients']})
    print('VALIDATION_COMPLETE_TEST_NOT_ACCESSED', flush=True)


if __name__ == '__main__':
    main()
