import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import copy


def pairwise_ranking_loss(logits, targets):
    pos_mask = targets == 1
    neg_mask = targets == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]

    diff = neg_logits.unsqueeze(1) - pos_logits.unsqueeze(0)
    loss = F.softplus(diff).mean()
    return loss


def group_runs_by_patient(all_runs_data):
    patient_dict = defaultdict(list)
    for r in all_runs_data:
        patient_dict[r['subject_id']].append(r)
    return dict(patient_dict)


from sklearn.model_selection import KFold


def build_kfold_splits(all_runs_data, n_splits=5, random_state=42):
    np.random.seed(random_state)
    patient_dict = group_runs_by_patient(all_runs_data)

    patient_outcomes = {}
    for pid, runs in patient_dict.items():
        outcome = runs[0].get('outcome_binary', float('nan'))
        patient_outcomes[pid] = outcome

    subjects = np.array(sorted(list(patient_dict.keys())))
    np.random.seed(random_state)
    np.random.shuffle(subjects)

    folds = []

    kf = KFold(n_splits=n_splits)

    for train_idx, test_idx in kf.split(subjects):
        test_subs = subjects[test_idx].tolist()
        train_val_subs = subjects[train_idx].tolist()

        val_size = max(1, int(len(train_val_subs) * 0.1))
        val_subs = train_val_subs[:val_size]
        train_subs = train_val_subs[val_size:]

        folds.append({
            'test_subjects': test_subs,
            'val_subjects': val_subs,
            'train_subjects': train_subs
        })

    return folds


def fit_feature_scalers_gpu(train_runs, device):
    abs_sum = 0.0
    abs_sq_sum = 0.0
    abs_count = 0
    diff_sum = 0.0
    diff_sq_sum = 0.0
    diff_count = 0

    for run in train_runs:
        node_abs = torch.tensor(np.stack(run['node_abs']), dtype=torch.float32, device=device)
        node_diff = torch.tensor(np.stack(run['node_diff']), dtype=torch.float32, device=device)

        abs_sum = abs_sum + node_abs.sum(dim=(0, 1))
        abs_sq_sum = abs_sq_sum + (node_abs ** 2).sum(dim=(0, 1))
        abs_count += node_abs.shape[0] * node_abs.shape[1]

        diff_sum = diff_sum + node_diff.sum(dim=(0, 1))
        diff_sq_sum = diff_sq_sum + (node_diff ** 2).sum(dim=(0, 1))
        diff_count += node_diff.shape[0] * node_diff.shape[1]

    abs_mean = abs_sum / abs_count
    abs_var = (abs_sq_sum / abs_count) - (abs_mean ** 2)
    abs_std = torch.sqrt(torch.clamp(abs_var, min=1e-8))

    diff_mean = diff_sum / diff_count
    diff_var = (diff_sq_sum / diff_count) - (diff_mean ** 2)
    diff_std = torch.sqrt(torch.clamp(diff_var, min=1e-8))

    return {'abs_mean': abs_mean, 'abs_std': abs_std, 'diff_mean': diff_mean, 'diff_std': diff_std}


def process_and_move_run_to_device(run, scalers, device):
    gpu_run = {k: v for k, v in run.items() if
               k not in ['node_abs', 'node_diff', 'edge_indices', 'edge_attrs', 'labels']}

    # 1. 节点特征打包上显卡
    node_abs = torch.tensor(np.stack(run['node_abs']), dtype=torch.float32, device=device)
    node_diff = torch.tensor(np.stack(run['node_diff']), dtype=torch.float32, device=device)

    # 2. 显卡内并行标准化（极速）
    gpu_run['node_abs'] = (node_abs - scalers['abs_mean']) / scalers['abs_std']
    gpu_run['node_diff'] = (node_diff - scalers['diff_mean']) / scalers['diff_std']

    # 3. 核心修复：将几十万个小碎片拓扑图，打包成 1 个超级大张量，一次性发给显卡！
    gpu_run['edge_indices'] = torch.tensor(np.stack(run['edge_indices']), dtype=torch.long, device=device)
    gpu_run['edge_attrs'] = torch.tensor(np.stack(run['edge_attrs']), dtype=torch.float32, device=device)

    # 4. 标签上显卡
    gpu_run['labels'] = torch.tensor(run['labels'], dtype=torch.float32, device=device)

    return gpu_run


def map_phase_to_id(phase_group):
    pg = str(phase_group).lower()
    if 'interictal' in pg:
        return 0
    elif 'ictal' in pg:
        return 1
    return 2


def train_one_patient_step(model, aggregator, patient_runs, optimizer, device, lambda_outcome, outcome_pos_weight):
    model.train()
    if aggregator is not None:
        aggregator.train()

    optimizer.zero_grad()

    run_embs = []
    phase_ids = []
    total_channel_loss = torch.tensor(0.0, device=device)

    for run in patient_runs:
        gpu_run = run

        out = model(gpu_run['node_abs'], gpu_run['node_diff'], gpu_run['edge_indices'], gpu_run['edge_attrs'],
                    return_embeddings=True)
        channel_logits = out['channel_logits']
        run_emb = out['run_emb']

        labels = gpu_run['labels']
        num_pos = labels.sum()
        num_neg = (labels == 0).sum()

        if num_pos > 0:
            dyn_pos_weight = torch.clamp(num_neg / num_pos, min=1.0, max=20.0)
        else:
            dyn_pos_weight = torch.tensor(1.0, device=device)

        bce_loss = F.binary_cross_entropy_with_logits(channel_logits, labels, pos_weight=dyn_pos_weight)
        rank_loss = pairwise_ranking_loss(channel_logits, labels)

        run_ch_loss = bce_loss + 0.5 * rank_loss
        total_channel_loss += run_ch_loss

        run_embs.append(run_emb)
        phase_ids.append(map_phase_to_id(run.get('phase_group', 'unknown')))

    avg_channel_loss = total_channel_loss / max(len(patient_runs), 1)

    patient_label_raw = patient_runs[0].get('outcome_binary', float('nan'))
    has_outcome = pd.notna(patient_label_raw)

    if aggregator is not None and has_outcome and len(run_embs) > 0:
        stacked_embs = torch.stack(run_embs)
        stacked_phases = torch.tensor(phase_ids, dtype=torch.long, device=device)

        agg_out = aggregator(stacked_embs, stacked_phases)
        outcome_logit = agg_out['outcome_logit']

        pt_label_t = torch.tensor([patient_label_raw], dtype=torch.float32, device=device)

        outcome_loss = F.binary_cross_entropy_with_logits(
            outcome_logit,
            pt_label_t,
            pos_weight=torch.tensor([outcome_pos_weight], device=device)
        )

        total_loss = avg_channel_loss + lambda_outcome * outcome_loss
    else:
        total_loss = avg_channel_loss
        outcome_loss = torch.tensor(0.0, device=device)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if aggregator is not None:
        torch.nn.utils.clip_grad_norm_(aggregator.parameters(), 1.0)

    optimizer.step()

    return avg_channel_loss.item(), outcome_loss.item(), total_loss.item()


def eval_one_patient_step(model, aggregator, patient_runs, device):
    model.eval()
    if aggregator is not None:
        aggregator.eval()

    run_results = []
    run_embs = []
    phase_ids = []

    with torch.no_grad():
        for run in patient_runs:
            gpu_run = run
            out = model(gpu_run['node_abs'], gpu_run['node_diff'], gpu_run['edge_indices'], gpu_run['edge_attrs'],
                        return_embeddings=True)

            probs = torch.sigmoid(out['channel_logits']).cpu().numpy()

            run_results.append({
                'run_id': run['run_id'],
                'channel_names_norm': run['channel_names_norm'],
                'probs': probs,
                'labels': run['labels'].cpu().numpy() if isinstance(run['labels'], torch.Tensor) else run['labels']
            })

            run_embs.append(out['run_emb'])
            phase_ids.append(map_phase_to_id(run.get('phase_group', 'unknown')))

        patient_outcome_res = None
        has_outcome = pd.notna(patient_runs[0].get('outcome_binary', float('nan')))

        if aggregator is not None and has_outcome and len(run_embs) > 0:
            stacked_embs = torch.stack(run_embs)
            stacked_phases = torch.tensor(phase_ids, dtype=torch.long, device=device)
            agg_out = aggregator(stacked_embs, stacked_phases)
            out_prob = torch.sigmoid(agg_out['outcome_logit']).item()

            w_raw = agg_out['attn_weights']
            if w_raw is not None:
                attn_w = w_raw.squeeze(-1).cpu().numpy().tolist()
            else:
                attn_w = []

            first_run = patient_runs[0]

            patient_outcome_res = {
                'subject_id': first_run['subject_id'],
                'true_outcome': float(first_run['outcome_binary']),
                'pred_prob': float(out_prob),
                'pred_label': 1.0 if out_prob > 0.5 else 0.0,
                'run_attn_weights': attn_w,
                'engel': first_run.get('engel', 'N/A'),
                'therapy': first_run.get('therapy', 'N/A'),
                'implant': first_run.get('implant', 'N/A'),
                'target': first_run.get('target', 'N/A'),
                'lesion_status': first_run.get('lesion_status', 'N/A')
            }

    return run_results, patient_outcome_res


def five_fold_cv(all_runs_data, model_class, model_kwargs, aggregator_class, agg_kwargs, epochs=40, lambda_outcome=0.3,
                 device='gpu', n_splits=5):
    folds = build_kfold_splits(all_runs_data, n_splits=n_splits)
    patient_dict = group_runs_by_patient(all_runs_data)

    fold_results_channel = []
    fold_results_outcome = []

    for fold_idx, fold in enumerate(folds):
        test_subs = fold['test_subjects']
        train_subs = fold['train_subjects']

        print(f"\n--- Fold {fold_idx + 1}/{len(folds)} | Test: {', '.join(test_subs)} ---")

        train_runs = [r for sid in train_subs for r in patient_dict[sid]]

        print(f"  Train patients: {len(train_subs)}, Test patients: {len(test_subs)}")

        if len(train_runs) == 0:
            print(f"Skipping fold {fold_idx + 1} due to empty train set.")
            continue

        # Fit scalars on GPU directly
        scalers = fit_feature_scalers_gpu(train_runs, device)

        # Efficiently apply scalars and upload to GPU in one step
        train_runs_scaled_grouped = [[process_and_move_run_to_device(r, scalers, device) for r in patient_dict[sid]] for
                                     sid in train_subs]
        test_runs_scaled_grouped = [[process_and_move_run_to_device(r, scalers, device) for r in patient_dict[sid]] for
                                    sid in test_subs]

        model = model_class(**model_kwargs).to(device)
        aggregator = aggregator_class(**agg_kwargs).to(device) if aggregator_class else None

        params = list(model.parameters())
        if aggregator:
            params += list(aggregator.parameters())

        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

        tr_outcomes = [patient_dict[sid][0].get('outcome_binary', float('nan')) for sid in train_subs]
        valid_tr_out = [o for o in tr_outcomes if pd.notna(o)]
        sum_pos = sum(valid_tr_out)
        sum_neg = len(valid_tr_out) - sum_pos
        outcome_pos_weight = max(sum_neg / max(sum_pos, 1), 1.0)

        for epoch in range(epochs):
            np.random.shuffle(train_runs_scaled_grouped)

            tr_ch_loss, tr_out_loss, tr_tot_loss = 0, 0, 0

            for pt_runs in train_runs_scaled_grouped:
                cl, ol, tl = train_one_patient_step(model, aggregator, pt_runs, optimizer, device, lambda_outcome,
                                                    outcome_pos_weight)
                tr_ch_loss += cl
                tr_out_loss += ol
                tr_tot_loss += tl

            n_tr_pts = len(train_runs_scaled_grouped)

            if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
                print(
                    f"  Epoch {epoch + 1}/{epochs} - TrLoss: {tr_tot_loss / n_tr_pts:.4f} (Ch: {tr_ch_loss / n_tr_pts:.4f}, Out: {tr_out_loss / n_tr_pts:.4f})")

        # After training epochs, use the final model state to evaluate tests (No val set/early stopping needed)
        for test_sub, test_runs_s in zip(test_subs, test_runs_scaled_grouped):
            test_c_res, test_o_res = eval_one_patient_step(model, aggregator, test_runs_s, device)

            fold_results_channel.append({
                'subject_id': test_sub,
                'channel_results': test_c_res,
                'n_test_runs': len(test_runs_s)
            })

            if test_o_res is not None:
                fold_results_outcome.append(test_o_res)

    return fold_results_channel, fold_results_outcome
