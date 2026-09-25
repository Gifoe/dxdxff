import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import copy
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

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


def fit_feature_scalers(train_runs):
    abs_scaler = StandardScaler()
    diff_scaler = StandardScaler()

    all_abs = []
    all_diff = []

    for run in train_runs:
        all_abs.append(np.concatenate(run['node_abs'], axis=0))
        all_diff.append(np.concatenate(run['node_diff'], axis=0))

    if all_abs:
        abs_scaler.fit(np.concatenate(all_abs, axis=0))
    if all_diff:
        diff_scaler.fit(np.concatenate(all_diff, axis=0))

    return abs_scaler, diff_scaler

def apply_feature_scalers(runs, abs_scaler, diff_scaler):
    scaled_runs = copy.deepcopy(runs)
    for run in scaled_runs:
        T = len(run['node_abs'])
        for t in range(T):
            run['node_abs'][t] = abs_scaler.transform(run['node_abs'][t])
            run['node_diff'][t] = diff_scaler.transform(run['node_diff'][t])
    return scaled_runs

def move_run_to_device(run, device):
    gpu_run = copy.copy(run)
    gpu_run['node_abs'] = torch.tensor(np.stack(run['node_abs']), dtype=torch.float32, device=device)
    gpu_run['node_diff'] = torch.tensor(np.stack(run['node_diff']), dtype=torch.float32, device=device)
    gpu_run['edge_indices'] = [torch.tensor(ei, dtype=torch.long, device=device) for ei in run['edge_indices']]
    gpu_run['edge_attrs'] = [torch.tensor(ea, dtype=torch.float32, device=device) for ea in run['edge_attrs']]
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
    total_channel_loss = 0.0

    for run in patient_runs:
        gpu_run = run

        out = model(gpu_run['node_abs'], gpu_run['node_diff'], gpu_run['edge_indices'], gpu_run['edge_attrs'], return_embeddings=True)
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
            out = model(gpu_run['node_abs'], gpu_run['node_diff'], gpu_run['edge_indices'], gpu_run['edge_attrs'], return_embeddings=True)
            
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

def five_fold_cv(all_runs_data, model_class, model_kwargs, aggregator_class, agg_kwargs, epochs=40, lambda_outcome=0.3, device='cpu', n_splits=5):
    folds = build_kfold_splits(all_runs_data, n_splits=n_splits)
    patient_dict = group_runs_by_patient(all_runs_data)

    fold_results_channel = []
    fold_results_outcome = []

    for fold_idx, fold in enumerate(folds):
        test_subs = fold['test_subjects']
        val_subs = fold['val_subjects']
        train_subs = fold['train_subjects']

        print(f"\n--- Fold {fold_idx+1}/{len(folds)} | Test: {', '.join(test_subs)} ---")
        
        train_runs = [r for sid in train_subs for r in patient_dict[sid]]
        val_runs_grouped = [patient_dict[sid] for sid in val_subs]
        test_runs_grouped = [patient_dict[sid] for sid in test_subs]

        print(f"  Train patients: {len(train_subs)}, Val patients: {len(val_subs)}, Test patients: {len(test_subs)}")

        if len(train_runs) == 0:
            print(f"Skipping fold {fold_idx+1} due to empty train set.")
            continue

        abs_scaler, diff_scaler = fit_feature_scalers(train_runs)
        
        train_runs_scaled = apply_feature_scalers(train_runs, abs_scaler, diff_scaler)
        train_runs_scaled_grouped = [[move_run_to_device(r, device) for r in apply_feature_scalers(patient_dict[sid], abs_scaler, diff_scaler)] for sid in train_subs]
        val_runs_scaled_grouped = [[move_run_to_device(r, device) for r in apply_feature_scalers(patient_dict[sid], abs_scaler, diff_scaler)] for sid in val_subs]
        test_runs_scaled_grouped = [[move_run_to_device(r, device) for r in apply_feature_scalers(patient_dict[sid], abs_scaler, diff_scaler)] for sid in test_subs]

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
        
        best_val_score = -float('inf')
        best_model_state = None
        best_agg_state = None

        for epoch in range(epochs):
            np.random.shuffle(train_runs_scaled_grouped)
            
            tr_ch_loss, tr_out_loss, tr_tot_loss = 0, 0, 0
            
            for pt_runs in train_runs_scaled_grouped:
                cl, ol, tl = train_one_patient_step(model, aggregator, pt_runs, optimizer, device, lambda_outcome, outcome_pos_weight)
                tr_ch_loss += cl
                tr_out_loss += ol
                tr_tot_loss += tl
                
            n_tr_pts = len(train_runs_scaled_grouped)
            val_ch_metrics = []
            val_out_probs = []
            val_out_labels = []

            for pt_runs in val_runs_scaled_grouped:
                c_res, o_res = eval_one_patient_step(model, aggregator, pt_runs, device)
                
                all_probs = np.concatenate([r['probs'] for r in c_res])
                all_labs = np.concatenate([r['labels'] for r in c_res])
                
                try:
                    ch_auc = roc_auc_score(all_labs, all_probs)
                except:
                    ch_auc = 0.5
                val_ch_metrics.append(ch_auc)
                
                if o_res is not None and pd.notna(o_res['true_outcome']):
                    val_out_probs.append(o_res['pred_prob'])
                    val_out_labels.append(o_res['true_outcome'])

            avg_val_ch_auc = np.mean(val_ch_metrics) if val_ch_metrics else 0.5
            
            val_out_auc = 0.5
            if len(val_out_probs) > 1 and len(np.unique(val_out_labels)) > 1:
                try:
                    val_out_auc = roc_auc_score(val_out_labels, val_out_probs)
                except:
                    pass

            combined_val_score = avg_val_ch_auc + lambda_outcome * val_out_auc

            if combined_val_score > best_val_score:
                best_val_score = combined_val_score
                best_model_state = copy.deepcopy(model.state_dict())
                if aggregator:
                    best_agg_state = copy.deepcopy(aggregator.state_dict())

            if (epoch+1) % 1 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - TrLoss: {tr_tot_loss/n_tr_pts:.4f} (Ch: {tr_ch_loss/n_tr_pts:.4f}, Out: {tr_out_loss/n_tr_pts:.4f}) | ValScore: {combined_val_score:.4f} (ChAUC: {avg_val_ch_auc:.4f}, OutAUC: {val_out_auc:.4f})")

        model.load_state_dict(best_model_state)
        if aggregator and best_agg_state:
            aggregator.load_state_dict(best_agg_state)
            
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