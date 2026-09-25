import re

with open("build_new2.py", "r", encoding="utf-8") as f:
    code = f.read()

train_epoch_old = r'''def train_epoch(
    model,
    patient_runs: Sequence[Dict[str, Any]],
    optimizer,
    scaler=None,
    max_windows_per_batch: int = 512,
    lambda_count: float = 0.3,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    channel_losses = []
    count_losses = []

    for run in patient_runs:
        n_windows = run["node_spec"].size(0)
        n_channels = run["labels"].size(0)
        true_count_run = run["labels"].sum()

        logits_chunks = []
        raw_count_chunks = []
        ratio_chunks = []

        for start_idx in range(0, n_windows, max_windows_per_batch):
            end_idx = min(start_idx + max_windows_per_batch, n_windows)
            with torch.autocast(device_type="cuda", enabled=(scaler is not None)):
                # model now returns up to 4 values depending on return_embeddings, but here return_embeddings is False by default
                out = model(
                    run["node_spec"][start_idx:end_idx],
                    run["node_conn"][start_idx:end_idx],
                    run["edge_indices"][start_idx:end_idx],
                    run["edge_attrs"][start_idx:end_idx],
                )
                chunk_logits, raw_count, ratio_run = out
            
            logits_chunks.append(chunk_logits)
            raw_count_chunks.append(raw_count)
            ratio_chunks.append(ratio_run)

        # Aggregate across chunks
        logits = torch.stack(logits_chunks, dim=0).mean(dim=0)
        raw_count_run = torch.stack(raw_count_chunks, dim=0).mean(dim=0).squeeze(-1)
        predicted_ratio_run = torch.stack(ratio_chunks, dim=0).mean(dim=0).squeeze(-1)

        channel_loss = _compute_channel_loss(logits, run["labels"])
        channel_losses.append(channel_loss)

        # Compute count loss
        pred_count = (raw_count_run + predicted_ratio_run * n_channels) / 2.0
        count_loss = F.smooth_l1_loss(pred_count, true_count_run.float())
        count_losses.append(count_loss)

    if not channel_losses:
        return {"loss": 0.0, "channel_loss": 0.0, "count_loss": 0.0}

    loss_ch = torch.stack(channel_losses).mean()
    loss_co = torch.stack(count_losses).mean()
    loss = loss_ch + lambda_count * loss_co

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "channel_loss": float(loss_ch.detach().cpu().item()),
        "count_loss": float(loss_co.detach().cpu().item()),
    }'''

train_epoch_new = r'''def train_epoch(
    model,
    patient_runs: Sequence[Dict[str, Any]],
    optimizer,
    scaler=None,
    max_windows_per_batch: int = 512,
    lambda_count: float = 0.3,
    lambda_rank: float = 0.1,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    channel_losses = []
    count_run_losses = []
    rank_losses = []
    
    pred_counts_run_cont = []
    true_counts_run = []
    w_r_list = []

    if not patient_runs:
        return {"loss": 0.0, "channel_loss": 0.0, "count_loss": 0.0}

    # patient level true count
    true_count_patient = patient_runs[0]["labels"].sum().float()
    N_patient = patient_runs[0]["labels"].size(0)

    for run in patient_runs:
        n_windows = run["node_spec"].size(0)
        n_channels = run["labels"].size(0)
        true_count_run = run["labels"].sum().float()

        logits_chunks = []
        raw_count_chunks = []
        ratio_chunks = []

        for start_idx in range(0, n_windows, max_windows_per_batch):
            end_idx = min(start_idx + max_windows_per_batch, n_windows)
            with torch.autocast(device_type="cuda", enabled=(scaler is not None)):
                out = model(
                    run["node_spec"][start_idx:end_idx],
                    run["node_conn"][start_idx:end_idx],
                    run["edge_indices"][start_idx:end_idx],
                    run["edge_attrs"][start_idx:end_idx],
                )
                chunk_logits, raw_count, ratio_run = out
            
            logits_chunks.append(chunk_logits)
            raw_count_chunks.append(raw_count)
            ratio_chunks.append(ratio_run)

        logits = torch.stack(logits_chunks, dim=0).mean(dim=0)
        raw_count_run = torch.stack(raw_count_chunks, dim=0).mean(dim=0).squeeze(-1)
        predicted_ratio_run = torch.stack(ratio_chunks, dim=0).mean(dim=0).squeeze(-1)

        channel_loss = _compute_channel_loss(logits, run["labels"])
        channel_losses.append(channel_loss)

        # rank loss
        is_ez = run["labels"] == 1
        is_non_ez = run["labels"] == 0
        if is_ez.any() and is_non_ez.any():
            logit_pos = logits[is_ez].mean()
            logit_neg = logits[is_non_ez].mean()
            rank_loss = torch.relu(0.5 - logit_pos + logit_neg)
            rank_losses.append(rank_loss)

        # run-level count
        pred_count_cont = (raw_count_run + predicted_ratio_run * n_channels) / 2.0
        delta_run = pred_count_cont - true_count_run
        
        tc_run_max1 = max(int(true_count_run.item()), 1)
        l_run_abs = F.smooth_l1_loss(pred_count_cont, true_count_run)
        l_run_rel = F.smooth_l1_loss(pred_count_cont / tc_run_max1, true_count_run / tc_run_max1)
        
        count_run_losses.append(l_run_abs + 0.5 * l_run_rel)
        
        pred_counts_run_cont.append(pred_count_cont)
        true_counts_run.append(true_count_run)

        # weights for patient fusion
        phase = run.get("phase_group", "unknown").lower()
        phase_weight = 2.0 if "ictal" in phase else 1.0
        confidence = torch.sigmoid(logits).mean().detach()
        w_r = n_windows * phase_weight * confidence
        w_r_list.append(w_r)

    if not channel_losses:
        return {"loss": 0.0, "channel_loss": 0.0, "count_loss": 0.0}

    loss_ch = torch.stack(channel_losses).mean()
    loss_rank = torch.stack(rank_losses).mean() if rank_losses else torch.tensor(0.0, device=loss_ch.device)
    loss_co_run = torch.stack(count_run_losses).mean() if count_run_losses else torch.tensor(0.0, device=loss_ch.device)

    # patient fusion count loss
    if sum(w_r_list) > 0:
        pred_count_pat_cont = sum(w * c for w, c in zip(w_r_list, pred_counts_run_cont)) / sum(w_r_list)
    else:
        pred_count_pat_cont = sum(pred_counts_run_cont) / len(pred_counts_run_cont)

    delta_pat = pred_count_pat_cont - true_count_patient
    tc_pat_max1 = max(int(true_count_patient.item()), 1)
    
    l_pat_abs = F.smooth_l1_loss(pred_count_pat_cont, true_count_patient)
    l_pat_rel = F.smooth_l1_loss(pred_count_pat_cont / tc_pat_max1, true_count_patient / tc_pat_max1)
    
    # over_weight = 1.0, under_weight = 1.0 (default)
    l_pat_bias = 1.0 * torch.relu(delta_pat)**2 + 1.0 * torch.relu(-delta_pat)**2
    
    loss_co_pat = l_pat_abs + 0.5 * l_pat_rel + 0.2 * l_pat_bias
    loss_co = 0.3 * loss_co_run + 0.7 * loss_co_pat

    loss = loss_ch + lambda_count * loss_co + lambda_rank * loss_rank

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "channel_loss": float(loss_ch.detach().cpu().item()),
        "count_loss": float(loss_co.detach().cpu().item()),
    }'''

code = code.replace(train_epoch_old, train_epoch_new)

# Now update eval and validation loop metrics in leave_one_patient_out_cv & five_fold_cv_channel_only

eval_epoch_old = r'''def eval_channel_epoch(model, test_runs: Sequence[Dict[str, Any]]):
    model.eval()
    per_subject_channel_results = defaultdict(list)
    max_windows_per_batch = 512

    with torch.no_grad():
        for run in test_runs:
            n_windows = run["node_spec"].size(0)
            n_channels = run["labels"].size(0)
            logits_chunks = []
            raw_count_chunks = []
            ratio_chunks = []

            for start_idx in range(0, n_windows, max_windows_per_batch):
                end_idx = min(start_idx + max_windows_per_batch, n_windows)
                out = model(
                    run["node_spec"][start_idx:end_idx],
                    run["node_conn"][start_idx:end_idx],
                    run["edge_indices"][start_idx:end_idx],
                    run["edge_attrs"][start_idx:end_idx],
                )
                chunk_logits, raw_count, ratio_run = out
                logits_chunks.append(chunk_logits)
                raw_count_chunks.append(raw_count)
                ratio_chunks.append(ratio_run)

            logits = torch.stack(logits_chunks, dim=0).mean(dim=0)
            raw_count_run = torch.stack(raw_count_chunks, dim=0).mean(dim=0).squeeze(-1).item()
            predicted_ratio_run = torch.stack(ratio_chunks, dim=0).mean(dim=0).squeeze(-1).item()

            pred_count = (raw_count_run + predicted_ratio_run * n_channels) / 2.0
            predicted_count_run = max(1, min(int(round(pred_count)), n_channels))
            true_count_run = int(run["labels"].sum().item())

            subject_id = run["subject_id"]
            # Weighting mechanism for later:
            # - n_windows
            # - phase (ictal gets higher)
            # - mean prob confidence
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            phase = run.get("phase_group", "unknown").lower()
            phase_weight = 2.0 if "ictal" in phase else 1.0
            confidence = np.mean(probs)
            w_r = n_windows * phase_weight * confidence

            per_subject_channel_results[subject_id].append(
                {
                    "subject_id": subject_id,
                    "run_id": run["run_id"],
                    "channel_names_norm": list(run.get("channel_names_norm", [])),
                    "probs": probs,
                    "labels": run["labels"].detach().cpu().numpy(),
                    "predicted_count_run": predicted_count_run,
                    "true_count_run": true_count_run,
                    "w_r": float(w_r),
                }
            )

    return [
        {
            "subject_id": subject_id,
            "channel_results": run_results,
            "n_test_runs": len(run_results),
        }
        for subject_id, run_results in per_subject_channel_results.items()
    ]'''

eval_epoch_new = r'''def eval_channel_epoch(model, test_runs: Sequence[Dict[str, Any]]):
    model.eval()
    per_subject_channel_results = defaultdict(list)
    max_windows_per_batch = 512

    with torch.no_grad():
        for run in test_runs:
            n_windows = run["node_spec"].size(0)
            n_channels = run["labels"].size(0)
            logits_chunks = []
            raw_count_chunks = []
            ratio_chunks = []

            for start_idx in range(0, n_windows, max_windows_per_batch):
                end_idx = min(start_idx + max_windows_per_batch, n_windows)
                out = model(
                    run["node_spec"][start_idx:end_idx],
                    run["node_conn"][start_idx:end_idx],
                    run["edge_indices"][start_idx:end_idx],
                    run["edge_attrs"][start_idx:end_idx],
                )
                chunk_logits, raw_count, ratio_run = out
                logits_chunks.append(chunk_logits)
                raw_count_chunks.append(raw_count)
                ratio_chunks.append(ratio_run)

            logits = torch.stack(logits_chunks, dim=0).mean(dim=0)
            raw_count_run = torch.stack(raw_count_chunks, dim=0).mean(dim=0).squeeze(-1).item()
            predicted_ratio_run = torch.stack(ratio_chunks, dim=0).mean(dim=0).squeeze(-1).item()

            pred_count = (raw_count_run + predicted_ratio_run * n_channels) / 2.0
            predicted_count_run = max(1, min(int(round(pred_count)), n_channels))
            true_count_run = int(run["labels"].sum().item())

            subject_id = run["subject_id"]
            # Weighting mechanism
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            phase = run.get("phase_group", "unknown").lower()
            phase_weight = 2.0 if "ictal" in phase else 1.0
            confidence = np.mean(probs)
            w_r = n_windows * phase_weight * confidence

            per_subject_channel_results[subject_id].append(
                {
                    "subject_id": subject_id,
                    "run_id": run["run_id"],
                    "channel_names_norm": list(run.get("channel_names_norm", [])),
                    "probs": probs,
                    "labels": run["labels"].detach().cpu().numpy(),
                    "predicted_count_run_cont": pred_count,
                    "predicted_count_run": predicted_count_run,
                    "true_count_run": true_count_run,
                    "w_r": float(w_r),
                }
            )

    return [
        {
            "subject_id": subject_id,
            "channel_results": run_results,
            "n_test_runs": len(run_results),
        }
        for subject_id, run_results in per_subject_channel_results.items()
    ]'''

code = code.replace(eval_epoch_old, eval_epoch_new)

# Update validation in leave_one_patient_out_cv

lopo_val_old = r'''            val_results = eval_channel_epoch(model, val_runs_device)
            
            # Simple heuristic evaluation for model selection.
            # Real evaluation logic is complex (see reporting), but we approximate count accuracy here for early stopping.
            exact_matches = 0
            val_pr_score = 0.0
            
            from sklearn.metrics import average_precision_score
            for pat_res in val_results:
                pat_runs = pat_res["channel_results"]
                
                w_runs = []
                p_counts = []
                
                pat_all_probs = []
                pat_all_labels = []
                
                for r in pat_runs:
                    w = r["w_r"]
                    w_runs.append(w)
                    p_counts.append(r["predicted_count_run"])
                    
                    pat_all_probs.extend(r["probs"])
                    pat_all_labels.extend(r["labels"])
                    
                if sum(w_runs) > 0:
                    pred_count_pat = round(sum(w * c for w, c in zip(w_runs, p_counts)) / sum(w_runs))
                else:
                    pred_count_pat = 1
                
                pat_labels_first_run = pat_runs[0]["labels"]
                true_cnt = int(pat_labels_first_run.sum())
                
                pred_count_pat = max(1, min(pred_count_pat, len(pat_labels_first_run)))
                if pred_count_pat == true_cnt:
                    exact_matches += 1
                    
                if len(np.unique(pat_all_labels)) > 1:
                    val_pr_score += average_precision_score(pat_all_labels, pat_all_probs)
                    
            val_exact_acc = exact_matches / len(val_results)
            val_pr_score = val_pr_score / len(val_results)
            
            score = val_exact_acc + val_pr_score * 0.0001 # Tie breaking
            
            if score > best_val_score + 1e-4:
                best_val_score = score
                best_epoch = epoch_idx
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                
            if (epoch_idx == 0 or (epoch_idx + 1) % 10 == 0):
                print(f"epoch {epoch_idx+1:03d}/{epochs} loss={float(np.mean(epoch_losses)):.4f} val_exact_acc={val_exact_acc:.4f}")
                
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch_idx+1}. Best epoch was {best_epoch+1}.")
                break'''

lopo_val_new = r'''            val_results = eval_channel_epoch(model, val_runs_device)
            mrae_sum = 0.0
            within_20pct = 0
            val_pr_score = 0.0
            
            from sklearn.metrics import average_precision_score
            for pat_res in val_results:
                pat_runs = pat_res["channel_results"]
                w_runs = [r["w_r"] for r in pat_runs]
                p_counts = [r["predicted_count_run_cont"] for r in pat_runs] # continuous
                
                pat_all_probs = []
                pat_all_labels = []
                for r in pat_runs:
                    pat_all_probs.extend(r["probs"])
                    pat_all_labels.extend(r["labels"])
                    
                if sum(w_runs) > 0:
                    pred_count_pat = round(sum(w * c for w, c in zip(w_runs, p_counts)) / sum(w_runs))
                else:
                    pred_count_pat = round(np.mean(p_counts))
                
                N_pat = len(pat_runs[0]["labels"])
                pred_count_pat = max(1, min(int(pred_count_pat), N_pat))
                true_cnt = int(pat_runs[0]["labels"].sum())
                
                rel_err = (pred_count_pat - true_cnt) / max(1, true_cnt)
                mrae_sum += abs(rel_err)
                if abs(rel_err) <= 0.20:
                    within_20pct += 1
                    
                if len(np.unique(pat_all_labels)) > 1:
                    val_pr_score += average_precision_score(pat_all_labels, pat_all_probs)
                    
            val_mrae = mrae_sum / max(len(val_results), 1)
            val_within_20pct = within_20pct / max(len(val_results), 1)
            val_pr_score = val_pr_score / max(len(val_results), 1)
            
            # primary: count_mrae (lower better), tie: within_20pct & PR
            score = -val_mrae + (val_within_20pct * 0.01) + (val_pr_score * 0.0001)
            
            if score > best_val_score + 1e-4:
                best_val_score = score
                best_epoch = epoch_idx
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                
            if (epoch_idx == 0 or (epoch_idx + 1) % 10 == 0):
                print(f"epoch {epoch_idx+1:03d}/{epochs} loss={float(np.mean(epoch_losses)):.4f} val_mrae={val_mrae:.4f} within_20pct={val_within_20pct:.4f}")
                
            if epoch_idx >= 50 and epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch_idx+1}. Best epoch was {best_epoch+1}.")
                break'''

code = code.replace(lopo_val_old, lopo_val_new)

# Update validation in five_fold_cv_channel_only

fold_val_old = r'''            val_results = eval_channel_epoch(model, val_runs_device)
            exact_matches = 0
            val_pr_score = 0.0
            
            from sklearn.metrics import average_precision_score
            for pat_res in val_results:
                pat_runs = pat_res["channel_results"]
                w_runs = [r["w_r"] for r in pat_runs]
                p_counts = [r["predicted_count_run"] for r in pat_runs]
                pat_all_probs = []
                pat_all_labels = []
                
                for r in pat_runs:
                    pat_all_probs.extend(r["probs"])
                    pat_all_labels.extend(r["labels"])
                    
                if sum(w_runs) > 0:
                    pred_count_pat = round(sum(w * c for w, c in zip(w_runs, p_counts)) / sum(w_runs))
                else:
                    pred_count_pat = 1
                
                pat_labels_first_run = pat_runs[0]["labels"]
                true_cnt = int(pat_labels_first_run.sum())
                pred_count_pat = max(1, min(pred_count_pat, len(pat_labels_first_run)))
                
                if pred_count_pat == true_cnt:
                    exact_matches += 1
                if len(np.unique(pat_all_labels)) > 1:
                    val_pr_score += average_precision_score(pat_all_labels, pat_all_probs)
                    
            val_exact_acc = exact_matches / len(val_results) if len(val_results) > 0 else 0.0
            val_pr_score = val_pr_score / len(val_results) if len(val_results) > 0 else 0.0
            score = val_exact_acc + val_pr_score * 0.0001
            
            if score > best_val_score + 1e-4:
                best_val_score = score
                best_epoch = epoch_idx
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                
            if (epoch_idx == 0 or (epoch_idx + 1) % 10 == 0):
                print(f"Fold {fold_idx} epoch {epoch_idx+1:03d}/{epochs} loss={float(np.mean(epoch_losses)):.4f} val_exact_acc={val_exact_acc:.4f}")
                
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch_idx+1}. Best epoch was {best_epoch+1}.")
                break'''

fold_val_new = r'''            val_results = eval_channel_epoch(model, val_runs_device)
            mrae_sum = 0.0
            within_20pct = 0
            val_pr_score = 0.0
            
            from sklearn.metrics import average_precision_score
            for pat_res in val_results:
                pat_runs = pat_res["channel_results"]
                w_runs = [r["w_r"] for r in pat_runs]
                p_counts = [r["predicted_count_run_cont"] for r in pat_runs]
                
                pat_all_probs = []
                pat_all_labels = []
                for r in pat_runs:
                    pat_all_probs.extend(r["probs"])
                    pat_all_labels.extend(r["labels"])
                    
                if sum(w_runs) > 0:
                    pred_count_pat = round(sum(w * c for w, c in zip(w_runs, p_counts)) / sum(w_runs))
                else:
                    pred_count_pat = round(np.mean(p_counts))
                
                N_pat = len(pat_runs[0]["labels"])
                pred_count_pat = max(1, min(int(pred_count_pat), N_pat))
                true_cnt = int(pat_runs[0]["labels"].sum())
                
                rel_err = (pred_count_pat - true_cnt) / max(1, true_cnt)
                mrae_sum += abs(rel_err)
                if abs(rel_err) <= 0.20:
                    within_20pct += 1
                    
                if len(np.unique(pat_all_labels)) > 1:
                    val_pr_score += average_precision_score(pat_all_labels, pat_all_probs)
                    
            val_mrae = mrae_sum / max(len(val_results), 1)
            val_within_20pct = within_20pct / max(len(val_results), 1)
            val_pr_score = val_pr_score / max(len(val_results), 1)
            
            score = -val_mrae + (val_within_20pct * 0.01) + (val_pr_score * 0.0001)
            
            if score > best_val_score + 1e-4:
                best_val_score = score
                best_epoch = epoch_idx
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                
            if (epoch_idx == 0 or (epoch_idx + 1) % 10 == 0):
                print(f"Fold {fold_idx} epoch {epoch_idx+1:03d}/{epochs} loss={float(np.mean(epoch_losses)):.4f} val_mrae={val_mrae:.4f} within_20pct={val_within_20pct:.4f}")
                
            if epoch_idx >= 50 and epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch_idx+1}. Best epoch was {best_epoch+1}.")
                break'''

code = code.replace(fold_val_old, fold_val_new)

with open("build_new2.py", "w", encoding="utf-8") as f:
    f.write(code)


# Now patch module9_inference_report.py
with open("module9_inference_report.py", "r", encoding="utf-8") as f:
    code_rpt = f.read()

rpt_old = r'''        run_abs_errors.append(abs(rpc - rtc))
        run_matches.append(int(rpc == rtc))
        
        # Rank channels in this run
        run_probs = run_prediction["probs"]
        run_channels = run_prediction["channel_names_norm"]
        ch_prob_pairs = list(zip(run_channels, run_probs))
        ch_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        top_k_run_set = set(ch for ch, p in ch_prob_pairs[:rpc])
        
        for ch, prob, lab in zip(run_channels, run_probs, run_prediction["labels"]):
            channel_run_probs[ch].append((prob, ch in top_k_run_set))
            channel_labels[ch] = lab

    # 4. Patient-level count fusion
    if sum(run_weights) > 0:
        predicted_count_patient = int(round(sum(w * c for w, c in zip(run_weights, run_pred_counts)) / sum(run_weights)))
    else:
        predicted_count_patient = int(np.mean(run_pred_counts))
        
    all_channels = list(channel_labels.keys())
    N_patient = len(all_channels)
    predicted_count_patient = max(1, min(predicted_count_patient, N_patient))
    
    true_ez_channels = [ch for ch, lab in channel_labels.items() if lab == 1.0]
    true_count_patient = len(true_ez_channels)'''

rpt_new = r'''        run_abs_errors.append(abs(rpc - rtc))
        run_matches.append(int(rpc == rtc))
        
        # Rank channels in this run
        run_probs = run_prediction["probs"]
        run_channels = run_prediction["channel_names_norm"]
        ch_prob_pairs = list(zip(run_channels, run_probs))
        ch_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        top_k_run_set = set(ch for ch, p in ch_prob_pairs[:rpc])
        
        for ch, prob, lab in zip(run_channels, run_probs, run_prediction["labels"]):
            channel_run_probs[ch].append((prob, ch in top_k_run_set))
            channel_labels[ch] = lab

    # 4. Patient-level count fusion
    if sum(run_weights) > 0:
        # Use continuous values stored in "predicted_count_run_cont" if possible? In our eval format they are saved
        p_counts_cont = [r.get("predicted_count_run_cont", r["predicted_count_run"]) for r in test_results_aggregated]
        predicted_count_patient = int(round(sum(w * c for w, c in zip(run_weights, p_counts_cont)) / sum(run_weights)))
    else:
        predicted_count_patient = int(round(np.mean(run_pred_counts)))
        
    all_channels = list(channel_labels.keys())
    N_patient = len(all_channels)
    predicted_count_patient = max(1, min(predicted_count_patient, N_patient))
    
    true_ez_channels = [ch for ch, lab in channel_labels.items() if lab == 1.0]
    true_count_patient = len(true_ez_channels)
    
    delta_count = predicted_count_patient - true_count_patient
    abs_err = abs(delta_count)
    rel_err = delta_count / max(1, true_count_patient)
    if delta_count > 0:
        prediction_direction = "overpredict"
    elif delta_count < 0:
        prediction_direction = "underpredict"
    else:
        prediction_direction = "exact_match"
'''

code_rpt = code_rpt.replace(rpt_old, rpt_new)

rpt_old2 = r'''    count_abs_error_patient = abs(predicted_count_patient - true_count_patient)
    metrics = {
        "n_true_ez": int(true_count_patient),
        "n_pred_ez": int(predicted_count_patient),
        "top_k_hit": float(top_k_hit),
        # run-level metrics aggregated
        "true_count_run_mean": float(np.mean(run_true_counts)),
        "predicted_count_run_mean": float(np.mean(run_pred_counts)),
        "count_abs_error_run_mean": float(np.mean(run_abs_errors)),
        "count_match_run_acc": float(np.mean(run_matches)),
        
        # patient count metrics
        "true_count_patient": int(true_count_patient),
        "predicted_count_patient": int(predicted_count_patient),
        "count_abs_error_patient": int(count_abs_error_patient),
        "count_match_patient": int(count_abs_error_patient == 0),
        "count_exact_match_acc": float(count_abs_error_patient == 0),
        "count_within_1_acc": float(count_abs_error_patient <= 1),
        "count_mae": float(count_abs_error_patient),
        
        "ACC": float(acc),
        "PREC": float(prec),
        "NPV": float(npv),
        "REC": float(rec),
        "SPEC": float(spec),
        "F1": float(f1),
        "AUC": float(auc_val) if not np.isnan(auc_val) else None,
        "AUC_PR": float(auc_pr_val) if not np.isnan(auc_pr_val) else None,
        "MCC": float(mcc),
        "selection_strategy": "top_predicted_K_patient",
    }

    report = {
        "subject_id": subject_id,
        "n_test_runs": n_test_runs,
        "Patient_Level_Counts": {
            "true_count_patient": int(true_count_patient),
            "predicted_count_patient": int(predicted_count_patient),
        },
        "Run_Level_Counts": {
            "predicted_count_runs": run_pred_counts,
            "weights": run_weights,
        },
        "Results": {
            "True_EZ": true_ez_channels,
            "Predicted_EZ": pred_ez,
            "True_Positive": tp_ez,
            "False_Positive": fp_ez,
            "False_Negative": fn_ez,
        },
        "Metrics": metrics,
    }'''

rpt_new2 = r'''    metrics = {
        "n_true_ez": int(true_count_patient),
        "n_pred_ez": int(predicted_count_patient),
        "top_k_hit": float(top_k_hit),
        # run-level metrics aggregated
        "true_count_run_mean": float(np.mean(run_true_counts)),
        "predicted_count_run_mean": float(np.mean(run_pred_counts)),
        "count_abs_error_run_mean": float(np.mean(run_abs_errors)),
        "count_match_run_acc": float(np.mean(run_matches)),
        
        # patient count metrics
        "true_count_patient": int(true_count_patient),
        "predicted_count_patient": int(predicted_count_patient),
        "count_exact_match_acc": float(abs_err == 0),
        "count_mae": float(abs_err),
        "count_mrae": float(abs(rel_err)),
        "count_bias": float(delta_count),
        "count_rel_bias": float(rel_err),
        "overpredict_rate": float(delta_count > 0),
        "underpredict_rate": float(delta_count < 0),
        "within_10pct_acc": float(abs(rel_err) <= 0.10),
        "within_20pct_acc": float(abs(rel_err) <= 0.20),
        
        "ACC": float(acc),
        "PREC": float(prec),
        "NPV": float(npv),
        "REC": float(rec),
        "SPEC": float(spec),
        "F1": float(f1),
        "AUC": float(auc_val) if not np.isnan(auc_val) else None,
        "AUC_PR": float(auc_pr_val) if not np.isnan(auc_pr_val) else None,
        "MCC": float(mcc),
        "selection_strategy": "top_predicted_K_patient",
    }

    report = {
        "subject_id": subject_id,
        "n_test_runs": n_test_runs,
        "Patient_Level_Counts": {
            "true_count_patient": int(true_count_patient),
            "predicted_count_patient": int(predicted_count_patient),
            "delta_count": int(delta_count),
            "abs_err": int(abs_err),
            "rel_err": float(rel_err),
            "prediction_direction": prediction_direction,
        },
        "Run_Level_Counts": {
            "predicted_count_runs": run_pred_counts,
            "true_count_runs": run_true_counts,
            "weights": run_weights,
            "count_abs_errors": run_abs_errors,
        },
        "Results": {
            "True_EZ": true_ez_channels,
            "Predicted_EZ_Thresh": pred_ez,
            "True_Positive": tp_ez,
            "False_Positive": fp_ez,
            "False_Negative": fn_ez,
        },
        "Metrics": metrics,
    }'''

code_rpt = code_rpt.replace(rpt_old2, rpt_new2)

with open("module9_inference_report.py", "w", encoding="utf-8") as f:
    f.write(code_rpt)
