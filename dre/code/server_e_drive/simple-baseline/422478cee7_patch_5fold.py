import re

with open("build_new2.py", "r", encoding="utf-8") as f:
    code = f.read()

func_5fold = r'''def five_fold_cv_channel_only(
    all_runs_data: Sequence[Dict[str, Any]],
    model_class,
    model_kwargs: Dict[str, Any],
    *,
    epochs: int = 200,
    device: Optional[str] = None,
    n_splits: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    random_state: int = 42,
    patience: int = 30,
):
    device_obj = configure_runtime(device)
    folds = build_kfold_splits(all_runs_data, n_splits=n_splits, random_state=random_state)
    
    all_channel_results = []
    
    for fold_idx, (train_runs_all, test_runs, train_subjects_all, _) in enumerate(folds, start=1):
        print(f"--- 5-Fold Outer Fold: {fold_idx}/{n_splits} ---")
        
        train_subjects_all = list(train_subjects_all)
        n_val = max(1, int(len(train_subjects_all) * 0.1))
        rng = np.random.default_rng(seed=random_state + fold_idx)
        rng.shuffle(train_subjects_all)
        
        val_subjects = train_subjects_all[:n_val]
        train_subjects = train_subjects_all[n_val:]
        
        train_runs = [run for run in train_runs_all if run["subject_id"] in train_subjects]
        val_runs = [run for run in train_runs_all if run["subject_id"] in val_subjects]
        
        scalers = fit_feature_scalers(train_runs, device=str(device_obj))
        train_runs_device = [process_run_to_device(run, scalers, device=str(device_obj)) for run in train_runs]
        val_runs_device = [process_run_to_device(run, scalers, device=str(device_obj)) for run in val_runs]
        test_runs_device = [process_run_to_device(run, scalers, device=str(device_obj)) for run in test_runs]

        patient_train_runs = defaultdict(list)
        for run in train_runs_device:
            patient_train_runs[run["subject_id"]].append(run)
            
        model = model_class(**model_kwargs).to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler() if str(device_obj).startswith("cuda") else None

        best_val_score = -1.0
        best_epoch = -1
        epochs_without_improvement = 0
        import copy
        best_model_state = copy.deepcopy(model.state_dict())

        for epoch_idx in range(epochs):
            rng.shuffle(train_subjects)
            epoch_losses = []

            for s_id in train_subjects:
                s_runs = patient_train_runs.get(s_id, [])
                if not s_runs: continue
                loss_dict = train_epoch(model, s_runs, optimizer, scaler)
                epoch_losses.append(loss_dict["loss"])
                
            val_results = eval_channel_epoch(model, val_runs_device)
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
                break
                
        model.load_state_dict(best_model_state)
        test_channel_results = eval_channel_epoch(model, test_runs_device)
        all_channel_results.extend(test_channel_results)
        
    return all_channel_results

'''

parts = code.split('def prepare_model_kwargs(')
_code = parts[0] + func_5fold + '\ndef prepare_model_kwargs(' + parts[1]

# Replace main method to run 5fold then LOPO
import re

main_pattern = r'''(def main\([\s\S]*?)(    channel_results = leave_one_patient_out_cv\([\s\S]*?return summarize_all_folds\(channel_results, output_path\))'''

new_main = r'''\1    print("\n" + "="*40)
    print("Running 5-Fold Cross Validation")
    print("="*40)
    channel_results_5fold = five_fold_cv_channel_only(
        all_runs_data,
        model_class=TwoBranchDynamicModel,
        model_kwargs=model_kwargs,
        epochs=epochs,
        device=str(device_obj),
        n_splits=n_splits,
    )
    
    output_path_5fold = output_path / "5fold"
    summarize_all_folds(channel_results_5fold, output_path_5fold)
    
    print("\n" + "="*40)
    print("Running Leave-One-Patient-Out (LOPO) Cross Validation")
    print("="*40)
    channel_results_lopo = leave_one_patient_out_cv(
        all_runs_data,
        model_class=TwoBranchDynamicModel,
        model_kwargs=model_kwargs,
        epochs=epochs,
        device=str(device_obj),
    )
    
    output_path_lopo = output_path / "lopo"
    return summarize_all_folds(channel_results_lopo, output_path_lopo)'''

_code = re.sub(main_pattern, new_main, _code)

with open("build_new2.py", "w", encoding="utf-8") as f:
    f.write(_code)
