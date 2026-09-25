import re

with open('d:/DRE-Research/Upenn-EI-Tpo/new-nips/build_new2_patched.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Make sure to import module9_inference_report.select_patient_predictions
import_str = 'from module9_inference_report import summarize_all_folds, select_patient_predictions\n'
text = re.sub(r'from module9_inference_report import summarize_all_folds\n', import_str, text)

# For leave_one_patient_out_cv
def replace_val_lopo(m):
    return """
        # Compute cap_ratio from training patients
        train_ez_ratios = []
        for s in train_subjects:
            s_runs = [r for r in train_runs if r["subject_id"] == s]
            if s_runs:
                lbls = np.array(s_runs[0]["labels"])
                if len(lbls) > 0:
                    train_ez_ratios.append(np.sum(lbls) / len(lbls))
        cap_ratio = float(np.percentile(train_ez_ratios, 90)) if train_ez_ratios else 0.35
        # cap_ratio should not be smaller than min possible 1/N
        
        scalers = fit_feature_scalers(train_runs, device=str(device_obj))"""

text = re.sub(r'\s+scalers = fit_feature_scalers\(train_runs, device=str\(device_obj\)\)', replace_val_lopo, text, count=1)


lopo_val_block = r"""
            mrae_sum = 0.0
            within_20pct = 0
            val_pr_score = 0.0
            val_patient_f1 = 0.0

            for pat_res in val_results:
                pat_runs = pat_res["channel_results"]
                res = select_patient_predictions(pat_runs, cap_ratio=cap_ratio)
                metrics = res["metrics"]
                
                mrae_sum += metrics["count_mrae"]
                if metrics["count_mrae"] <= 0.20:
                    within_20pct += 1
                
                if metrics["AUC_PR"] is not None:
                    val_pr_score += metrics["AUC_PR"]
                if metrics["F1"] is not None:
                    val_patient_f1 += metrics["F1"]

            val_mrae = mrae_sum / max(len(val_results), 1)
            val_within_20pct = within_20pct / max(len(val_results), 1)
            val_pr_score = val_pr_score / max(len(val_results), 1)
            val_patient_f1 = val_patient_f1 / max(len(val_results), 1)

            # primary: validation scores (PR and F1), tie: count_mrae (lower better)
            score = (val_pr_score + val_patient_f1) - val_mrae * 0.1"""

text = re.sub(r'\s+mrae_sum = 0\.0\n\s+within_20pct = 0\n\s+val_pr_score = 0\.0\n.*?score = val_pr_score - val_mrae \* 0\.1', lopo_val_block, text, flags=re.DOTALL, count=1)


# For five_fold_cv
text = re.sub(r'\s+scalers = fit_feature_scalers\(train_runs, device=str\(device_obj\)\)', replace_val_lopo, text, count=1)

five_fold_val_block = r"""
            mrae_sum = 0.0
            within_20pct = 0
            val_pr_score = 0.0
            val_patient_f1 = 0.0

            for pat_res in val_results:
                pat_runs = pat_res["channel_results"]
                res = select_patient_predictions(pat_runs, cap_ratio=cap_ratio)
                metrics = res["metrics"]
                
                mrae_sum += metrics["count_mrae"]
                if metrics["count_mrae"] <= 0.20:
                    within_20pct += 1
                
                if metrics["AUC_PR"] is not None:
                    val_pr_score += metrics["AUC_PR"]
                if metrics["F1"] is not None:
                    val_patient_f1 += metrics["F1"]

            val_mrae = mrae_sum / max(len(val_results), 1)
            val_within_20pct = within_20pct / max(len(val_results), 1)
            val_pr_score = val_pr_score / max(len(val_results), 1)
            val_patient_f1 = val_patient_f1 / max(len(val_results), 1)

            # primary: validation scores (PR + F1), tie: count_mrae (lower better)
            score = (val_pr_score + val_patient_f1) - val_mrae * 0.1"""

text = re.sub(r'\s+mrae_sum = 0\.0\n\s+within_20pct = 0\n\s+val_pr_score = 0\.0\n.*?score = val_pr_score - val_mrae \* 0\.1', five_fold_val_block, text, flags=re.DOTALL, count=1)


# Finally, pass cap_ratio to test phase report generation
# In lopo
lopo_test = r"""
        test_mrae_sum = 0.0
        test_within_20pct = 0
        from module9_inference_report import generate_channel_report
        for pat_res in test_results:
            pat_runs = pat_res["channel_results"]
            metrics, df_scores = generate_channel_report(pat_runs, len(pat_runs), output_dir, subject_id=pat_res["subject_id"], cap_ratio=cap_ratio)"""
            
text = re.sub(r'\s+test_mrae_sum = 0\.0\n\s+test_within_20pct = 0\n\s+for pat_res in test_results:.*?output_dir, subject_id=pat_res\["subject_id"\]\)', lopo_test, text, flags=re.DOTALL, count=1)


# In five fold
five_fold_test = r"""
        test_mrae_sum = 0.0
        test_within_20pct = 0
        from module9_inference_report import generate_channel_report
        for pat_res in test_results:
            pat_runs = pat_res["channel_results"]
            metrics, df_scores = generate_channel_report(pat_runs, len(pat_runs), output_dir, subject_id=pat_res["subject_id"], cap_ratio=cap_ratio)"""
text = re.sub(r'\s+test_mrae_sum = 0\.0\n\s+test_within_20pct = 0\n\s+for pat_res in test_results:.*?output_dir, subject_id=pat_res\["subject_id"\]\)', five_fold_test, text, flags=re.DOTALL, count=1)

with open('d:/DRE-Research/Upenn-EI-Tpo/new-nips/build_new2_patched.py', 'w', encoding='utf-8') as f:
    f.write(text)
