import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from module1_file_discovery import discover_bids_files
from module2_preprocessing import load_and_preprocess_edf
from module3_labels_metadata import parse_labels_and_metadata
from module4_time_windows import create_time_windows
from module5_feature_extraction import process_window_features
from module6_graph_construction import build_dynamic_graph
from module7_model import TemporalDynamicGraphModel
from module8_training import leave_one_run_out_cv
from module9_inference_report import generate_patient_report

def process_single_run_helper(row, first_run_channels, unified_meta_df):
    print(f"  -> Processing run {row['run_id']}...")
    edf_path = row['edf_path']
    channels_path = row['channels_path']
    if not edf_path or not channels_path:
        return None
        
    try:
        raw, picked_chans, data = load_and_preprocess_edf(edf_path, first_run_channels, target_sfreq=250.0)
        
        # 【新增对齐逻辑】强制 raw 按照 unified_meta_df 的顺序排列通道
        target_ch_names = unified_meta_df['channel_name'].tolist()
        
        # 安全检查：确保我们要对齐的通道都在当前 EDF 实际加载出来的通道中
        missing_in_raw = set(target_ch_names) - set(raw.ch_names)
        if missing_in_raw:
            print(f"     Warning: Common channels {missing_in_raw} missing in EDF {edf_path}. Skipping run.")
            return None
            
        raw.reorder_channels(target_ch_names)
        data = raw.get_data() # 重新获取对齐后的数据
    except Exception as e:
        print(f"     Error loading EDF {edf_path}: {e}")
        return None
        
    windows_df = create_time_windows(raw, row, win_len_sec=2.0, step_sec=1.0)
    
    run_node_abs = []
    run_node_diff = []
    run_edge_indices = []
    run_edge_attrs = []
    
    for w_idx, w_row in windows_df.iterrows():
        if w_row['unusable_mask']:
            continue
            
        win_data = data[:, w_row['start_sample']:w_row['end_sample']]
        feats = process_window_features(win_data, sfreq=250.0, contacts_meta=unified_meta_df)
        e_idx, e_attr = build_dynamic_graph(win_data, unified_meta_df, top_k=5)
        
        run_node_abs.append(torch.tensor(feats['node_abs_feat'], dtype=torch.float32))
        run_node_diff.append(torch.tensor(feats['node_diff_feat'], dtype=torch.float32))
        run_edge_indices.append(torch.tensor(e_idx, dtype=torch.long))
        run_edge_attrs.append(torch.tensor(e_attr, dtype=torch.float32))
        
    if len(run_node_abs) > 0:
        return {
            'run_id': row['run_id'],
            'node_abs': run_node_abs,
            'node_diff': run_node_diff,
            'edge_indices': run_edge_indices,
            'edge_attrs': run_edge_attrs,
            'labels': torch.tensor(unified_meta_df['is_ez'].values, dtype=torch.long)
        }
    return None

def process_single_patient(subject_folder, dataset_dir, output_root):
    subject_id = subject_folder.name
    print(f"\n{'='*50}")
    print(f"Processing Subject: {subject_id}")
    print(f"{'='*50}")
    
    output_dir = os.path.join(output_root, subject_id)
    os.makedirs(output_dir, exist_ok=True)
    
    runs_df = discover_bids_files(dataset_dir, subject_id=subject_id)
    
    # Filter out empty or mostly empty entries if discover_bids_files matched anything invalid
    if runs_df.empty:
        print(f"No EDF files found for {subject_id}. skipping.")
        return
        
    valid_channels_sets = []
    for idx, row in runs_df.iterrows():
        if row['channels_path'] and os.path.exists(row['channels_path']):
            meta_df = parse_labels_and_metadata(row['channels_path'])
            valid_chans = set(meta_df[meta_df['is_valid'] == 1]['channel_name'].tolist())
            valid_channels_sets.append(valid_chans)
            
    if valid_channels_sets:
        common_channels = set.intersection(*valid_channels_sets)
    else:
        common_channels = set()
        
    first_run_channels_row = runs_df.dropna(subset=['channels_path'])
    if first_run_channels_row.empty:
        print(f"Error: No channels_path found in any run for {subject_id}. Skipping.")
        return
        
    first_run_channels = first_run_channels_row.iloc[0]['channels_path']
    unified_meta_df = parse_labels_and_metadata(first_run_channels)
    unified_meta_df = unified_meta_df[unified_meta_df['channel_name'].isin(common_channels)].reset_index(drop=True)
    
    print(f"Total valid common channels across runs: {len(unified_meta_df)}")
    
    if len(unified_meta_df) == 0:
        print(f"No common valid channels found for {subject_id}. Skipping.")
        return
        
    print(f"Launching parallel CPU processing (-1 cores) for {len(runs_df)} runs...")
    results = Parallel(n_jobs=-1)(
        delayed(process_single_run_helper)(row, first_run_channels, unified_meta_df) 
        for idx, row in runs_df.iterrows()
    )
    
    all_runs_data = [res for res in results if res is not None]
    
    # 【优化方向一】物理隔离，发作期与间期“分而治之” (Data Decoupling)
    phase_groups = {'Ictal': [], 'Interictal': []}
    for run in all_runs_data:
        if 'interictal' in run['run_id'].lower():
            phase_groups['Interictal'].append(run)
        else:
            phase_groups['Ictal'].append(run)
            
    patient_metrics = []
    phase_probs = {}
    
    for phase_name, phase_runs in phase_groups.items():
        if len(phase_runs) < 2:
            print(f"At least 2 valid runs are needed for CV for {subject_id} {phase_name}. Found {len(phase_runs)}. Skipping phase.")
            continue
            
        print(f"\n>>>> Normalizing features for {phase_name} runs...")
        all_abs_feats = []
        all_diff_feats = []
        for run in phase_runs:
            for t in range(len(run['node_abs'])):
                all_abs_feats.append(run['node_abs'][t].numpy())
                all_diff_feats.append(run['node_diff'][t].numpy())
                
        abs_scaler = StandardScaler()
        diff_scaler = StandardScaler()
        
        abs_stacked = np.concatenate(all_abs_feats, axis=0) 
        diff_stacked = np.concatenate(all_diff_feats, axis=0)
        
        abs_scaler.fit(abs_stacked)
        diff_scaler.fit(diff_stacked)
        
        for run in phase_runs:
            for t in range(len(run['node_abs'])):
                orig_abs = run['node_abs'][t].numpy()
                orig_diff = run['node_diff'][t].numpy()
                
                scaled_abs = abs_scaler.transform(orig_abs)
                scaled_diff = diff_scaler.transform(orig_diff)
                
                run['node_abs'][t] = torch.tensor(scaled_abs, dtype=torch.float32)
                run['node_diff'][t] = torch.tensor(scaled_diff, dtype=torch.float32)

        abs_dim = phase_runs[0]['node_abs'][0].shape[1]
        diff_dim = phase_runs[0]['node_diff'][0].shape[1]
        edge_dim = phase_runs[0]['edge_attrs'][0].shape[1]
        
        model_kwargs = {
            'abs_in_dim': abs_dim,
            'diff_in_dim': diff_dim,
            'edge_in_dim': edge_dim,
            'hidden_dim': 128,      # 【优化方向四】解放 5090，扩充模型容量 (从 32 改为 128)
            'dropout_rate': 0.4     # 增强防过拟合
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Starting {phase_name} training for {subject_id} on {device} (Leave-One-Run-Out CV)...")
        
        # Standard training for research, increasing epochs 
        cv_results = leave_one_run_out_cv(phase_runs, TemporalDynamicGraphModel, model_kwargs, epochs=100, device=device)
        
        # 【方案一】提取本次留一交叉验证的平均预测概率，临时存放在内存中用于之后的临床后融合 (Late Fusion)
        prob_matrix = np.zeros((len(cv_results), len(unified_meta_df)))
        for i, run_id in enumerate(cv_results.keys()):
            prob_matrix[i, :] = cv_results[run_id]['probs']
        phase_probs[phase_name] = np.mean(prob_matrix, axis=0)
        
        print(f"Generating {phase_name} report (Focusing on Top-K & AUC)...")
        true_labels = unified_meta_df['is_ez'].values
        phase_output_dir = os.path.join(output_dir, phase_name)
        metrics = generate_patient_report(cv_results, true_labels, unified_meta_df, phase_output_dir, patient_id=subject_id, outcome="Unknown")
        metrics['phase'] = phase_name
        patient_metrics.append(metrics)
        
    # === 【方案一】后融合 / 概率联合 (Ensemble Fusion) ===
    # 如果该病人同时具备发作期和间期的数据，在分别跑完模型后汇总它们的分数
    if 'Ictal' in phase_probs and 'Interictal' in phase_probs:
        print(f"\n>>>> Performing Ensemble Fusion (Late Fusion) for {subject_id}...")
        # 设定临床加权：发作期特征剧烈，置信度高占 0.7；间期特征偶发，作为补充占 0.3
        fused_prob = 0.7 * phase_probs['Ictal'] + 0.3 * phase_probs['Interictal']
        
        # 伪造一个 cv_results 字典格式以喂给你的打分报告系统
        fused_cv = {'Fused_Ensemble': {'probs': fused_prob.tolist()}}
        
        fused_output_dir = os.path.join(output_dir, 'Fused')
        fused_metrics = generate_patient_report(fused_cv, true_labels, unified_meta_df, fused_output_dir, patient_id=subject_id, outcome="Unknown")
        fused_metrics['phase'] = 'Fused'
        patient_metrics.append(fused_metrics)

    return patient_metrics

def main():
    dataset_dir = Path(r"E:\DRE-nips\dataest")
    output_root = Path(r"E:\DRE-nips\pipeline\outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Iterate dynamically through all patient folders that start with sub-HUP
    subject_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sub-HUP")]
    
    all_metrics = []
    
    for subject_folder in sorted(subject_dirs):
        m_list = process_single_patient(subject_folder, str(dataset_dir), str(output_root))
        if m_list is not None:
            for m in m_list:
                m['subject'] = subject_folder.name
                all_metrics.append(m)
                
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        
        print("\n" + "="*85)
        print("FINAL AVERAGE METRICS BY PHASE (Leave-One-Run-Out CV):")
        print("="*85)
        
        # 将 Fused（联合打分）一并打印出来进行对比
        for phase in ['Ictal', 'Interictal', 'Fused']:
            phase_df = df_metrics[df_metrics['phase'] == phase]
            if not phase_df.empty:
                avg = phase_df[['ACC', 'PREC', 'NPV', 'REC', 'SPEC', 'F1', 'AUC', 'AUC_PR', 'MCC', 'top_k_hit']].mean()
                print(f"\n--- Phase: {phase} ---")
                print("Model & Method & Top-K Hit & AUC & AUC-PR & ACC & PREC & NPV & REC & SPEC & F1 & MCC \\\\")
                print("\\midrule")
                print(f"Ours ({phase}) & TemporalGNN & {avg['top_k_hit']:.4f} & {avg['AUC']:.4f} & {avg['AUC_PR']:.4f} & {avg['ACC']:.4f} & {avg['PREC']:.4f} & {avg['NPV']:.4f} & {avg['REC']:.4f} & {avg['SPEC']:.4f} & {avg['F1']:.4f} & {avg['MCC']:.4f} \\\\")
        
        print("="*85)
        
        df_metrics.to_csv(output_root / 'all_patients_metrics_summary.csv', index=False)

if __name__ == "__main__":
    main()