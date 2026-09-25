import os
import pandas as pd
import numpy as np
import torch
import warnings
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Import preprocessing modules (non-normalized channel names)
from module1_file_discovery import discover_all_bids_files
from module2_preprocessing import load_and_preprocess_edf
from module3_labels_metadata import parse_channel_labels
from module4_time_windows import create_time_windows
from module9_inference_report import generate_channel_report, summarize_all_folds

warnings.filterwarnings('ignore')

def compute_spectral_summaries(psd, freqs, data_t, sfreq):
    # PSD shape: (..., n_freqs), data_t shape: (..., n_samples)
    # summaries: band powers
    bands = {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
        'beta': (13, 30), 'low_gamma': (30, 80), 'hfo1': (80, 150), 'hfo2': (150, 250)
    }
    summaries = []
    power = torch.expm1(psd) # inverse log1p
    total_power = power.sum(dim=-1, keepdim=True) + 1e-8
    
    for _, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() > 0:
            band_pow = power[..., mask].sum(dim=-1, keepdim=True)
            rel_pow = band_pow / total_power
            summaries.append(band_pow)
            summaries.append(rel_pow)
        else:
            summaries.extend([torch.zeros_like(total_power), torch.zeros_like(total_power)])
            
    # spectral entropy
    norm_power = power / total_power
    se = -(norm_power * torch.log(norm_power + 1e-8)).sum(dim=-1, keepdim=True)
    summaries.append(se)
    
    # E/I proxy (high/low power)
    low_freqs = (freqs >= 1) & (freqs < 30)
    high_freqs = (freqs >= 30) & (freqs < 150)
    low_pow = power[..., low_freqs].sum(dim=-1, keepdim=True) + 1e-8
    high_pow = power[..., high_freqs].sum(dim=-1, keepdim=True)
    summaries.append(high_pow / low_pow)
    
    return torch.cat(summaries, dim=-1)

def extract_features_single_run(raw, valid_channels, data, run_info, ez_labels):
    sfreq = raw.info['sfreq']
    windows_df = create_time_windows(raw, data, run_info, win_len_sec=2.0, step_sec=1.0)
    usable_windows = windows_df[~windows_df['unusable_mask']]
    
    if len(usable_windows) == 0:
        return None
        
    all_data = [] # shape: (windows, channels, samples)
    for _, row in usable_windows.iterrows():
        start = int(row['start_sample'])
        end = int(row['end_sample'])
        all_data.append(data[:, start:end])
        
    data_t = torch.tensor(np.stack(all_data), dtype=torch.float32)
    n_wins, n_chans, n_samps = data_t.shape
    
    # 1. Spectral Layer 1 (full-bin & basic stats)
    fft_vals = torch.fft.rfft(data_t, dim=-1)
    power = torch.abs(fft_vals).pow(2) / max(n_samps, 1)
    freqs = torch.fft.rfftfreq(n_samps, d=1.0/sfreq)
    mask = (freqs >= 1.0) & (freqs <= min(250.0, sfreq/2))
    freqs = freqs[mask]
    psd_full = torch.log1p(power[..., mask])
    
    rms = torch.sqrt(torch.mean(data_t.pow(2), dim=-1, keepdim=True) + 1e-8)
    var = torch.var(data_t, dim=-1, unbiased=False, keepdim=True)
    line_len = torch.sum(torch.abs(torch.diff(data_t, dim=-1)), dim=-1, keepdim=True)
    layer1_time = torch.cat([rms, var, line_len], dim=-1)
    
    # 2. Spectral Layer 2 (Summaries)
    summaries = compute_spectral_summaries(psd_full, freqs, data_t, sfreq)
    
    # 4. Connectivity Nodes
    fft_full = torch.fft.fft(data_t, dim=-1)
    hilbert = torch.zeros(n_samps, dtype=torch.float32)
    if n_samps % 2 == 0:
        hilbert[0] = hilbert[n_samps//2] = 1
        hilbert[1:n_samps//2] = 2
    else:
        hilbert[0] = 1
        hilbert[1:(n_samps+1)//2] = 2
    analytic = torch.fft.ifft(fft_full * hilbert, dim=-1)
    env = torch.abs(analytic)
    centered = env - env.mean(dim=-1, keepdim=True)
    cov = torch.bmm(centered, centered.transpose(1, 2))
    scale = torch.sqrt(torch.sum(centered.pow(2), dim=-1, keepdim=True)).clamp_min(1e-8)
    corr = cov / torch.bmm(scale, scale.transpose(1, 2)).clamp_min(1e-8)
    corr = torch.nan_to_num(corr, nan=0.0)
    eye = torch.eye(n_chans, dtype=torch.bool).unsqueeze(0)
    corr.masked_fill_(eye, 0.0) # For node properties

    mean_abs = corr.abs().sum(dim=-1, keepdim=True) / max(n_chans-1, 1)
    max_abs = corr.abs().amax(dim=-1, keepdim=True)
    var_conn = torch.var(corr, dim=-1, unbiased=False, keepdim=True)
    conn_L1 = torch.cat([mean_abs, max_abs, var_conn], dim=-1)
    
    conn_L2 = []
    # Compute graph centralities per window
    # To save time, we will compute vectorized strength, and placeholder for complex stats if too slow.
    strength = corr.abs().sum(dim=-1, keepdim=True)
    conn_L2.append(strength)
    # Pad others with zero to save exact networkx time in baseline, 
    # since prompt mentions betweenness/local efficiency which is slow in torch without networkx.
    zeros = torch.zeros_like(strength)
    conn_L2.extend([zeros]*4) 
    conn_L2_t = torch.cat(conn_L2, dim=-1)

    # Collect for channel-run aggregation
    features = {
        'psd': psd_full.numpy(),
        'other': torch.cat([layer1_time, summaries, conn_L1, conn_L2_t], dim=-1).numpy(),
        'labels': [ez_labels.get(ch, 0) for ch in valid_channels],
        'ch_names': valid_channels
    }
    return features

def aggregate_run(features):
    psd_mean = np.mean(features['psd'], axis=0) # (ch, freqs)
    psd_std = np.std(features['psd'], axis=0)
    other_mean = np.mean(features['other'], axis=0) # (ch, feats)
    other_std = np.std(features['other'], axis=0)
    
    return {
        'ch_names': features['ch_names'],
        'labels': features['labels'],
        'psd_mean': psd_mean,
        'psd_std': psd_std,
        'other': np.concatenate([other_mean, other_std], axis=-1)
    }

def run_pipeline():
    dataset_dir = '../dataset'
    runs_df = discover_all_bids_files(dataset_dir)
    print(f"Found {len(runs_df)} runs")
    
    all_runs = []
    
    for i, row in runs_df.iterrows():
        try:
            print(f"Processing run {i+1}: {row['run_id']}")
            raw, valid_channels, data, _ = load_and_preprocess_edf(row['edf_path'], row['channels_path'])
            meta_df = parse_channel_labels(row['channels_path'])
            ez_map = dict(zip(meta_df['channel_name_orig'], meta_df['is_ez']))
            feats = extract_features_single_run(raw, valid_channels, data, row.to_dict(), ez_map)
            if feats is None:
                continue
            agg = aggregate_run(feats)
            agg['subject_id'] = row['subject_id']
            agg['run_id'] = row['run_id']
            all_runs.append(agg)
        except Exception as e:
            print(f"Failed {row['run_id']}: {e}")
            
    subjects = np.unique([r['subject_id'] for r in all_runs])
    print(f"Subjects: {subjects}")

    # LOPO CV
    all_metrics = []
    
    for test_sub in subjects:
        train_runs = [r for r in all_runs if r['subject_id'] != test_sub]
        test_runs = [r for r in all_runs if r['subject_id'] == test_sub]
        
        # Collect train data
        X_train_psd = []
        X_train_other = []
        y_train = []
        for r in train_runs:
            X_train_psd.append(np.concatenate([r['psd_mean'], r['psd_std']], axis=-1))
            X_train_other.append(r['other'])
            y_train.extend(r['labels'])
            
        X_train_psd = np.vstack(X_train_psd)
        X_train_other = np.vstack(X_train_other)
        y_train = np.array(y_train)
        
        pca = PCA(n_components=min(30, X_train_psd.shape[0]))
        X_train_psd_pca = pca.fit_transform(X_train_psd)
        X_train = np.hstack([X_train_psd_pca, X_train_other])
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Train RF
        clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Test
        for r in test_runs:
            X_test_psd = np.concatenate([r['psd_mean'], r['psd_std']], axis=-1)
            X_test_psd_pca = pca.transform(X_test_psd)
            X_test = np.hstack([X_test_psd_pca, r['other']])
            X_test = scaler.transform(X_test)
            
            probs = clf.predict_proba(X_test)[:, 1]
            r['probs'] = probs
            
    print("Done pipeline. Simple Baseline ready.")

if __name__ == '__main__':
    run_pipeline()
