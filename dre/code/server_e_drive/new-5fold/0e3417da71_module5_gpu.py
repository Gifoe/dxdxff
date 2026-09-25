import torch
import numpy as np
import pandas as pd

def extract_absolute_features_torch(data_t, sfreq, min_freq=1.0, max_freq=120.0):
    """
    Extracts high-dimensional Power Spectral Density (PSD) and basic time-domain features.
    data_t: (B, C, T) tensor
    """
    B, C, T = data_t.shape
    
    # Compute Power Spectral Density via FFT
    fft_vals = torch.fft.rfft(data_t, dim=-1)
    power = torch.abs(fft_vals)**2 / T
    freqs = torch.fft.rfftfreq(T, d=1.0/sfreq, device=data_t.device)
    
    # Extract raw frequency bins to provide high-entropy inputs for Cross-Attention
    idx_band = (freqs >= min_freq) & (freqs <= max_freq)
    psd_features = torch.log(power[..., idx_band] + 1e-10) # (B, C, F_bins) typically ~100-240 dims
    
    # Keep time-domain macro features
    rms = torch.sqrt(torch.mean(data_t**2, dim=-1, keepdim=True)) # (B, C, 1)
    var = torch.var(data_t, dim=-1, unbiased=False, keepdim=True) # (B, C, 1)
    line_length = torch.sum(torch.abs(torch.diff(data_t, dim=-1)), dim=-1, keepdim=True) # (B, C, 1)
    
    # Concatenate features along last dim
    concat_feats = torch.cat([psd_features, rms, var, line_length], dim=-1)
    
    return concat_feats

def extract_local_differential_batched(abs_features, contacts_meta):
    """
    abs_features: (B, C, F) tensor
    contacts_meta: DataFrame
    """
    B, C, F_dim = abs_features.shape
    device = abs_features.device
    
    diff_feats = torch.zeros_like(abs_features)
    has_neighbor = torch.zeros(C, device=device)
    
    df = contacts_meta.copy().reset_index(drop=True)
    
    # We can pre-compute a mapping of indices
    # We will compute it efficiently loop over metadata
    for i, row in df.iterrows():
        c_group = row['contact_group']
        c_num = row['contact_number']
        
        if pd.isna(c_num):
            continue
            
        neighbors = df[(df['contact_group'] == c_group) & 
                       ((df['contact_number'] == c_num - 1) | (df['contact_number'] == c_num + 1))]
        if len(neighbors) > 0:
            neighbor_indices = neighbors.index.tolist()
            # Mean across neighbors
            neighbor_mean = torch.mean(abs_features[:, neighbor_indices, :], dim=1) # (B, F_dim)
            diff_feats[:, i, :] = abs_features[:, i, :] - neighbor_mean
            has_neighbor[i] = 1.0
            
    return diff_feats, has_neighbor

def process_batched_window_features(windows_data, sfreq, contacts_meta, device='cpu'):
    """
    windows_data: numpy array (B, C, T) or torch tensor
    """
    if not isinstance(windows_data, torch.Tensor):
        data_t = torch.tensor(windows_data, dtype=torch.float32, device=device)
    else:
        data_t = windows_data.to(device=device, dtype=torch.float32)
        
    abs_feat = extract_absolute_features_torch(data_t, sfreq)
    diff_feat, has_neighbor = extract_local_differential_batched(abs_feat, contacts_meta)
    
    return {
        'node_abs_feat': abs_feat, # (B, C, High_dim)
        'node_diff_feat': diff_feat, # (B, C, High_dim)
        'has_neighbor': has_neighbor # (C,)
    }
