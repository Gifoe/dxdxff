import torch
import numpy as np
import pandas as pd

def compute_bandpower_torch(data_t, sfreq, bands):
    """
    data_t: (B, C, T) tensor
    """
    B, C, T = data_t.shape
    # Use torch.stft or real fft
    # Welch's method typically averages over sliding windows. 
    # For a fast approximation and since these are ~2s windows, we can just use rfft on the whole window.
    # To compute power spectral density:
    fft_vals = torch.fft.rfft(data_t, dim=-1)
    # Power = absolute value squared / sequence length
    power = torch.abs(fft_vals)**2 / T
    
    # frequencies:
    freqs = torch.fft.rfftfreq(T, d=1.0/sfreq, device=data_t.device)
    
    features = []
    for band_name, (l_freq, h_freq) in bands.items():
        idx_band = (freqs >= l_freq) & (freqs <= h_freq)
        # Sum over freq bins
        bp = torch.sum(power[..., idx_band], dim=-1)
        log_bp = torch.log(bp + 1e-10)
        features.append(log_bp)
        
    return torch.stack(features, dim=-1) # (B, C, n_bands)

def extract_absolute_features_torch(data_t, sfreq):
    """
    data_t: (B, C, T) tensor
    """
    bands = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'low_gamma': (30.0, 70.0),
        'high_gamma': (70.0, 150.0)
    }
    bp_features = compute_bandpower_torch(data_t, sfreq, bands) # (B, C, 6)
    
    rms = torch.sqrt(torch.mean(data_t**2, dim=-1)) # (B, C)
    var = torch.var(data_t, dim=-1, unbiased=False) # (B, C)
    line_length = torch.sum(torch.abs(torch.diff(data_t, dim=-1)), dim=-1) # (B, C)
    
    # Concatenate features along last dim
    # bp_features is (B, C, 6)
    # rms, var, line_length need to be (B, C, 1)
    concat_feats = torch.cat([
        bp_features,
        rms.unsqueeze(-1),
        var.unsqueeze(-1),
        line_length.unsqueeze(-1)
    ], dim=-1) # (B, C, 9)
    
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
        'node_abs_feat': abs_feat, # (B, C, 9)
        'node_diff_feat': diff_feat, # (B, C, 9)
        'has_neighbor': has_neighbor # (C,)
    }
