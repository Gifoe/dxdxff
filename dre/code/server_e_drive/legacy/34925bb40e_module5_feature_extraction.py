import numpy as np
import pandas as pd
import scipy.signal

def compute_bandpower(data, sfreq, bands):
    """
    data: (n_channels, n_samples)
    """
    n_ch, n_samples = data.shape
    # Using welch method
    freqs, psd = scipy.signal.welch(data, sfreq, nperseg=min(n_samples, int(sfreq)), axis=-1)
    
    features = []
    for band_name, (l_freq, h_freq) in bands.items():
        idx_band = np.logical_and(freqs >= l_freq, freqs <= h_freq)
        bp = np.sum(psd[:, idx_band], axis=1)
        # log bandpower (avoid log(0))
        log_bp = np.log(bp + 1e-10)
        features.append(log_bp)
        
    return np.stack(features, axis=1) # (n_ch, n_bands)

def extract_absolute_features(data, sfreq):
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 70),
        'high_gamma': (70, 150)
    }
    bp_features = compute_bandpower(data, sfreq, bands)
    
    rms = np.sqrt(np.mean(data**2, axis=-1))
    var = np.var(data, axis=-1)
    line_length = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)
    
    return np.concatenate([bp_features, rms[:, None], var[:, None], line_length[:, None]], axis=1)

def extract_local_differential(abs_features, contacts_meta):
    """
    Computes diff relative to local neighbors.
    abs_features: (n_channels, n_features) array matching contacts_meta order
    contacts_meta: DataFrame with stem and index
    """
    diff_feats = np.zeros_like(abs_features)
    has_neighbor = np.zeros(len(contacts_meta))
    
    df = contacts_meta.copy()
    
    for i, row in df.iterrows():
        stem = row['stem']
        idx = row['index']
        if pd.isna(idx):
            continue
            
        # find neighbors
        neighbors = df[(df['stem'] == stem) & ((df['index'] == idx - 1) | (df['index'] == idx + 1))]
        if len(neighbors) > 0:
            neighbor_indices = neighbors.index.tolist()
            neighbor_mean = np.mean(abs_features[neighbor_indices], axis=0)
            diff_feats[i] = abs_features[i] - neighbor_mean
            has_neighbor[i] = 1
            
    return diff_feats, has_neighbor

def process_window_features(win_data, sfreq, contacts_meta):
    """
    Extracts absolute and diff features for a single time window.
    """
    abs_feat = extract_absolute_features(win_data, sfreq)
    diff_feat, has_neighbor = extract_local_differential(abs_feat, contacts_meta)
    
    return {
        'node_abs_feat': abs_feat,
        'node_diff_feat': diff_feat,
        'has_neighbor': has_neighbor
    }
