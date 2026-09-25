import mne
import numpy as np
import pandas as pd

def load_and_preprocess_edf(edf_path, channels_path, target_sfreq=250.0):
    channels_df = pd.read_csv(channels_path, sep='	')
    channels_df['norm_name'] = channels_df['name'].str.strip()
    channels_df['type_upper'] = channels_df['type'].astype(str).str.upper()
    valid_mask = channels_df['type_upper'].isin(['ECOG', 'SEEG', 'STEREOEEG']) & (channels_df['status'] != 'bad')
    valid_channels = channels_df.loc[valid_mask, 'norm_name'].tolist()
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    raw.rename_channels({ch: ch.strip() for ch in raw.ch_names})
    
    existing_valid = [ch for ch in valid_channels if ch in raw.ch_names]
    raw.pick(existing_valid)
    
    freqs = np.arange(60, raw.info['sfreq'] / 2, 60)
    if len(freqs) > 0:
        raw.notch_filter(freqs=freqs, verbose='ERROR')
        
    h_freq = min(150.0, raw.info['sfreq'] / 2 - 1.0)
    raw.filter(l_freq=1.0, h_freq=h_freq, verbose='ERROR')
    
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)
        
    data = raw.get_data()
    return raw, existing_valid, data
