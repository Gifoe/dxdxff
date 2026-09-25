import mne
import numpy as np
import pandas as pd
import re

def normalize_channel_name(name: str) -> str:
    """
    Standardizes channel names to match across EDF and TSV.
    1. 转为大写
    2. 去掉首尾空格
    3. 去掉前缀 'EEG ', 'POL '
    4. 去掉后缀 '-REF', ' REF', '_REF'
    5. 删除中间多余空格
    6. 对末尾数字去前导零
    7. 示例： 'EEG LAD 01-Ref' -> 'LAD1'
    """
    if not isinstance(name, str):
        return str(name)
        
    s = name.upper().strip()
    
    # Remove prefixes
    if s.startswith("EEG "):
        s = s[4:].strip()
    elif s.startswith("POL "):
        s = s[4:].strip()
        
    # Remove suffixes
    for suffix in ["-REF", " REF", "_REF"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            
    # Remove all spaces inside
    s = s.replace(" ", "")
    
    # Remove leading zeros from trailing numbers (e.g., LAD01 -> LAD1, RAH003 -> RAH3)
    match = re.match(r'^([A-Z]+)0*(\d+)$', s)
    if match:
        s = f"{match.group(1)}{match.group(2)}"
        
    return s

def load_and_preprocess_edf(edf_path, channels_path, target_sfreq=250.0):
    channels_df = pd.read_csv(channels_path, sep='\t')
    channels_df['name_norm'] = channels_df['name'].apply(normalize_channel_name)
    channels_df['type_upper'] = channels_df['type'].astype(str).str.upper()
    
    valid_mask = channels_df['type_upper'].isin(['ECOG', 'SEEG', 'STEREOEEG']) & (channels_df['status'] != 'bad')
    valid_channels_norm = channels_df.loc[valid_mask, 'name_norm'].tolist()
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    
    original_to_norm = {}
    norm_to_original = {}
    
    for ch in raw.ch_names:
        norm_ch = normalize_channel_name(ch)
        original_to_norm[ch] = norm_ch
        
        if norm_ch in norm_to_original:
            print(f"WARNING: Duplicate normalized channel name {norm_ch} found for {ch} and {norm_to_original[norm_ch]}. Keeping first.")
        else:
            norm_to_original[norm_ch] = ch
            
    raw.rename_channels(original_to_norm)
    
    # Now raw.ch_names are normalized
    existing_valid_norm = [ch for ch in valid_channels_norm if ch in raw.ch_names]
    
    # Ensure unique in case of edge cases
    seen = set()
    unique_existing_valid_norm = []
    for x in existing_valid_norm:
        if x not in seen:
            unique_existing_valid_norm.append(x)
            seen.add(x)
            
    raw.pick(unique_existing_valid_norm)
    
    freqs = np.arange(60, raw.info['sfreq'] / 2, 60)
    if len(freqs) > 0:
        raw.notch_filter(freqs=freqs, verbose='ERROR')
        
    h_freq = min(150.0, raw.info['sfreq'] / 2 - 1.0)
    raw.filter(l_freq=1.0, h_freq=h_freq, verbose='ERROR')
    
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)
        
    data = raw.get_data().astype(np.float32)
    return raw, unique_existing_valid_norm, data, original_to_norm
