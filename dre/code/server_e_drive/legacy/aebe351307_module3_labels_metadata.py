import pandas as pd
import re

def parse_labels_and_metadata(channels_path):
    df = pd.read_csv(channels_path, sep='	')
    df['norm_name'] = df['name'].str.strip()
    
    meta_records = []
    for _, row in df.iterrows():
        ch_name = row['norm_name']
        ch_type = str(row['type']).upper()
        status = row.get('status', 'good')
        status_desc = str(row.get('status_description', '')).lower()
        
        is_soz = 1 if 'soz' in status_desc else 0
        is_resected = 1 if 'resect' in status_desc else 0
        is_ez = 1 if (is_soz or is_resected) else 0
        
        is_valid = 1 if status != 'bad' and ch_type in ['SEEG', 'ECOG', 'STEREOEEG'] else 0
        
        match = re.match(r'^([a-zA-Z]+)(\d+)$', ch_name)
        if match:
            stem = match.group(1)
            index = int(match.group(2))
        else:
            stem = ch_name
            index = None
            
        meta_records.append({
            'channel_name': ch_name,
            'stem': stem,
            'index': index,
            'is_soz': is_soz,
            'is_resected': is_resected,
            'is_ez': is_ez,
            'is_valid': is_valid,
            'type': ch_type
        })
        
    return pd.DataFrame(meta_records)
