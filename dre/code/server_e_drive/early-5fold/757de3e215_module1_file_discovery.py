import os
from pathlib import Path
import pandas as pd
import json
import re

def load_participants_table(participants_path):
    if not os.path.exists(participants_path):
        print(f"Warning: participants.tsv not found at {participants_path}")
        return pd.DataFrame()

    df = pd.read_csv(participants_path, sep='\t')
    if 'participant_id' in df.columns:
        df['participant_id'] = df['participant_id'].astype(str).str.strip()

    if 'outcome' in df.columns:
        df['outcome'] = df['outcome'].astype(str).str.upper().str.strip()
        def map_outcome(val):
            if val == 'S': return 1.0
            elif val == 'F': return 0.0
            else: return float('nan')
        df['outcome_binary'] = df['outcome'].apply(map_outcome)
    else:
        df['outcome'] = 'UNKNOWN'
        df['outcome_binary'] = float('nan')

    expected_cols = ['engel', 'therapy', 'implant', 'target', 'lesion_status', 'age', 'sex', 'hand', 'age_onset']
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 'N/A'

    return df

def discover_all_bids_files(root_dir, participants_path=None, subject_filter=None):
    records = []
    root_path = Path(root_dir)

    if subject_filter:
        subject_dirs = [root_path / subject_filter]
    else:
        subject_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("sub-HUP")]

    for subject_folder in subject_dirs:
        if not subject_folder.exists():
            continue

        subject_id = subject_folder.name
        edf_files = list(subject_folder.rglob(f"*{subject_id}*.edf"))

        for edf_path in edf_files:
            stem = edf_path.name.replace('_ieeg.edf', '').replace('.edf', '')
            parent_dir = edf_path.parent

            channels_file = parent_dir / f"{stem}_channels.tsv"
            events_file = parent_dir / f"{stem}_events.tsv"
            json_file = parent_dir / f"{stem}_ieeg.json"

            task_match = re.search(r'task-([a-zA-Z0-9]+)', stem)
            task = task_match.group(1) if task_match else 'unknown'

            run_match = re.search(r'run-([0-9]+)', stem)
            run_num = run_match.group(1) if run_match else '01'
            run_id = f"{subject_id}__{task}__run-{run_num}"

            phase_group = 'interictal' if 'interictal' in task.lower() else 'ictal' if 'ictal' in task.lower() else task

            sfreq, line_freq, duration = None, None, None
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        sfreq = meta.get('SamplingFrequency')
                        line_freq = meta.get('PowerLineFrequency')
                        duration = meta.get('RecordingDuration')
                except Exception: pass

            onset, offset = None, None
            if events_file.exists():
                try:
                    events_df = pd.read_csv(events_file, sep='\t')
                    onset_row = events_df[events_df['trial_type'] == 'sz onset']
                    offset_row = events_df[events_df['trial_type'] == 'sz offset']
                    if not onset_row.empty: onset = onset_row.iloc[0]['onset']
                    if not offset_row.empty: offset = offset_row.iloc[0]['onset']
                except Exception: pass

            records.append({
                'subject_id': subject_id,
                'run_id': run_id,
                'task': task,
                'phase_group': phase_group,
                'edf_path': str(edf_path),
                'channels_path': str(channels_file) if channels_file.exists() else None,
                'events_path': str(events_file) if events_file.exists() else None,
                'json_path': str(json_file) if json_file.exists() else None,
                'sampling_frequency': sfreq,
                'line_frequency': line_freq,
                'duration': duration,
                'seizure_onset': onset,
                'seizure_offset': offset
            })

    runs_df = pd.DataFrame(records)
    if runs_df.empty: return runs_df

    if participants_path:
        pd.options.mode.chained_assignment = None
        clinical_df = load_participants_table(participants_path)
        if not clinical_df.empty:
            runs_df = pd.merge(runs_df, clinical_df, left_on='subject_id', right_on='participant_id', how='left')

    expected_meta = ['outcome', 'outcome_binary', 'engel', 'therapy', 'implant', 'target', 'lesion_status', 'age', 'sex', 'hand', 'age_onset']
    for c in expected_meta:
        if c not in runs_df.columns:
            runs_df[c] = float('nan') if c == 'outcome_binary' else 'UNKNOWN' if c == 'outcome' else 'N/A'

    runs_df['outcome'] = runs_df['outcome'].fillna('UNKNOWN')
    for c in ['engel', 'therapy', 'implant', 'target', 'lesion_status']:
        runs_df[c] = runs_df[c].fillna('N/A')

    runs_df = runs_df.sort_values(['subject_id', 'task', 'run_id']).reset_index(drop=True)
    return runs_df
