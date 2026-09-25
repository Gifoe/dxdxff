import numpy as np
import pandas as pd

def create_time_windows(raw, run_info, win_len_sec=2.0, step_sec=1.0):
    """
    Slices the continuous signal into windows and assigns phase labels. 
    Maintains subject and run metadata to support LOPOCV.
    """
    sfreq = raw.info['sfreq']
    total_samples = raw.n_times
    total_sec = total_samples / sfreq
    
    windows = []
    
    # Metadata extraction
    subject_id = run_info.get('subject_id', 'unknown')
    run_id = run_info['run_id']
    task = run_info['task']
    phase_group = run_info.get('phase_group', 'unknown')
    onset = run_info['seizure_onset']
    offset = run_info['seizure_offset']
    
    win_len_samples = int(win_len_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    
    start_idx = 0
    window_index = 0
    
    data = raw.get_data() # (n_chan, n_times)
    
    while start_idx + win_len_samples <= total_samples:
        end_idx = start_idx + win_len_samples
        start_t = start_idx / sfreq
        end_t = end_idx / sfreq
        
        # Determine internal phase
        phase = 'interictal' # Default
        if 'ictal' in task.lower():
            if onset is not None and offset is not None:
                if end_t <= onset:
                    pre_offset = onset - end_t
                    if pre_offset <= 60:
                        phase = 'preictal'
                    else:
                        phase = 'interictal' # or far-preictal
                elif start_t >= onset and end_t <= offset:
                    phase = 'ictal'
                elif start_t >= offset:
                    post_offset = start_t - offset
                    if post_offset <= 60:
                        phase = 'postictal'
                    else:
                        phase = 'interictal' # or far-postictal
                else:
                    phase = 'transition'
            else:
                phase = 'unknown_ictal'
                
        # Simple artifact mask logic
        win_data = data[:, start_idx:end_idx]
        # Example criteria: reject if max amplitude across any channel is > 10mV or all zeros
        amplitude_range = np.max(win_data, axis=1) - np.min(win_data, axis=1)
        unusable = False
        if np.any(amplitude_range > 10000e-6) or np.any(amplitude_range == 0):
            unusable = True
            
        windows.append({
            'subject_id': subject_id,
            'run_id': run_id,
            'task': task,
            'phase_group': phase_group,
            'window_index': window_index,
            'start_sec': start_t,
            'end_sec': end_t,
            'phase': phase,
            'unusable_mask': unusable,
            'start_sample': start_idx,
            'end_sample': end_idx
        })
        
        start_idx += step_samples
        window_index += 1
        
    return pd.DataFrame(windows)
