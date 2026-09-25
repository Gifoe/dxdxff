import numpy as np
import pandas as pd


def _safe_float(value):
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value_f):
        return None
    return value_f


def create_time_windows(raw, data, run_info, win_len_sec=2.0, step_sec=1.0):
    """
    Slice the continuous signal into fixed windows and assign phase labels.
    For short recordings, keep one full-length window so MAT runs are not dropped.
    """
    sfreq = raw.info["sfreq"]
    total_samples = raw.n_times

    windows = []

    subject_id = run_info.get("subject_id", "unknown")
    run_id = run_info["run_id"]
    task = run_info["task"]
    phase_group = run_info.get("phase_group", "unknown")
    onset = _safe_float(run_info.get("seizure_onset"))
    offset = _safe_float(run_info.get("seizure_offset"))

    win_len_samples = int(win_len_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    if win_len_samples <= 0 or step_samples <= 0:
        raise ValueError(
            f"Invalid window parameters: win_len_sec={win_len_sec}, step_sec={step_sec}, sfreq={sfreq}"
        )

    if total_samples <= 0:
        return pd.DataFrame(windows)

    if total_samples < win_len_samples:
        window_ranges = [(0, total_samples)]
    else:
        window_ranges = []
        start_idx = 0
        while start_idx + win_len_samples <= total_samples:
            window_ranges.append((start_idx, start_idx + win_len_samples))
            start_idx += step_samples

    for window_index, (start_idx, end_idx) in enumerate(window_ranges):
        start_t = start_idx / sfreq
        end_t = end_idx / sfreq

        phase = "interictal"
        task_lower = str(task).lower()
        if "ictal" in task_lower and "interictal" not in task_lower:
            if onset is not None and offset is not None:
                if end_t <= onset:
                    pre_offset = onset - end_t
                    phase = "preictal" if pre_offset <= 60 else "interictal"
                elif start_t >= onset and end_t <= offset:
                    phase = "ictal"
                elif start_t >= offset:
                    post_offset = start_t - offset
                    phase = "postictal" if post_offset <= 60 else "interictal"
                else:
                    phase = "transition"
            else:
                phase = "unknown_ictal"

        win_data = data[:, start_idx:end_idx]
        amplitude_range = np.max(win_data, axis=1) - np.min(win_data, axis=1)
        bad_channels = (amplitude_range > 2000e-6) | (amplitude_range == 0)
        unusable = bad_channels.mean() > 0.2

        windows.append(
            {
                "subject_id": subject_id,
                "run_id": run_id,
                "task": task,
                "phase_group": phase_group,
                "window_index": window_index,
                "start_sec": start_t,
                "end_sec": end_t,
                "phase": phase,
                "unusable_mask": unusable,
                "start_sample": start_idx,
                "end_sample": end_idx,
            }
        )

    return pd.DataFrame(windows)
