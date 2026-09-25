from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import concurrent.futures

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError("networkx is required for the simple phase feature table pipeline.") from exc

try:
    from scipy.signal import hilbert, welch
except ImportError as exc:
    raise ImportError("scipy is required for the simple phase feature table pipeline.") from exc


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module1_file_discovery import discover_all_bids_files
from module2_preprocessing import load_and_preprocess_edf
from module3_labels_metadata import parse_channel_labels
from module4_time_windows import create_time_windows


PHASES: Tuple[str, ...] = ("ictal", "interictal")
GLOBAL_METRIC_NAMES: Tuple[str, ...] = ("GE", "GEdiff", "Q", "T", "kden")
NODE_METRIC_NAMES: Tuple[str, ...] = ("BC", "C", "CCe", "LE", "str")
BAND_DEFINITIONS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 80.0),
    ("ripple", 80.0, 150.0),
    ("fast_ripple", 150.0, 250.0),
)

EPS = 1e-8
DEFAULT_DATASET_DIR = Path(r"E:\DRE-nips\dataest")
DEFAULT_OUTPUT_DIR = CURRENT_FILE.parent / "outputs"

META_COLUMNS: Tuple[str, ...] = (
    "subject_id",
    "run_id",
    "task",
    "phase_group",
    "outcome",
    "channel_name_orig",
    "channel_name_norm",
    "channel_type",
    "contact_group",
    "contact_number",
    "status_description",
    "is_soz",
    "is_resected",
    "is_ez",
    "sampling_frequency",
    "target_sfreq",
)


def phase_feature_columns(phase: str) -> List[str]:
    columns = [f"{phase}_psd_{band_name}" for band_name, _, _ in BAND_DEFINITIONS]
    columns.extend(f"{phase}_{metric_name}" for metric_name in GLOBAL_METRIC_NAMES)
    columns.extend(f"{phase}_{metric_name}" for metric_name in NODE_METRIC_NAMES)
    return columns


def expected_output_columns() -> List[str]:
    columns = list(META_COLUMNS)
    for phase in PHASES:
        columns.append(f"{phase}_n_windows")
        columns.extend(phase_feature_columns(phase))
    return columns


def _resolve_participants_path(dataset_path: Path) -> Optional[Path]:
    candidates = (
        dataset_path / "participants.tsv",
        dataset_path / "dataset" / "participants.tsv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compute_band_powers(
    windows_data: np.ndarray,
    sfreq: float,
    band_definitions: Sequence[Tuple[str, float, float]] = BAND_DEFINITIONS,
) -> np.ndarray:
    if windows_data.ndim != 3:
        raise ValueError(f"windows_data must be 3D, got shape {windows_data.shape}")

    _, _, num_samples = windows_data.shape
    if num_samples < 4:
        raise ValueError("Each time window must contain at least 4 samples for PSD estimation.")

    nperseg = min(1024, int(num_samples))
    noverlap = max(0, nperseg // 2)
    freqs, psd = welch(
        windows_data,
        fs=float(sfreq),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=-1,
    )

    band_powers = np.full((*psd.shape[:-1], len(band_definitions)), np.nan, dtype=np.float64)
    for band_idx, (_, low_freq, high_freq) in enumerate(band_definitions):
        if band_idx == len(band_definitions) - 1:
            band_mask = (freqs >= float(low_freq)) & (freqs <= float(high_freq))
        else:
            band_mask = (freqs >= float(low_freq)) & (freqs < float(high_freq))

        if not np.any(band_mask):
            continue

        band_powers[..., band_idx] = np.trapz(psd[..., band_mask], freqs[band_mask], axis=-1)

    return band_powers


def compute_envelope_correlation(window_data: np.ndarray) -> np.ndarray:
    if window_data.ndim != 2:
        raise ValueError(f"window_data must be 2D, got shape {window_data.shape}")

    num_channels = window_data.shape[0]
    if num_channels == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if num_channels == 1:
        return np.zeros((1, 1), dtype=np.float64)

    analytic_signal = hilbert(window_data, axis=-1)
    envelopes = np.abs(analytic_signal)
    corr = np.corrcoef(envelopes)
    corr = np.asarray(corr, dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    return corr


def threshold_connectivity_matrix(weight_matrix: np.ndarray, edge_quantile: float) -> np.ndarray:
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError("weight_matrix must be square.")

    thresholded = np.array(weight_matrix, dtype=np.float64, copy=True)
    thresholded = np.nan_to_num(thresholded, nan=0.0, posinf=0.0, neginf=0.0)
    thresholded = np.maximum(thresholded, thresholded.T)
    np.fill_diagonal(thresholded, 0.0)

    upper_triangle = thresholded[np.triu_indices_from(thresholded, k=1)]
    positive_edges = upper_triangle[upper_triangle > 0]
    if positive_edges.size == 0:
        return np.zeros_like(thresholded)

    quantile = float(np.clip(edge_quantile, 0.0, 1.0))
    cutoff = np.quantile(positive_edges, quantile)
    keep_mask = (thresholded >= cutoff) & (thresholded > 0)

    if not np.any(np.triu(keep_mask, k=1)):
        max_weight = positive_edges.max()
        keep_mask = thresholded >= max_weight

    pruned = np.where(keep_mask, thresholded, 0.0)
    pruned = np.maximum(pruned, pruned.T)
    np.fill_diagonal(pruned, 0.0)
    return pruned


def build_graphs_from_adjacency(adjacency: np.ndarray) -> Tuple[nx.Graph, nx.Graph]:
    num_nodes = adjacency.shape[0]
    weighted_graph = nx.Graph()
    binary_graph = nx.Graph()
    weighted_graph.add_nodes_from(range(num_nodes))
    binary_graph.add_nodes_from(range(num_nodes))

    for src in range(num_nodes):
        for dst in range(src + 1, num_nodes):
            weight = float(adjacency[src, dst])
            if weight <= 0.0:
                continue
            distance = 1.0 / max(weight, EPS)
            weighted_graph.add_edge(src, dst, weight=weight, distance=distance)
            binary_graph.add_edge(src, dst)

    return weighted_graph, binary_graph


def weighted_global_efficiency(graph: nx.Graph) -> float:
    num_nodes = graph.number_of_nodes()
    if num_nodes <= 1:
        return 0.0

    efficiency_sum = 0.0
    for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight="distance"):
        for target, path_length in lengths.items():
            if source == target or path_length <= 0.0:
                continue
            efficiency_sum += 1.0 / path_length

    return efficiency_sum / float(num_nodes * (num_nodes - 1))


def global_diffusion_efficiency(adjacency: np.ndarray, binary_graph: nx.Graph) -> float:
    num_nodes = adjacency.shape[0]
    if num_nodes <= 1:
        return 0.0

    total_efficiency = 0.0
    for component in nx.connected_components(binary_graph):
        component_nodes = np.array(sorted(component), dtype=int)
        if component_nodes.size <= 1:
            continue

        sub_adjacency = adjacency[np.ix_(component_nodes, component_nodes)].astype(np.float64, copy=False)
        strengths = sub_adjacency.sum(axis=1)
        if np.any(strengths <= 0.0):
            continue

        transition = sub_adjacency / strengths[:, None]
        stationary = strengths / strengths.sum()
        stationary_tile = np.tile(stationary, (component_nodes.size, 1))
        identity = np.eye(component_nodes.size, dtype=np.float64)

        try:
            fundamental = np.linalg.inv(identity - transition + stationary_tile)
        except np.linalg.LinAlgError:
            fundamental = np.linalg.pinv(identity - transition + stationary_tile)

        diag_entries = np.diag(fundamental)[None, :]
        mfpt = (diag_entries - fundamental) / np.clip(stationary[None, :], EPS, None)
        np.fill_diagonal(mfpt, np.inf)

        component_efficiency = np.divide(
            1.0,
            mfpt,
            out=np.zeros_like(mfpt),
            where=np.isfinite(mfpt) & (mfpt > 0.0),
        )
        total_efficiency += component_efficiency.sum()

    return total_efficiency / float(num_nodes * (num_nodes - 1))


def compute_modularity(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0 or graph.number_of_nodes() <= 1:
        return 0.0

    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    if len(communities) <= 1:
        return 0.0

    return float(nx.algorithms.community.modularity(graph, communities, weight="weight"))


def compute_node_local_efficiencies(graph: nx.Graph) -> np.ndarray:
    local_efficiencies = np.zeros(graph.number_of_nodes(), dtype=np.float64)
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) <= 1:
            local_efficiencies[node] = 0.0
            continue

        subgraph = graph.subgraph(neighbors).copy()
        local_efficiencies[node] = weighted_global_efficiency(subgraph)
    return local_efficiencies


def compute_topology_metrics(weight_matrix: np.ndarray, edge_quantile: float) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    adjacency = threshold_connectivity_matrix(weight_matrix, edge_quantile=edge_quantile)
    num_nodes = adjacency.shape[0]

    global_metrics = {metric_name: 0.0 for metric_name in GLOBAL_METRIC_NAMES}
    node_metrics = {
        metric_name: np.zeros(num_nodes, dtype=np.float64)
        for metric_name in NODE_METRIC_NAMES
    }

    if num_nodes == 0:
        return global_metrics, node_metrics

    weighted_graph, binary_graph = build_graphs_from_adjacency(adjacency)

    global_metrics["GE"] = weighted_global_efficiency(weighted_graph)
    global_metrics["GEdiff"] = global_diffusion_efficiency(adjacency, binary_graph)
    global_metrics["Q"] = compute_modularity(weighted_graph)
    global_metrics["T"] = float(nx.transitivity(binary_graph)) if binary_graph.number_of_edges() > 0 else 0.0
    global_metrics["kden"] = float(nx.density(binary_graph)) if num_nodes > 1 else 0.0

    bc_map = nx.betweenness_centrality(weighted_graph, weight="distance", normalized=True)
    c_map = nx.clustering(weighted_graph, weight="weight")
    cce_map = nx.closeness_centrality(weighted_graph, distance="distance")
    le_values = compute_node_local_efficiencies(weighted_graph)
    strength_values = adjacency.sum(axis=1)

    for node in range(num_nodes):
        node_metrics["BC"][node] = float(bc_map.get(node, 0.0))
        node_metrics["C"][node] = float(c_map.get(node, 0.0))
        node_metrics["CCe"][node] = float(cce_map.get(node, 0.0))
        node_metrics["LE"][node] = float(le_values[node])
        node_metrics["str"][node] = float(strength_values[node])

    return global_metrics, node_metrics


def summarize_phase_features(
    windows_data: np.ndarray,
    sfreq: float,
    edge_quantile: float,
) -> Optional[Dict[str, Any]]:
    if windows_data.size == 0 or windows_data.shape[0] == 0:
        return None

    band_powers = compute_band_powers(windows_data, sfreq=sfreq)

    global_rows: List[List[float]] = []
    node_rows = {metric_name: [] for metric_name in NODE_METRIC_NAMES}

    for window_data in windows_data:
        weight_matrix = compute_envelope_correlation(window_data)
        global_metrics, node_metrics = compute_topology_metrics(weight_matrix, edge_quantile=edge_quantile)
        global_rows.append([global_metrics[metric_name] for metric_name in GLOBAL_METRIC_NAMES])
        for metric_name in NODE_METRIC_NAMES:
            node_rows[metric_name].append(node_metrics[metric_name])

    global_array = np.asarray(global_rows, dtype=np.float64)
    global_mean = {
        metric_name: float(global_array[:, metric_idx].mean())
        for metric_idx, metric_name in enumerate(GLOBAL_METRIC_NAMES)
    }
    node_mean = {
        metric_name: np.stack(metric_values, axis=0).mean(axis=0)
        for metric_name, metric_values in node_rows.items()
    }

    return {
        "n_windows": int(windows_data.shape[0]),
        "band_powers": band_powers.mean(axis=0),
        "global_metrics": global_mean,
        "node_metrics": node_mean,
    }


def build_empty_phase_summary(num_channels: int) -> Dict[str, Any]:
    return {
        "n_windows": 0,
        "band_powers": np.full((num_channels, len(BAND_DEFINITIONS)), np.nan, dtype=np.float64),
        "global_metrics": {metric_name: float("nan") for metric_name in GLOBAL_METRIC_NAMES},
        "node_metrics": {
            metric_name: np.full(num_channels, np.nan, dtype=np.float64)
            for metric_name in NODE_METRIC_NAMES
        },
    }


def align_contacts_meta(
    channels_path: str,
    picked_channels_norm: Sequence[str],
    ez_definition: str,
) -> pd.DataFrame:
    contacts_meta = parse_channel_labels(channels_path, ez_definition=ez_definition)
    contacts_meta = contacts_meta[contacts_meta["is_valid"] == 1].copy()
    if contacts_meta.empty:
        return contacts_meta

    channel_order = {channel_name: idx for idx, channel_name in enumerate(picked_channels_norm)}
    contacts_meta = contacts_meta[contacts_meta["channel_name_norm"].isin(channel_order)].copy()
    if contacts_meta.empty:
        return contacts_meta

    contacts_meta["picked_order"] = contacts_meta["channel_name_norm"].map(channel_order)
    contacts_meta = contacts_meta.sort_values("picked_order").reset_index(drop=True)
    return contacts_meta


def collect_phase_windows(
    data: np.ndarray,
    windows_df: pd.DataFrame,
    phase_name: str,
) -> np.ndarray:
    if windows_df.empty:
        return np.zeros((0, data.shape[0], 0), dtype=np.float64)

    phase_mask = windows_df["phase"].astype(str).str.lower() == phase_name
    phase_rows = windows_df.loc[phase_mask].reset_index(drop=True)
    if phase_rows.empty:
        return np.zeros((0, data.shape[0], 0), dtype=np.float64)

    phase_windows = np.stack(
        [
            data[:, int(row.start_sample): int(row.end_sample)]
            for row in phase_rows.itertuples(index=False)
        ],
        axis=0,
    ).astype(np.float64, copy=False)
    return phase_windows


def extract_run_rows(
    run_info: Dict[str, Any],
    *,
    target_sfreq: float,
    win_len_sec: float,
    step_sec: float,
    edge_quantile: float,
    ez_definition: str,
    bandpass_low: float,
    bandpass_high: float,
) -> List[Dict[str, Any]]:
    edf_path = run_info.get("edf_path")
    channels_path = run_info.get("channels_path")
    if not edf_path or not channels_path:
        raise ValueError("Both edf_path and channels_path are required.")

    raw = None
    try:
        raw, picked_channels_norm, data, _ = load_and_preprocess_edf(
            edf_path,
            channels_path,
            target_sfreq=target_sfreq,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
        )

        contacts_meta = align_contacts_meta(
            channels_path=channels_path,
            picked_channels_norm=picked_channels_norm,
            ez_definition=ez_definition,
        )
        if contacts_meta.empty:
            raise ValueError("No valid channels remained after alignment with the EDF file.")

        reorder_idx = contacts_meta["picked_order"].to_numpy(dtype=int)
        data = data[reorder_idx]

        windows_df = create_time_windows(
            raw,
            data,
            run_info,
            win_len_sec=win_len_sec,
            step_sec=step_sec,
        )
        usable_windows = windows_df.loc[~windows_df["unusable_mask"]].reset_index(drop=True)

        num_channels = len(contacts_meta)
        phase_summaries: Dict[str, Dict[str, Any]] = {}
        for phase_name in PHASES:
            phase_windows = collect_phase_windows(data, usable_windows, phase_name=phase_name)
            if phase_windows.shape[0] == 0:
                phase_summaries[phase_name] = build_empty_phase_summary(num_channels)
                continue

            phase_summaries[phase_name] = summarize_phase_features(
                phase_windows,
                sfreq=float(raw.info["sfreq"]),
                edge_quantile=edge_quantile,
            )

        rows: List[Dict[str, Any]] = []
        for channel_idx, meta_row in contacts_meta.iterrows():
            row = {
                "subject_id": run_info.get("subject_id"),
                "run_id": run_info.get("run_id"),
                "task": run_info.get("task"),
                "phase_group": run_info.get("phase_group"),
                "outcome": run_info.get("outcome"),
                "channel_name_orig": meta_row.get("channel_name_orig"),
                "channel_name_norm": meta_row.get("channel_name_norm"),
                "channel_type": meta_row.get("type"),
                "contact_group": meta_row.get("contact_group"),
                "contact_number": meta_row.get("contact_number"),
                "status_description": meta_row.get("status_description"),
                "is_soz": int(meta_row.get("is_soz", 0)),
                "is_resected": int(meta_row.get("is_resected", 0)),
                "is_ez": int(meta_row.get("is_ez", 0)),
                "sampling_frequency": float(raw.info["sfreq"]),
                "target_sfreq": float(target_sfreq),
            }

            for phase_name in PHASES:
                summary = phase_summaries[phase_name]
                row[f"{phase_name}_n_windows"] = int(summary["n_windows"])

                for band_idx, (band_name, _, _) in enumerate(BAND_DEFINITIONS):
                    row[f"{phase_name}_psd_{band_name}"] = _safe_float(
                        summary["band_powers"][channel_idx, band_idx]
                    )

                for metric_name in GLOBAL_METRIC_NAMES:
                    row[f"{phase_name}_{metric_name}"] = _safe_float(
                        summary["global_metrics"][metric_name]
                    )

                for metric_name in NODE_METRIC_NAMES:
                    row[f"{phase_name}_{metric_name}"] = _safe_float(
                        summary["node_metrics"][metric_name][channel_idx]
                    )

            rows.append(row)

        return rows
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()


def rows_to_dataframe(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    columns = expected_output_columns()
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df[columns]
    df = df.sort_values(["subject_id", "run_id", "channel_name_norm"]).reset_index(drop=True)
    return df


def errors_to_dataframe(errors: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    columns = ["subject_id", "run_id", "task", "edf_path", "channels_path", "error"]
    if not errors:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(errors, columns=columns)


def process_single_run(
    run_idx: int,
    run_info_dict: Dict[str, Any],
    target_sfreq: float,
    win_len_sec: float,
    step_sec: float,
    edge_quantile: float,
    ez_definition: str,
    bandpass_low: float,
    bandpass_high: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_id = run_info_dict.get("run_id", f"run_{run_idx}")
    print(f"[{run_idx}] extracting simple features for {run_id}")
    
    all_rows = []
    errors = []
    try:
        run_rows = extract_run_rows(
            run_info_dict,
            target_sfreq=target_sfreq,
            win_len_sec=win_len_sec,
            step_sec=step_sec,
            edge_quantile=edge_quantile,
            ez_definition=ez_definition,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
        )
        all_rows.extend(run_rows)
    except Exception as exc:
        errors.append(
            {
                "subject_id": run_info_dict.get("subject_id"),
                "run_id": run_info_dict.get("run_id"),
                "task": run_info_dict.get("task"),
                "edf_path": run_info_dict.get("edf_path"),
                "channels_path": run_info_dict.get("channels_path"),
                "error": str(exc),
            }
        )
    return all_rows, errors


def build_feature_table(
    runs_df: pd.DataFrame,
    *,
    target_sfreq: float,
    win_len_sec: float,
    step_sec: float,
    edge_quantile: float,
    ez_definition: str,
    bandpass_low: float,
    bandpass_high: float,
    max_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    run_tasks = []
    for run_idx, (_, run_row) in enumerate(runs_df.iterrows(), start=1):
        run_info = run_row.to_dict()
        run_tasks.append((
            run_idx,
            run_info,
            target_sfreq,
            win_len_sec,
            step_sec,
            edge_quantile,
            ez_definition,
            bandpass_low,
            bandpass_high
        ))

    print(f"Starting feature extraction for {len(run_tasks)} runs using {max_workers} processes...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_run, *task_args): task_args[0]
            for task_args in run_tasks
        }
        for future in concurrent.futures.as_completed(futures):
            res_rows, res_errors = future.result()
            all_rows.extend(res_rows)
            errors.extend(res_errors)

    return rows_to_dataframe(all_rows), errors_to_dataframe(errors)


def write_manifest(
    output_dir: Path,
    *,
    dataset_dir: Path,
    participants_path: Optional[Path],
    target_sfreq: float,
    win_len_sec: float,
    step_sec: float,
    edge_quantile: float,
    ez_definition: str,
    success_only: bool,
    subject_filter: Optional[str],
) -> None:
    manifest = {
        "dataset_dir": str(dataset_dir),
        "participants_path": str(participants_path) if participants_path is not None else None,
        "feature_layout": {
            "row_unit": "one row per subject_id + run_id + channel_name_norm",
            "phases": list(PHASES),
            "per_phase_feature_count": len(BAND_DEFINITIONS) + len(GLOBAL_METRIC_NAMES) + len(NODE_METRIC_NAMES),
            "psd_band_count": len(BAND_DEFINITIONS),
            "global_topology_feature_count": len(GLOBAL_METRIC_NAMES),
            "node_topology_feature_count": len(NODE_METRIC_NAMES),
        },
        "band_definitions_hz": [
            {"name": band_name, "low_hz": low_hz, "high_hz": high_hz}
            for band_name, low_hz, high_hz in BAND_DEFINITIONS
        ],
        "topology_global_metrics": list(GLOBAL_METRIC_NAMES),
        "topology_node_metrics": list(NODE_METRIC_NAMES),
        "assumptions": {
            "ictal_and_interictal_are_extracted_separately": True,
            "preictal_postictal_transition_are_excluded": True,
            "global_metrics_are_repeated_on_each_channel_row_within_the_same_run_phase": True,
            "functional_connectivity": "absolute amplitude-envelope correlation",
            "graph_sparsification": f"keep edges at or above the upper-{(1.0 - edge_quantile) * 100.0:.1f}% tail within each window",
        },
        "runtime": {
            "target_sfreq": float(target_sfreq),
            "win_len_sec": float(win_len_sec),
            "step_sec": float(step_sec),
            "edge_quantile": float(edge_quantile),
            "ez_definition": ez_definition,
            "success_only": bool(success_only),
            "subject_filter": subject_filter,
        },
    }

    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple per-run per-channel wide feature table with ictal/interictal PSD and topology features.",
    )
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subject-filter", type=str, default=None)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--target-sfreq", type=float, default=512.0)
    parser.add_argument("--win-len-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--edge-quantile", type=float, default=0.80)
    parser.add_argument("--ez-definition", type=str, default="soz_or_resected")
    parser.add_argument("--bandpass-low", type=float, default=1.0)
    parser.add_argument("--bandpass-high", type=float, default=250.0)
    parser.add_argument("--max-workers", type=int, default=4, help="Number of CPU cores to use for processing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants_path = _resolve_participants_path(dataset_dir)
    runs_df = discover_all_bids_files(
        dataset_dir,
        participants_path=str(participants_path) if participants_path is not None else None,
        subject_filter=args.subject_filter,
        success_only=bool(args.success_only),
    )
    if runs_df.empty:
        raise FileNotFoundError(f"No runs were discovered under {dataset_dir}.")

    features_df, errors_df = build_feature_table(
        runs_df,
        target_sfreq=float(args.target_sfreq),
        win_len_sec=float(args.win_len_sec),
        step_sec=float(args.step_sec),
        edge_quantile=float(args.edge_quantile),
        ez_definition=str(args.ez_definition),
        bandpass_low=float(args.bandpass_low),
        bandpass_high=float(args.bandpass_high),
        max_workers=int(args.max_workers),
    )

    output_csv = output_dir / "run_channel_phase_features_wide.csv"
    features_df.to_csv(output_csv, index=False)

    errors_csv = output_dir / "run_feature_extraction_errors.csv"
    errors_df.to_csv(errors_csv, index=False)

    write_manifest(
        output_dir,
        dataset_dir=dataset_dir,
        participants_path=participants_path,
        target_sfreq=float(args.target_sfreq),
        win_len_sec=float(args.win_len_sec),
        step_sec=float(args.step_sec),
        edge_quantile=float(args.edge_quantile),
        ez_definition=str(args.ez_definition),
        success_only=bool(args.success_only),
        subject_filter=args.subject_filter,
    )

    print(f"Saved feature table to {output_csv}")
    print(f"Saved run-level errors to {errors_csv}")


if __name__ == "__main__":
    main()
