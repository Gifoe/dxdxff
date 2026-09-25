from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SPECTRAL_MIN_FREQ = 1.0
DEFAULT_SPECTRAL_MAX_FREQ = 250.0
DEFAULT_HIGH_FREQ_BAND_WEIGHTS = (
    (80.0, 150.0, 1.5),
    (150.0, 250.0, 2.0),
)

def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preferred_str = str(preferred).strip().lower()
    if preferred_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred_str)


def configure_runtime(preferred: Optional[str] = None) -> torch.device:
    device = resolve_device(preferred)
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    return device


def _ensure_float_tensor(array_like: Any, device: torch.device) -> torch.Tensor:
    if isinstance(array_like, torch.Tensor):
        return array_like.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array_like, dtype=torch.float32, device=device)


def extract_spectral_features_torch(
    data_t: torch.Tensor,
    sfreq: float,
    min_freq: float = DEFAULT_SPECTRAL_MIN_FREQ,
    max_freq: float = DEFAULT_SPECTRAL_MAX_FREQ,
    spectral_band_weights = DEFAULT_HIGH_FREQ_BAND_WEIGHTS,
) -> torch.Tensor:
    data_t = _ensure_float_tensor(data_t, data_t.device)
    _, _, num_samples = data_t.shape

    fft_vals = torch.fft.rfft(data_t, dim=-1)
    power = torch.abs(fft_vals).pow(2) / max(num_samples, 1)
    freqs = torch.fft.rfftfreq(num_samples, d=1.0 / float(sfreq)).to(data_t.device)

    effective_max_freq = min(float(max_freq), float(sfreq) / 2.0)
    freq_mask = (freqs >= float(min_freq)) & (freqs <= effective_max_freq)
    if not torch.any(freq_mask):
        raise ValueError(
            f"No spectral bins available between {min_freq} and {effective_max_freq} Hz at sfreq={sfreq}."
        )
    psd_features = torch.log1p(power[..., freq_mask])
    selected_freqs = freqs[freq_mask]

    if spectral_band_weights:
        freq_weights = torch.ones_like(selected_freqs, dtype=psd_features.dtype, device=psd_features.device)
        for band_low, band_high, weight in spectral_band_weights:
            upper_bound_inclusive = float(band_high) >= effective_max_freq
            if upper_bound_inclusive:
                band_mask = (selected_freqs >= float(band_low)) & (selected_freqs <= float(band_high))
            else:
                band_mask = (selected_freqs >= float(band_low)) & (selected_freqs < float(band_high))
            freq_weights = torch.where(
                band_mask,
                torch.full_like(freq_weights, float(weight)),
                freq_weights,
            )
        psd_features = psd_features * freq_weights.view(*([1] * (psd_features.dim() - 1)), -1)

    rms = torch.sqrt(torch.mean(data_t.pow(2), dim=-1, keepdim=True) + 1e-8)
    variance = torch.var(data_t, dim=-1, unbiased=False, keepdim=True)
    line_length = torch.sum(torch.abs(torch.diff(data_t, dim=-1)), dim=-1, keepdim=True)

    return torch.cat([psd_features, rms, variance, line_length], dim=-1)


def compute_envelope_correlation_batched(data_t: Any) -> torch.Tensor:
    if not isinstance(data_t, torch.Tensor):
        data_t = torch.as_tensor(data_t, dtype=torch.float32)

    data_t = data_t.to(dtype=torch.float32)
    batch_size, num_channels, num_samples = data_t.shape

    if num_channels == 0:
        return data_t.new_zeros((batch_size, 0, 0))

    fft_vals = torch.fft.fft(data_t, dim=-1)
    hilbert_kernel = torch.zeros(num_samples, dtype=torch.float32, device=data_t.device)
    if num_samples % 2 == 0:
        hilbert_kernel[0] = 1.0
        hilbert_kernel[num_samples // 2] = 1.0
        hilbert_kernel[1 : num_samples // 2] = 2.0
    else:
        hilbert_kernel[0] = 1.0
        hilbert_kernel[1 : (num_samples + 1) // 2] = 2.0

    analytic_signal = torch.fft.ifft(fft_vals * hilbert_kernel, dim=-1)
    envelope = torch.abs(analytic_signal)

    centered = envelope - envelope.mean(dim=-1, keepdim=True)
    cov = torch.bmm(centered, centered.transpose(1, 2))
    scale = torch.sqrt(torch.sum(centered.pow(2), dim=-1, keepdim=True)).clamp_min(1e-8)
    corr = cov / torch.bmm(scale, scale.transpose(1, 2)).clamp_min(1e-8)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    eye = torch.eye(num_channels, dtype=torch.bool, device=data_t.device).unsqueeze(0)
    corr = corr.masked_fill(eye, 1.0)
    return corr


def extract_connectivity_node_features_batch(env_corrs: Any) -> torch.Tensor:
    if not isinstance(env_corrs, torch.Tensor):
        env_corrs = torch.as_tensor(env_corrs, dtype=torch.float32)

    env_corrs = env_corrs.to(dtype=torch.float32)
    _, num_channels, _ = env_corrs.shape

    if num_channels == 0:
        return env_corrs.new_zeros((*env_corrs.shape[:2], 3))

    eye = torch.eye(num_channels, dtype=torch.bool, device=env_corrs.device).unsqueeze(0)
    corr_wo_self = env_corrs.masked_fill(eye, 0.0)
    denom = max(num_channels - 1, 1)

    mean_abs = corr_wo_self.abs().sum(dim=-1, keepdim=True) / float(denom)
    max_abs = corr_wo_self.abs().amax(dim=-1, keepdim=True)
    var_conn = torch.var(corr_wo_self, dim=-1, unbiased=False, keepdim=True)
    return torch.cat([mean_abs, max_abs, var_conn], dim=-1)


def build_dynamic_graphs_batched(
    windows_data: Any,
    contacts_meta: pd.DataFrame,
    top_k: int = 5,
    device: Optional[str] = None,
):
    device_obj = configure_runtime(device)
    data_t = _ensure_float_tensor(windows_data, device_obj)
    batch_size, num_channels, _ = data_t.shape

    if len(contacts_meta) != num_channels:
        raise ValueError(
            f"contacts_meta length ({len(contacts_meta)}) does not match channel count ({num_channels})."
        )

    env_corrs = compute_envelope_correlation_batched(data_t)
    if env_corrs.device != device_obj:
        env_corrs = env_corrs.to(device_obj)

    groups = contacts_meta["contact_group"].fillna("").astype(str).to_numpy()
    numbers = pd.to_numeric(contacts_meta["contact_number"], errors="coerce").to_numpy(dtype=np.float32)

    same_group_mat = groups[:, None] == groups[None, :]
    dist_mat = np.abs(numbers[:, None] - numbers[None, :])
    dist_mat[~np.isfinite(dist_mat)] = np.inf
    dist_mat[~same_group_mat] = np.inf
    topo_adj_mat = same_group_mat & np.isfinite(dist_mat) & (dist_mat == 1.0)

    same_group_t = torch.as_tensor(same_group_mat.astype(np.float32), device=device_obj)
    dist_safe = np.where(np.isfinite(dist_mat), dist_mat, 10.0).astype(np.float32)
    dist_t = torch.as_tensor(dist_safe, device=device_obj)
    topo_src, topo_dst = np.where(topo_adj_mat)
    topo_src_t = torch.as_tensor(topo_src, dtype=torch.long, device=device_obj)
    topo_dst_t = torch.as_tensor(topo_dst, dtype=torch.long, device=device_obj)

    eye = torch.eye(num_channels, dtype=torch.bool, device=device_obj).unsqueeze(0)
    abs_corr = env_corrs.abs().masked_fill(eye, -torch.inf)
    effective_top_k = min(max(int(top_k), 0), max(num_channels - 1, 0))
    if effective_top_k > 0:
        _, topk_indices = torch.topk(abs_corr, k=effective_top_k, dim=-1)
    else:
        topk_indices = None

    batched_edge_indices = []
    batched_edge_attrs = []
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device_obj)
    empty_edge_attr = torch.empty((0, 4), dtype=torch.float32, device=device_obj)

    for batch_idx in range(batch_size):
        edge_index_parts = []
        edge_attr_parts = []

        if effective_top_k > 0:
            src = torch.arange(num_channels, device=device_obj).repeat_interleave(effective_top_k)
            dst = topk_indices[batch_idx].reshape(-1)
            corr_weight = env_corrs[batch_idx, src, dst]
            is_struct = torch.zeros_like(corr_weight)
            same_group = same_group_t[src, dst]
            distance = dist_t[src, dst].clamp(0.0, 10.0)

            edge_index_parts.append(torch.stack([src, dst], dim=0))
            edge_attr_parts.append(torch.stack([corr_weight, is_struct, same_group, distance], dim=-1))

        if topo_src_t.numel() > 0:
            topo_weight = env_corrs[batch_idx, topo_src_t, topo_dst_t]
            topo_type = torch.ones_like(topo_weight)
            topo_same_group = same_group_t[topo_src_t, topo_dst_t]
            topo_distance = dist_t[topo_src_t, topo_dst_t].clamp(0.0, 10.0)

            edge_index_parts.append(torch.stack([topo_src_t, topo_dst_t], dim=0))
            edge_attr_parts.append(
                torch.stack([topo_weight, topo_type, topo_same_group, topo_distance], dim=-1)
            )

        if edge_index_parts:
            final_edge_index = torch.cat(edge_index_parts, dim=1)
            final_edge_attr = torch.cat(edge_attr_parts, dim=0)
        else:
            final_edge_index = empty_edge_index
            final_edge_attr = empty_edge_attr

        batched_edge_indices.append(final_edge_index.cpu())
        batched_edge_attrs.append(final_edge_attr.cpu())

    return env_corrs, batched_edge_indices, batched_edge_attrs


def process_batched_window_features(
    windows_data: Any,
    sfreq: float,
    env_corrs: Optional[Any] = None,
    device: Optional[str] = None,
    min_freq: float = DEFAULT_SPECTRAL_MIN_FREQ,
    max_freq: float = DEFAULT_SPECTRAL_MAX_FREQ,
    spectral_band_weights = DEFAULT_HIGH_FREQ_BAND_WEIGHTS,
) -> Dict[str, torch.Tensor]:
    device_obj = configure_runtime(device)
    data_t = _ensure_float_tensor(windows_data, device_obj)

    if env_corrs is None:
        env_corrs_t = compute_envelope_correlation_batched(data_t)
    else:
        env_corrs_t = _ensure_float_tensor(env_corrs, device_obj)

    node_spec = extract_spectral_features_torch(
        data_t,
        sfreq,
        min_freq=min_freq,
        max_freq=max_freq,
        spectral_band_weights=spectral_band_weights,
    )
    node_conn = extract_connectivity_node_features_batch(env_corrs_t)

    return {"node_spec": node_spec, "node_conn": node_conn}


class LocalSpectralEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps, num_channels, feat_dim = x.shape
        out = self.net(x.reshape(time_steps * num_channels, feat_dim))
        return out.reshape(time_steps, num_channels, -1)


class EdgeAwareMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.3):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _forward_single(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x

        src = edge_index[0].long().to(x.device)
        dst = edge_index[1].long().to(x.device)
        edge_features = self.edge_proj(edge_attr.to(device=x.device, dtype=torch.float32))

        messages = x[src] + edge_features
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, dst, messages)

        degree = x.new_zeros((x.size(0), 1))
        degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=x.device, dtype=x.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)

        updated = self.update(torch.cat([x, aggregated], dim=-1))
        return self.norm(x + updated)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self._forward_single(x, edge_index, edge_attr)

        if edge_index.numel() == 0:
            return x

        src = edge_index[:, 0, :].long().to(x.device)
        dst = edge_index[:, 1, :].long().to(x.device)
        edge_features = self.edge_proj(edge_attr.to(device=x.device, dtype=torch.float32))

        hidden_dim = x.size(-1)
        src_feat = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, hidden_dim))
        messages = src_feat + edge_features

        aggregated = x.new_zeros(x.shape)
        aggregated.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, hidden_dim), messages)

        degree = x.new_zeros((x.size(0), x.size(1), 1))
        ones = x.new_ones((dst.size(0), dst.size(1), 1))
        degree.scatter_add_(1, dst.unsqueeze(-1), ones)

        aggregated = aggregated / degree.clamp_min(1.0)
        updated = self.update(torch.cat([x, aggregated], dim=-1))
        return self.norm(x + updated)


class GraphSpatialEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Identity() if node_dim == hidden_dim else nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeAwareMessagePassing(hidden_dim, edge_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor, edge_attrs: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_indices, edge_attrs)
        return h


class TwoBranchDynamicModel(nn.Module):
    def __init__(
        self,
        spec_in_dim: int,
        conn_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_phase_types: int = 6,
    ):
        super().__init__()
        self.spectral_enc = LocalSpectralEncoder(spec_in_dim, hidden_dim, dropout=dropout)
        self.conn_proj = nn.Sequential(
            nn.Linear(conn_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.graph_enc = GraphSpatialEncoder(hidden_dim, edge_in_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.fusion_ln = nn.LayerNorm(hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.temp_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.temp_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ez_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.count_ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_spec: torch.Tensor,
        node_conn: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_attrs: torch.Tensor,
        return_embeddings: bool = False,
    ):
        if node_spec.dim() != 3 or node_conn.dim() != 3:
            raise ValueError("node_spec and node_conn must have shape (time, channels, features).")

        x_spec = self.spectral_enc(node_spec)
        x_conn = self.conn_proj(node_conn)
        x_graph = self.graph_enc(x_conn, edge_indices, edge_attrs)

        fused_out, _ = self.cross_attn(x_graph, x_spec, x_spec)
        fused = self.fusion_ln(fused_out + x_graph)
        
        time_steps, num_channels, _ = fused.shape

        seq_input = fused.transpose(0, 1).contiguous()
        gru_out, _ = self.temporal_gru(seq_input)
        attn_weights = F.softmax(self.temp_attn(gru_out), dim=1)
        channel_embeddings = torch.sum(gru_out * attn_weights, dim=1)
        channel_embeddings = self.temp_proj(channel_embeddings)
        channel_logits = self.ez_head(channel_embeddings).squeeze(-1)
        
        run_embedding = channel_embeddings.mean(dim=0)
        predicted_ratio_run = torch.sigmoid(self.count_ratio_head(run_embedding))

        if return_embeddings:
            return channel_logits, channel_embeddings, predicted_ratio_run
        return channel_logits, predicted_ratio_run


__all__ = [
    "DEFAULT_HIGH_FREQ_BAND_WEIGHTS",
    "DEFAULT_SPECTRAL_MAX_FREQ",
    "DEFAULT_SPECTRAL_MIN_FREQ",
    "GraphSpatialEncoder",
    "LocalSpectralEncoder",
    "TwoBranchDynamicModel",
    "build_dynamic_graphs_batched",
    "compute_envelope_correlation_batched",
    "configure_runtime",
    "extract_connectivity_node_features_batch",
    "extract_spectral_features_torch",
    "process_batched_window_features",
    "resolve_device",
]

