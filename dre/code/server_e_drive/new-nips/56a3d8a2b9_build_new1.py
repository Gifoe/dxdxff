from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SPECTRAL_MIN_FREQ = 1.0
DEFAULT_SPECTRAL_MAX_FREQ = 250.0
DEFAULT_SPECTRAL_BANDS: Sequence[Tuple[str, float, float]] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 80.0),
    ("high_gamma", 80.0, 150.0),
    ("ripple", 150.0, 250.0),
)
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


def _build_topology_templates(contacts_meta: pd.DataFrame, device: torch.device) -> Dict[str, torch.Tensor]:
    groups = contacts_meta["contact_group"].fillna("").astype(str).to_numpy()
    numbers = pd.to_numeric(contacts_meta["contact_number"], errors="coerce").to_numpy(dtype=np.float32)

    same_group_mat = groups[:, None] == groups[None, :]
    dist_mat = np.abs(numbers[:, None] - numbers[None, :])
    dist_mat[~np.isfinite(dist_mat)] = np.inf
    dist_mat[~same_group_mat] = np.inf

    topo_adj_mat = same_group_mat & np.isfinite(dist_mat) & (dist_mat == 1.0)
    topo_adj_t = torch.as_tensor(topo_adj_mat, dtype=torch.bool, device=device)

    topo_src, topo_dst = np.where(topo_adj_mat)
    topo_src_t = torch.as_tensor(topo_src, dtype=torch.long, device=device)
    topo_dst_t = torch.as_tensor(topo_dst, dtype=torch.long, device=device)

    degree = topo_adj_t.to(dtype=torch.float32).sum(dim=-1, keepdim=True)
    boundary_flags = (degree <= 1.0).to(dtype=torch.float32)

    return {
        "topo_adj": topo_adj_t,
        "topo_src": topo_src_t,
        "topo_dst": topo_dst_t,
        "degree": degree,
        "boundary_flags": boundary_flags,
    }


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
    restricted_power = power[..., freq_mask]
    selected_freqs = freqs[freq_mask]

    band_power_list: List[torch.Tensor] = []
    for _, band_low, band_high in DEFAULT_SPECTRAL_BANDS:
        low = max(float(band_low), float(min_freq))
        high = min(float(band_high), effective_max_freq)

        if high <= low:
            band_power = restricted_power.new_zeros(restricted_power.shape[:2])
        else:
            upper_inclusive = high >= effective_max_freq
            band_mask = (selected_freqs >= low) & (selected_freqs <= high if upper_inclusive else selected_freqs < high)
            if torch.any(band_mask):
                band_power = restricted_power[..., band_mask].mean(dim=-1)
            else:
                band_power = restricted_power.new_zeros(restricted_power.shape[:2])

        band_scale = 1.0
        if spectral_band_weights:
            for weighted_low, weighted_high, weight in spectral_band_weights:
                overlap = max(0.0, min(high, float(weighted_high)) - max(low, float(weighted_low)))
                if overlap > 0.0:
                    band_scale = max(band_scale, float(weight))
        band_power_list.append(band_power * band_scale)

    band_power = torch.stack(band_power_list, dim=-1)
    band_log_power = torch.log1p(band_power)

    eps = 1e-8
    delta = band_power[..., 0:1]
    theta = band_power[..., 1:2]
    alpha = band_power[..., 2:3]
    beta = band_power[..., 3:4]
    low_gamma = band_power[..., 4:5]
    high_gamma = band_power[..., 5:6]
    ripple = band_power[..., 6:7]

    ratio_theta_delta = theta / (delta + eps)
    ratio_beta_alpha = beta / (alpha + eps)
    ratio_high_low_gamma = high_gamma / (low_gamma + eps)
    ratio_ripple_background = ripple / (beta + low_gamma + eps)

    band_prob = band_power / band_power.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy_norm = torch.log(
        torch.tensor(float(max(int(band_power.shape[-1]), 2)), dtype=band_prob.dtype, device=band_prob.device)
    ).clamp_min(1.0)
    spectral_entropy = -torch.sum(band_prob * torch.log(band_prob.clamp_min(eps)), dim=-1, keepdim=True) / entropy_norm

    rms = torch.sqrt(torch.mean(data_t.pow(2), dim=-1, keepdim=True) + 1e-8)
    variance = torch.var(data_t, dim=-1, unbiased=False, keepdim=True)
    line_length = torch.sum(torch.abs(torch.diff(data_t, dim=-1)), dim=-1, keepdim=True)

    return torch.cat(
        [
            band_log_power,
            ratio_theta_delta,
            ratio_beta_alpha,
            ratio_high_low_gamma,
            ratio_ripple_background,
            spectral_entropy,
            rms,
            variance,
            line_length,
        ],
        dim=-1,
    )


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


def _compute_dynamic_topology_edge_features(
    data_t: torch.Tensor,
    env_corrs: torch.Tensor,
    topo_src_t: torch.Tensor,
    topo_dst_t: torch.Tensor,
    num_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = data_t.shape[0]
    if topo_src_t.numel() == 0:
        empty_edge_attr = data_t.new_zeros((batch_size, 0, 4))
        empty_weight_mat = data_t.new_zeros((batch_size, num_channels, num_channels))
        return empty_edge_attr, empty_weight_mat

    env_similarity = env_corrs[:, topo_src_t, topo_dst_t].clamp(-1.0, 1.0)

    rms = torch.sqrt(torch.mean(data_t.pow(2), dim=-1) + 1e-8)
    amp_src = rms[:, topo_src_t]
    amp_dst = rms[:, topo_dst_t]
    amp_similarity = 1.0 - torch.abs(amp_src - amp_dst) / (amp_src + amp_dst + 1e-6)
    amp_similarity = amp_similarity.clamp(0.0, 1.0)

    if data_t.size(-1) > 1:
        diff_t = torch.diff(data_t, dim=-1)
        diff_src = diff_t[:, topo_src_t, :]
        diff_dst = diff_t[:, topo_dst_t, :]
        diff_num = torch.mean(diff_src * diff_dst, dim=-1)
        diff_den = torch.sqrt(
            torch.mean(diff_src.pow(2), dim=-1) * torch.mean(diff_dst.pow(2), dim=-1)
        ).clamp_min(1e-8)
        diff_consistency = (diff_num / diff_den).clamp(-1.0, 1.0)
    else:
        diff_consistency = torch.zeros_like(env_similarity)

    amp_similarity_signed = amp_similarity * 2.0 - 1.0
    topo_weight = (0.5 * env_similarity + 0.3 * amp_similarity_signed + 0.2 * diff_consistency).clamp(-1.0, 1.0)

    edge_attr = torch.stack([topo_weight, env_similarity, amp_similarity, diff_consistency], dim=-1)

    topo_weight_mats = data_t.new_zeros((batch_size, num_channels, num_channels))
    topo_weight_mats[:, topo_src_t, topo_dst_t] = topo_weight
    return edge_attr, topo_weight_mats


def extract_connectivity_node_features_batch(
    env_corrs: Any,
    topology_weight_mats: Any,
    topology_adj: Any,
    boundary_flags: Any,
) -> torch.Tensor:
    if not isinstance(env_corrs, torch.Tensor):
        env_corrs = torch.as_tensor(env_corrs, dtype=torch.float32)
    env_corrs = env_corrs.to(dtype=torch.float32)

    topo_weight_t = _ensure_float_tensor(topology_weight_mats, env_corrs.device)
    topo_adj_t = torch.as_tensor(topology_adj, dtype=torch.bool, device=env_corrs.device)
    boundary_t = _ensure_float_tensor(boundary_flags, env_corrs.device)

    batch_size, num_channels, _ = env_corrs.shape
    if num_channels == 0:
        return env_corrs.new_zeros((batch_size, 0, 7))

    if boundary_t.dim() == 1:
        boundary_t = boundary_t.view(-1, 1)
    boundary_t = boundary_t[:num_channels]

    topo_adj_f = topo_adj_t.to(dtype=env_corrs.dtype)
    degree = topo_adj_f.sum(dim=-1, keepdim=True)
    norm_degree = degree / degree.max().clamp_min(1.0)

    degree_rep = degree.unsqueeze(0).expand(batch_size, -1, -1)
    norm_degree_rep = norm_degree.unsqueeze(0).expand(batch_size, -1, -1)
    boundary_rep = boundary_t.unsqueeze(0).expand(batch_size, -1, -1)

    strength = (topo_weight_t.abs() * topo_adj_f.unsqueeze(0)).sum(dim=-1, keepdim=True)

    neighbor_strength_sum = torch.matmul(
        topo_adj_f.unsqueeze(0).expand(batch_size, -1, -1),
        strength,
    )
    mean_neighbor_strength = neighbor_strength_sum / degree.view(1, num_channels, 1).clamp_min(1.0)

    local_efficiency = env_corrs.new_zeros((batch_size, num_channels, 1))
    clustering_coeff = env_corrs.new_zeros((batch_size, num_channels, 1))

    for node_idx in range(num_channels):
        neighbor_idx = torch.where(topo_adj_t[node_idx])[0]
        n_neighbors = int(neighbor_idx.numel())
        if n_neighbors < 2:
            continue

        sub_corr = env_corrs[:, neighbor_idx][:, :, neighbor_idx].abs()
        diag_mask = torch.eye(n_neighbors, dtype=torch.bool, device=env_corrs.device).unsqueeze(0)
        pair_sum = sub_corr.masked_fill(diag_mask, 0.0).sum(dim=(1, 2))
        norm = float(n_neighbors * (n_neighbors - 1))
        local_efficiency[:, node_idx, 0] = pair_sum / max(norm, 1.0)

        neighbor_topo = topo_adj_t[neighbor_idx][:, neighbor_idx]
        possible_edges = float(neighbor_topo.sum().item())
        if possible_edges > 0:
            sub_weights = topo_weight_t[:, neighbor_idx][:, :, neighbor_idx].abs()
            clustering_coeff[:, node_idx, 0] = (
                sub_weights * neighbor_topo.to(dtype=sub_weights.dtype).unsqueeze(0)
            ).sum(dim=(1, 2)) / possible_edges
        else:
            clustering_coeff[:, node_idx, 0] = local_efficiency[:, node_idx, 0]

    return torch.cat(
        [
            degree_rep,
            norm_degree_rep,
            strength,
            local_efficiency,
            clustering_coeff,
            mean_neighbor_strength,
            boundary_rep,
        ],
        dim=-1,
    )


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

    env_corrs = compute_envelope_correlation_batched(data_t).to(device_obj)
    topology = _build_topology_templates(contacts_meta, device_obj)

    edge_attrs_t, topology_weight_mats = _compute_dynamic_topology_edge_features(
        data_t,
        env_corrs,
        topology["topo_src"],
        topology["topo_dst"],
        num_channels,
    )

    batched_edge_indices = []
    batched_edge_attrs = []
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device_obj)
    empty_edge_attr = torch.empty((0, 4), dtype=torch.float32, device=device_obj)

    for batch_idx in range(batch_size):
        if topology["topo_src"].numel() > 0:
            final_edge_index = torch.stack([topology["topo_src"], topology["topo_dst"]], dim=0)
            final_edge_attr = edge_attrs_t[batch_idx]
        else:
            final_edge_index = empty_edge_index
            final_edge_attr = empty_edge_attr

        batched_edge_indices.append(final_edge_index.cpu())
        batched_edge_attrs.append(final_edge_attr.cpu())

    return (
        env_corrs,
        batched_edge_indices,
        batched_edge_attrs,
        topology_weight_mats,
        topology["topo_adj"],
        topology["boundary_flags"],
    )


def process_batched_window_features(
    windows_data: Any,
    sfreq: float,
    env_corrs: Optional[Any] = None,
    topology_weight_mats: Optional[Any] = None,
    topology_adj: Optional[Any] = None,
    boundary_flags: Optional[Any] = None,
    device: Optional[str] = None,
    min_freq: float = DEFAULT_SPECTRAL_MIN_FREQ,
    max_freq: float = DEFAULT_SPECTRAL_MAX_FREQ,
    spectral_band_weights = DEFAULT_HIGH_FREQ_BAND_WEIGHTS,
) -> Dict[str, torch.Tensor]:
    device_obj = configure_runtime(device)
    data_t = _ensure_float_tensor(windows_data, device_obj)
    _, num_channels, _ = data_t.shape

    if env_corrs is None:
        env_corrs_t = compute_envelope_correlation_batched(data_t)
    else:
        env_corrs_t = _ensure_float_tensor(env_corrs, device_obj)

    if topology_adj is None:
        topology_adj_t = torch.zeros((num_channels, num_channels), dtype=torch.bool, device=device_obj)
    else:
        topology_adj_t = torch.as_tensor(topology_adj, dtype=torch.bool, device=device_obj)

    if topology_weight_mats is None:
        topology_weight_mats_t = env_corrs_t * topology_adj_t.to(dtype=env_corrs_t.dtype).unsqueeze(0)
    else:
        topology_weight_mats_t = _ensure_float_tensor(topology_weight_mats, device_obj)

    if boundary_flags is None:
        degree = topology_adj_t.to(dtype=torch.float32).sum(dim=-1, keepdim=True)
        boundary_flags_t = (degree <= 1.0).to(dtype=torch.float32)
    else:
        boundary_flags_t = _ensure_float_tensor(boundary_flags, device_obj)

    node_spec = extract_spectral_features_torch(
        data_t,
        sfreq,
        min_freq=min_freq,
        max_freq=max_freq,
        spectral_band_weights=spectral_band_weights,
    )
    node_conn = extract_connectivity_node_features_batch(
        env_corrs_t,
        topology_weight_mats_t,
        topology_adj_t,
        boundary_flags_t,
    )

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
        self.window_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
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
        channel_context = torch.sum(gru_out * attn_weights, dim=1, keepdim=True)
        channel_context = channel_context.expand(-1, time_steps, -1)

        window_repr = self.window_proj(torch.cat([gru_out, channel_context], dim=-1))

        window_logits = self.ez_head(window_repr).squeeze(-1).transpose(0, 1).contiguous()
        window_embeddings = window_repr.transpose(0, 1).contiguous()

        window_run_embedding = window_embeddings.mean(dim=1)
        predicted_ratio_sequence = torch.sigmoid(self.count_ratio_head(window_run_embedding)).squeeze(-1)

        if return_embeddings:
            return window_logits, window_embeddings, predicted_ratio_sequence
        return window_logits, predicted_ratio_sequence


__all__ = [
    "DEFAULT_HIGH_FREQ_BAND_WEIGHTS",
    "DEFAULT_SPECTRAL_BANDS",
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

