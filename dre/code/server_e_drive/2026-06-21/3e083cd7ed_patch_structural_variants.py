from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def write(path: str, text: str) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        bak = p.with_suffix(p.suffix + ".bak_a0a7")
        if not bak.exists():
            bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(text, encoding="utf-8")
    print(f"patched {path}")


# 1) config.py: preserve user-selected positive_label.
config_path = ROOT / "neuroez_c" / "config.py"
if config_path.exists():
    txt = config_path.read_text(encoding="utf-8")
    start = txt.find("def apply_pruned_defaults(args) -> None:")
    if start >= 0:
        prefix = txt[:start]
        new_config = prefix + '''def apply_pruned_defaults(args) -> None:
    setattr(args, "model_family", "b0_pruned_ez_backbone")
    positive_label = str(getattr(args, "positive_label", "nez")).strip().lower()
    if positive_label not in {"nez", "ez"}:
        raise ValueError(f"Unsupported positive_label={positive_label!r}; expected 'nez' or 'ez'.")
    setattr(args, "positive_label", positive_label)
    setattr(args, "score_semantics", "ez_probability" if positive_label == "ez" else "nez_probability")
    if not getattr(args, "window_cache_path", None):
        setattr(args, "window_cache_path", DEFAULT_WINDOW_CACHE)
    if not getattr(args, "b0_feature_parts", None):
        setattr(args, "b0_feature_parts", "abs,delta,zdelta,ratio")
    if not getattr(args, "b0_feature_groups", None):
        setattr(args, "b0_feature_groups", "spectral_classical")
    setattr(args, "drop_high_ez_fraction_lzu", bool(getattr(args, "drop_high_ez_fraction_lzu", True)))
    setattr(args, "lzu_max_ez_fraction", float(getattr(args, "lzu_max_ez_fraction", 0.40)))


__all__ = ["DEFAULT_WINDOW_CACHE", "apply_pruned_defaults"]
'''
        config_path.write_text(new_config, encoding="utf-8")
        print("patched neuroez_c/config.py")


# 2) run_neuroez_c.py: add structural experiment args if missing.
run_path = ROOT / "run_neuroez_c.py"
if run_path.exists():
    txt = run_path.read_text(encoding="utf-8")
    if "--temporal_pooling" not in txt:
        anchor = '    parser.add_argument("--use_patient_relative_z", type=_str_to_bool, nargs="?", const=True, default=True)\n'
        insert = anchor + '''    parser.add_argument("--temporal_pooling", type=str, default="mean", choices=["mean", "mean_max_topk", "gated_attention"])
    parser.add_argument("--temporal_topk_frac", type=float, default=0.20)
    parser.add_argument("--seizure_aggregation", type=str, default="mean_std", choices=["mean_std", "attention"])
    parser.add_argument("--use_center_affine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--center_affine_strength", type=float, default=0.10)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
'''
        txt = txt.replace(anchor, insert)
        run_path.write_text(txt, encoding="utf-8")
        print("patched run_neuroez_c.py args")


write("patient_channel_ranker.py", r'''from __future__ import annotations

import torch
from torch import nn


class PatientChannelClassifier(nn.Module):
    """
    Lightweight patient-relative channel classifier.

    logits/scores represent the configured positive class:
      positive_label='nez': sigmoid(logits)=p(NEZ)
      positive_label='ez' : sigmoid(logits)=p(EZ)

    Reports always expose score_nez and score_ez with correct semantics.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 2,
        dropout: float = 0.25,
        use_patient_relative_z: bool = True,
        positive_label: str = "nez",
    ) -> None:
        super().__init__()
        dim = int(input_dim)
        self.use_patient_relative_z = bool(use_patient_relative_z)
        self.positive_label = str(positive_label).strip().lower()
        if self.positive_label not in {"nez", "ez"}:
            raise ValueError(f"Unsupported positive_label={self.positive_label!r}; expected 'nez' or 'ez'.")
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )

    def forward(self, patient_channel_embedding: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = (
            _patient_relative_zscore(patient_channel_embedding, channel_mask)
            if self.use_patient_relative_z
            else patient_channel_embedding * channel_mask.float().unsqueeze(-1)
        )
        key_padding_mask = ~channel_mask
        all_invalid = key_padding_mask.all(dim=1)
        if torch.any(all_invalid):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid] = False
        context, _ = self.channel_attn(h, h, h, key_padding_mask=key_padding_mask)
        h = self.attn_norm(h + self.dropout(context))
        logits = self.classifier(h).squeeze(-1).masked_fill(~channel_mask, -1e9)
        score_positive = torch.sigmoid(logits)
        if self.positive_label == "ez":
            score_ez = score_positive
            score_nez = 1.0 - score_positive
        else:
            score_nez = score_positive
            score_ez = 1.0 - score_positive
        return {
            "logits": logits,
            "scores": score_positive,
            "score_nez": score_nez,
            "score_ez": score_ez,
        }


def _patient_relative_zscore(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    mask = channel_mask.float().unsqueeze(-1)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / count
    var = (((x - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
    z = (x - mean) / torch.sqrt(var + 1e-5)
    return z * mask


__all__ = ["PatientChannelClassifier"]
''')


write("temporal_encoder.py", r'''from __future__ import annotations

import math
import torch
from torch import nn


class ChannelTemporalEncoder(nn.Module):
    """
    Temporal encoder over peri-onset windows.

    Input:  [B, S, T, C, D]
    Output: [B, S, C, D]

    pooling='mean' preserves the original baseline.
    pooling='mean_max_topk' concatenates masked mean, masked max, and top-k evidence mean,
    then projects back to D.
    pooling='gated_attention' learns per-window weights inside each seizure-channel.
    """

    def __init__(
        self,
        model_dim: int = 96,
        pooling: str = "mean",
        topk_frac: float = 0.20,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.pooling = str(pooling).strip().lower()
        self.topk_frac = float(topk_frac)
        if self.pooling not in {"mean", "mean_max_topk", "gated_attention"}:
            raise ValueError(f"Unsupported temporal pooling: {self.pooling}")
        self.proj = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.model_dim),
        ) if self.pooling == "mean_max_topk" else None
        self.attn = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, 1),
        ) if self.pooling == "gated_attention" else None
        self.attn_norm = nn.LayerNorm(self.model_dim) if self.pooling == "gated_attention" else None

    def forward(
        self,
        window_embeddings: torch.Tensor,
        seizure_channel_mask: torch.Tensor | None = None,
        window_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, t, c, d = window_embeddings.shape
        if seizure_channel_mask is None:
            seizure_channel_mask = torch.ones((b, s, c), dtype=torch.bool, device=window_embeddings.device)
        time_mask = seizure_channel_mask[:, :, None, :].expand(b, s, t, c)
        if window_mask is not None:
            time_mask = time_mask & window_mask[:, :, :, None].expand(b, s, t, c)
        weights = time_mask.float()
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1.0)

        if self.pooling == "mean":
            pooled = (window_embeddings * weights.unsqueeze(-1)).sum(dim=2) / denom.squeeze(2).unsqueeze(-1)
            pooled = pooled * seizure_channel_mask.float().unsqueeze(-1)
            temporal_weights = (weights / denom).permute(0, 1, 3, 2).contiguous()
            return pooled, temporal_weights

        if self.pooling == "mean_max_topk":
            mean = (window_embeddings * weights.unsqueeze(-1)).sum(dim=2) / denom.squeeze(2).unsqueeze(-1)
            masked = window_embeddings.masked_fill(~time_mask.unsqueeze(-1), -1e9)
            maxv = masked.max(dim=2).values
            maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))

            evidence = window_embeddings.norm(dim=-1).masked_fill(~time_mask, -1e9)
            k = max(1, int(math.ceil(float(t) * max(1e-6, min(self.topk_frac, 1.0)))))
            k = min(k, t)
            top_idx = evidence.topk(k=k, dim=2).indices
            gather_idx = top_idx.unsqueeze(-1).expand(b, s, k, c, d)
            top_values = torch.gather(window_embeddings, dim=2, index=gather_idx)
            top_valid = torch.gather(time_mask, dim=2, index=top_idx).float()
            top_mean = (top_values * top_valid.unsqueeze(-1)).sum(dim=2) / top_valid.sum(dim=2).clamp_min(1.0).unsqueeze(-1)

            pooled = self.proj(torch.cat([mean, maxv, top_mean], dim=-1))
            pooled = pooled * seizure_channel_mask.float().unsqueeze(-1)
            temporal_weights = (weights / denom).permute(0, 1, 3, 2).contiguous()
            return pooled, temporal_weights

        # gated_attention
        logits = self.attn(window_embeddings).squeeze(-1).masked_fill(~time_mask, -1e9)
        all_invalid = ~time_mask.any(dim=2)
        if torch.any(all_invalid):
            logits = logits.masked_fill(all_invalid[:, :, None, :].expand_as(logits), 0.0)
        attn = torch.softmax(logits, dim=2) * weights
        attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled = (window_embeddings * attn.unsqueeze(-1)).sum(dim=2)
        pooled = self.attn_norm(pooled) if self.attn_norm is not None else pooled
        pooled = pooled * seizure_channel_mask.float().unsqueeze(-1)
        temporal_weights = attn.permute(0, 1, 3, 2).contiguous()
        return pooled, temporal_weights


__all__ = ["ChannelTemporalEncoder"]
''')


write("seizure_aggregator.py", r'''from __future__ import annotations

import torch
from torch import nn


class CrossSeizureMILAggregator(nn.Module):
    """
    Cross-seizure aggregation for each patient-channel.

    aggregation='mean_std' preserves the original baseline.
    aggregation='attention' learns seizure-level MIL weights and concatenates attention mean + weighted std.
    """

    def __init__(self, model_dim: int = 96, aggregation: str = "mean_std", dropout: float = 0.0, **_: object) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.aggregation = str(aggregation).strip().lower()
        if self.aggregation not in {"mean_std", "attention"}:
            raise ValueError(f"Unsupported seizure aggregation: {self.aggregation}")
        self.output_dim = self.model_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, 1),
        ) if self.aggregation == "attention" else None

    def forward(
        self,
        seizure_channel_embedding: torch.Tensor,
        seizure_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = seizure_mask[:, :, None] & seizure_channel_mask
        weights = valid.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)

        if self.aggregation == "mean_std":
            mean = (seizure_channel_embedding * weights.unsqueeze(-1)).sum(dim=1) / denom.squeeze(1).unsqueeze(-1)
            centered = (seizure_channel_embedding - mean[:, None, :, :]) * weights.unsqueeze(-1)
            var = centered.square().sum(dim=1) / denom.squeeze(1).unsqueeze(-1)
            std = torch.sqrt(var.clamp_min(1e-8))
            embedding = torch.cat([mean, std], dim=-1)
            embedding = embedding * valid.any(dim=1).float().unsqueeze(-1)
            seizure_weights = (weights / denom).permute(0, 2, 1).contiguous()
            return embedding, seizure_weights

        logits = self.attn(seizure_channel_embedding).squeeze(-1).masked_fill(~valid, -1e9)
        all_invalid = ~valid.any(dim=1)
        if torch.any(all_invalid):
            logits = logits.masked_fill(all_invalid[:, None, :].expand_as(logits), 0.0)
        attn = torch.softmax(logits, dim=1) * weights
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mean = (seizure_channel_embedding * attn.unsqueeze(-1)).sum(dim=1)
        centered = (seizure_channel_embedding - mean[:, None, :, :]) * attn.unsqueeze(-1)
        var = centered.square().sum(dim=1)
        std = torch.sqrt(var.clamp_min(1e-8))
        embedding = torch.cat([mean, std], dim=-1)
        embedding = embedding * valid.any(dim=1).float().unsqueeze(-1)
        seizure_weights = attn.permute(0, 2, 1).contiguous()
        return embedding, seizure_weights


__all__ = ["CrossSeizureMILAggregator"]
''')


write("neuroez_c/model.py", r'''from __future__ import annotations

from typing import Any

import torch
from torch import nn

from graph_spectral_encoder import WindowGraphSpectralEncoder
from patient_channel_ranker import PatientChannelClassifier
from seizure_aggregator import CrossSeizureMILAggregator
from temporal_encoder import ChannelTemporalEncoder
from .diffusion_residual import DiffusionSourceResidualEncoder
from .physics_dynamics import NeuralDynamicsResidualEncoder


_CENTER_TO_ID = {
    "hup": 0,
    "lzu": 1,
    "multicenter": 2,
    "pediatric": 3,
    "nih": 4,
    "jhu": 5,
    "other": 6,
}


def _center_from_subject(subject_id: str) -> str:
    sid = str(subject_id).strip().lower()
    if ":" in sid:
        return sid.split(":", 1)[0]
    return "other"


class NeuroEZCModel(nn.Module):
    """B0-Pruned patient-level spectral/classical model for EZ/NEZ localization."""

    def __init__(self, args: Any | None = None) -> None:
        super().__init__()
        self.model_dim = int(getattr(args, "model_dim", 32) if args is not None else 32)
        dropout = float(getattr(args, "dropout", 0.40) if args is not None else 0.40)
        num_heads = int(getattr(args, "num_heads", 2) if args is not None else 2)
        use_channel_attention = bool(getattr(args, "use_channel_attention", True) if args is not None else True)
        self.use_physics_dynamics = bool(getattr(args, "use_physics_dynamics", False) if args is not None else False)
        self.use_diffusion_residual = bool(getattr(args, "use_diffusion_residual", False) if args is not None else False)
        self.use_center_affine = bool(getattr(args, "use_center_affine", False) if args is not None else False)
        self.center_affine_strength = float(getattr(args, "center_affine_strength", 0.10) if args is not None else 0.10)

        self.b0_encoder = WindowGraphSpectralEncoder(
            model_dim=self.model_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_channel_attention=use_channel_attention,
        )
        if self.use_center_affine:
            self.center_scale = nn.Embedding(len(_CENTER_TO_ID), self.model_dim)
            self.center_bias = nn.Embedding(len(_CENTER_TO_ID), self.model_dim)
            nn.init.zeros_(self.center_scale.weight)
            nn.init.zeros_(self.center_bias.weight)

        if self.use_physics_dynamics:
            self.physics_encoder = NeuralDynamicsResidualEncoder(args, model_dim=self.model_dim)
            self.physics_gate = nn.Linear(2 * self.model_dim, self.model_dim)
            nn.init.zeros_(self.physics_gate.weight)
            nn.init.constant_(self.physics_gate.bias, float(getattr(args, "physics_gate_init", -3.0) if args is not None else -3.0))
        if self.use_diffusion_residual:
            self.diffusion_encoder = DiffusionSourceResidualEncoder(args, model_dim=self.model_dim)
            self.diffusion_gate = nn.Linear(2 * self.model_dim, self.model_dim)
            nn.init.zeros_(self.diffusion_gate.weight)
            nn.init.constant_(self.diffusion_gate.bias, float(getattr(args, "diffusion_gate_init", -4.0) if args is not None else -4.0))

        self.m1_norm = nn.LayerNorm(self.model_dim)
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.temporal_encoder = ChannelTemporalEncoder(
            model_dim=self.model_dim,
            pooling=str(getattr(args, "temporal_pooling", "mean") if args is not None else "mean"),
            topk_frac=float(getattr(args, "temporal_topk_frac", 0.20) if args is not None else 0.20),
            dropout=dropout,
        )
        self.seizure_aggregator = CrossSeizureMILAggregator(
            model_dim=self.model_dim,
            aggregation=str(getattr(args, "seizure_aggregation", "mean_std") if args is not None else "mean_std"),
            dropout=dropout,
        )
        self.channel_classifier = PatientChannelClassifier(
            input_dim=self.seizure_aggregator.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_patient_relative_z=bool(getattr(args, "use_patient_relative_z", True) if args is not None else True),
            positive_label=str(getattr(args, "positive_label", "nez") if args is not None else "nez"),
        )

    def _apply_center_affine(self, h: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        if not self.use_center_affine:
            return h
        subject_ids = batch.get("subject_id", [])
        ids = []
        for sid in subject_ids:
            center = _center_from_subject(str(sid))
            ids.append(_CENTER_TO_ID.get(center, _CENTER_TO_ID["other"]))
        if not ids:
            return h
        center_ids = torch.tensor(ids, dtype=torch.long, device=h.device)
        scale = 1.0 + self.center_affine_strength * torch.tanh(self.center_scale(center_ids)).view(h.shape[0], 1, 1, 1, self.model_dim)
        bias = self.center_affine_strength * self.center_bias(center_ids).view(h.shape[0], 1, 1, 1, self.model_dim)
        return h * scale + bias

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        seizure_channel_mask = batch["seizure_channel_mask"]
        h_b0 = self.b0_encoder(batch["b0_features"], None, seizure_channel_mask)
        h_b0 = self._apply_center_affine(h_b0, batch)
        branch_losses: dict[str, torch.Tensor] = {}

        if self.use_physics_dynamics:
            if "physics_features" not in batch:
                raise KeyError("Batch is missing physics_features while use_physics_dynamics=True.")
            h_dyn, physics_losses = self.physics_encoder(
                batch["physics_features"],
                batch.get("window_mask"),
                seizure_channel_mask,
                batch.get("window_centers"),
            )
            gate = torch.sigmoid(self.physics_gate(torch.cat([h_b0, h_dyn], dim=-1)))
            h_m1 = self.m1_norm(h_b0 + gate * h_dyn)
            branch_losses.update(physics_losses)
            branch_losses["physics_gate_mean"] = gate.mean()
        else:
            h_m1 = self.m1_norm(h_b0)

        if self.use_diffusion_residual:
            if "physics_features" not in batch:
                raise KeyError("Batch is missing physics_features while use_diffusion_residual=True.")
            h_diff, diffusion_losses = self.diffusion_encoder(
                batch["physics_features"],
                batch.get("diffusion_adjacency"),
                batch.get("window_mask"),
                seizure_channel_mask,
                batch.get("window_centers"),
            )
            gate_diff = torch.sigmoid(self.diffusion_gate(torch.cat([h_m1, h_diff], dim=-1)))
            fused = self.final_norm(h_m1 + gate_diff * h_diff)
            branch_losses.update(diffusion_losses)
            branch_losses["diffusion_gate_mean"] = gate_diff.mean()
        else:
            fused = h_m1

        seizure_channel_embedding, temporal_weights = self.temporal_encoder(
            fused,
            seizure_channel_mask,
            window_mask=batch.get("window_mask"),
        )
        patient_channel_embedding, seizure_weights = self.seizure_aggregator(
            seizure_channel_embedding,
            batch["seizure_mask"],
            seizure_channel_mask,
        )
        output = self.channel_classifier(patient_channel_embedding, batch["channel_mask"])
        output.update(
            {
                "patient_channel_embedding": patient_channel_embedding,
                "task_embedding": patient_channel_embedding,
                "seizure_channel_embedding": seizure_channel_embedding,
                "temporal_attention": temporal_weights,
                "seizure_attention": seizure_weights,
            }
        )
        output.update(branch_losses)
        return output


__all__ = ["NeuroEZCModel"]
''')


# 3) exp_ez_hybrid.py: add label smoothing to BCE and ensure pairwise ranking knows positive_label.
exp_path = ROOT / "exp_ez_hybrid.py"
if exp_path.exists():
    txt = exp_path.read_text(encoding="utf-8")
    old = '''def _masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_ez: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weight_mode: str,
    ez_negative_weight: torch.Tensor,
) -> torch.Tensor:
    valid = mask & (labels >= 0.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="none")
    if str(class_weight_mode).lower() == "ez_negative":
        weights = torch.ones_like(losses)
        weights = torch.where(labels_ez[valid] > 0.5, ez_negative_weight.to(logits.device).expand_as(weights), weights)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return losses.mean()
'''
    new = '''def _masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_ez: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weight_mode: str,
    ez_negative_weight: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = mask & (labels >= 0.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    targets = labels
    smooth = float(label_smoothing)
    if smooth > 0.0:
        smooth = max(0.0, min(smooth, 0.49))
        targets = labels * (1.0 - smooth) + 0.5 * smooth
    losses = F.binary_cross_entropy_with_logits(logits[valid], targets[valid], reduction="none")
    if str(class_weight_mode).lower() == "ez_negative":
        weights = torch.ones_like(losses)
        weights = torch.where(labels_ez[valid] > 0.5, ez_negative_weight.to(logits.device).expand_as(weights), weights)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return losses.mean()
'''
    if old in txt:
        txt = txt.replace(old, new)
    if "label_smoothing=float(getattr(self.args, \"label_smoothing\", 0.0))," not in txt:
        txt = txt.replace(
'''            ez_negative_weight=ez_negative_weight,
        )
''',
'''            ez_negative_weight=ez_negative_weight,
            label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
        )
''')
    # Make pairwise ranking compatible with positive_label if the old function is still present.
    txt = txt.replace(
'''def _ez_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    margin: float = 0.10,
) -> torch.Tensor:
    valid = channel_mask & (labels_ez >= 0.0)
    score_nez = torch.sigmoid(logits)
    score_ez = 1.0 - score_nez
''',
'''def _ez_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    margin: float = 0.10,
    positive_label: str = "nez",
) -> torch.Tensor:
    valid = channel_mask & (labels_ez >= 0.0)
    score_positive = torch.sigmoid(logits)
    score_ez = score_positive if str(positive_label).lower() == "ez" else 1.0 - score_positive
''')
    txt = txt.replace(
'''                margin=ranking_margin,
            )
''',
'''                margin=ranking_margin,
                positive_label=str(getattr(self.args, "positive_label", "nez")),
            )
''')
    exp_path.write_text(txt, encoding="utf-8")
    print("patched exp_ez_hybrid.py label smoothing / ranking semantics")

print("done")
