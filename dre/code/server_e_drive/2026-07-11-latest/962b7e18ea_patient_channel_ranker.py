from __future__ import annotations

import torch
from torch import nn


class PatientChannelClassifier(nn.Module):
    """
    Lightweight patient-relative channel classifier.

    Logits represent the configured positive label. The default positive label
    remains NEZ for backward compatibility; when positive_label="ez",
    sigmoid(logits) is p(EZ).
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 2,
        dropout: float = 0.25,
        use_patient_relative_z: bool = True,
        positive_label: str = "nez",
        use_a9v8_lcbo: bool = False,
        eval_score_fusion_gamma: float = 0.10,
    ) -> None:
        super().__init__()
        dim = int(input_dim)
        self.use_patient_relative_z = bool(use_patient_relative_z)
        self.positive_label = str(positive_label).strip().lower()
        self.use_a9v8_lcbo = bool(use_a9v8_lcbo)
        self.eval_score_fusion_gamma = float(eval_score_fusion_gamma)
        if self.positive_label not in {"nez", "ez"}:
            raise ValueError(f"Unsupported positive_label={positive_label!r}; expected 'nez' or 'ez'.")
        if self.use_a9v8_lcbo and self.positive_label != "ez":
            raise ValueError("A9v8 LCBO classifier requires positive_label='ez'.")
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
        if self.use_a9v8_lcbo:
            self.core_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(dim, 1),
            )
            self.broad_head = nn.Sequential(
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
        if self.use_a9v8_lcbo:
            logits_core = self.core_head(h).squeeze(-1).masked_fill(~channel_mask, -1e9)
            logits_broad = self.broad_head(h).squeeze(-1).masked_fill(~channel_mask, -1e9)
            logits_eval = (logits_broad + self.eval_score_fusion_gamma * logits_core).masked_fill(~channel_mask, -1e9)
            score_core = torch.sigmoid(logits_core).masked_fill(~channel_mask, 0.0)
            score_broad = torch.sigmoid(logits_broad).masked_fill(~channel_mask, 0.0)
            score_eval = torch.sigmoid(logits_eval).masked_fill(~channel_mask, 0.0)
            return {
                "logits": logits_eval,
                "scores": score_eval,
                "score_nez": (1.0 - score_eval).masked_fill(~channel_mask, 0.0),
                "score_ez": score_eval,
                "logits_core": logits_core,
                "logits_broad": logits_broad,
                "logits_eval": logits_eval,
                "score_core": score_core,
                "score_broad": score_broad,
                "score_eval": score_eval,
            }
        logits = self.classifier(h).squeeze(-1).masked_fill(~channel_mask, -1e9)
        scores = torch.sigmoid(logits)
        score_ez = scores if self.positive_label == "ez" else 1.0 - scores
        score_nez = scores if self.positive_label == "nez" else 1.0 - scores
        return {
            "logits": logits,
            "scores": scores,
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
