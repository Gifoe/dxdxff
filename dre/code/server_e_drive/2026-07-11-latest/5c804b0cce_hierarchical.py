from __future__ import annotations

import torch


def _masked_stats(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=dim).clamp_min(1)
    clean = torch.where(expanded, values, torch.zeros_like(values))
    mean = clean.sum(dim=dim) / count
    variance = (torch.where(expanded, (values - mean.unsqueeze(dim)).square(), torch.zeros_like(values)).sum(dim=dim) / count).clamp_min(0)
    maximum = torch.where(expanded, values, torch.full_like(values, -torch.inf)).amax(dim=dim)
    maximum = torch.where(mask.any(dim=dim).unsqueeze(-1), maximum, torch.zeros_like(maximum))
    return torch.cat([mean, variance.sqrt(), maximum], dim=-1)


class SharedHierPoolV1(torch.nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.window_dim = self.token_dim * 3
        self.seizure_dim = self.window_dim * 3
        self.patient_dim = self.seizure_dim * 3
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.patient_dim),
            torch.nn.Linear(self.patient_dim, int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 5 or valid_mask.shape != tokens.shape[:-1]:
            raise ValueError("SharedHierPoolV1 expects tokens [B,S,W,C,D] and matching [B,S,W,C] mask.")
        window = _masked_stats(tokens, valid_mask, dim=3)
        window_mask = valid_mask.any(dim=3)
        seizure = _masked_stats(window, window_mask, dim=2)
        seizure_mask = window_mask.any(dim=2)
        patient = _masked_stats(seizure, seizure_mask, dim=1)
        return {"window_embeddings": window, "seizure_embeddings": seizure, "patient_embedding": patient, "logits": self.head(patient).squeeze(-1)}


class TokenHierarchicalOutcomeModel(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, aggregator: SharedHierPoolV1, *, frozen_backbone: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.frozen_backbone = bool(frozen_backbone)
        if self.frozen_backbone:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_backbone:
            self.encoder.eval()
        return self

    def forward(self, batch: dict[str, torch.Tensor], *, persist_diagnostics: bool = True) -> dict[str, torch.Tensor]:
        del persist_diagnostics
        values = batch["feature_x"]
        mask = batch["window_channel_mask"]
        if values.ndim != 5:
            raise ValueError("TokenHierarchicalOutcomeModel requires [B,S,W,C,input] values.")
        valid_values = values[mask].unsqueeze(1)
        if self.frozen_backbone:
            with torch.no_grad():
                encoded = self.encoder(valid_values)
        else:
            encoded = self.encoder(valid_values)
        token_dim = int(encoded.shape[-1])
        restored = torch.zeros((*values.shape[:-1], token_dim), dtype=encoded.dtype, device=encoded.device)
        restored[mask] = encoded
        output = self.aggregator(restored, mask)
        output["valid_token_count"] = mask.sum()
        return output


__all__ = ["SharedHierPoolV1", "TokenHierarchicalOutcomeModel"]
