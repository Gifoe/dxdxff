from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RawBrainBERTModelConfig:
    patch_dim: int
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_time_patches: int = 4096
    max_freq_patches: int = 512

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RawBrainBERTEncoder(nn.Module):
    """Masked spectrogram-patch encoder used only for V3-HNC post-processing."""

    def __init__(
        self,
        *,
        patch_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_time_patches: int = 4096,
        max_freq_patches: int = 512,
    ) -> None:
        super().__init__()
        self.config = RawBrainBERTModelConfig(
            patch_dim=int(patch_dim),
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            max_time_patches=int(max_time_patches),
            max_freq_patches=int(max_freq_patches),
        )
        self.patch_projection = nn.Linear(self.config.patch_dim, self.config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(self.config.d_model))
        self.time_embedding = nn.Embedding(self.config.max_time_patches, self.config.d_model)
        self.freq_embedding = nn.Embedding(self.config.max_freq_patches, self.config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.reconstruction_head = nn.Linear(self.config.d_model, self.config.patch_dim)

        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def tensor_from_numpy(self, value: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.dtype == torch.bool:
            return tensor.to(next(self.parameters()).device)
        return tensor.to(next(self.parameters()).device, dtype=torch.float32)

    def from_numpy(
        self,
        patches: np.ndarray,
        time_ids: np.ndarray,
        freq_ids: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        patch_tensor = self.tensor_from_numpy(patches)
        time_tensor = torch.as_tensor(time_ids, dtype=torch.long, device=patch_tensor.device)
        freq_tensor = torch.as_tensor(freq_ids, dtype=torch.long, device=patch_tensor.device)
        mask_tensor = None if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=patch_tensor.device)
        return self.forward(patch_tensor, time_tensor, freq_tensor, mask=mask_tensor)

    def forward(
        self,
        patches: torch.Tensor,
        time_ids: torch.Tensor,
        freq_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if patches.ndim != 3:
            raise ValueError(f"patches must have shape [B, N, patch_dim], got {tuple(patches.shape)}")
        if patches.shape[-1] != self.config.patch_dim:
            raise ValueError(f"Expected patch_dim={self.config.patch_dim}, got {patches.shape[-1]}")
        time_ids = time_ids.to(device=patches.device, dtype=torch.long).clamp(0, self.config.max_time_patches - 1)
        freq_ids = freq_ids.to(device=patches.device, dtype=torch.long).clamp(0, self.config.max_freq_patches - 1)
        x = self.patch_projection(patches)
        if mask is not None:
            mask = mask.to(device=patches.device, dtype=torch.bool)
            x = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), x)
        x = x + self.time_embedding(time_ids) + self.freq_embedding(freq_ids)
        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)
        reconstruction = self.reconstruction_head(hidden)
        return {"hidden": hidden, "reconstruction": reconstruction}


def masked_patch_loss(reconstruction: torch.Tensor, original_patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=reconstruction.device, dtype=torch.bool)
    original_patches = original_patches.to(device=reconstruction.device, dtype=reconstruction.dtype)
    if mask.any():
        return F.mse_loss(reconstruction[mask], original_patches[mask])
    return F.mse_loss(reconstruction, original_patches)


def make_random_patch_mask(valid_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    valid_mask = valid_mask.to(dtype=torch.bool)
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [B, N].")
    ratio = float(mask_ratio)
    if ratio <= 0:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    random_values = torch.rand(valid_mask.shape, device=valid_mask.device)
    mask = (random_values < ratio) & valid_mask
    for row_idx in range(mask.shape[0]):
        if valid_mask[row_idx].any() and not mask[row_idx].any():
            valid_indices = torch.nonzero(valid_mask[row_idx], as_tuple=False).flatten()
            chosen = valid_indices[torch.randint(0, valid_indices.numel(), (1,), device=valid_mask.device)]
            mask[row_idx, chosen] = True
    return mask


def build_encoder_from_checkpoint_payload(payload: dict[str, Any]) -> RawBrainBERTEncoder:
    config = dict(payload.get("model_config") or {})
    model = RawBrainBERTEncoder(**config)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    return model


__all__ = [
    "RawBrainBERTEncoder",
    "RawBrainBERTModelConfig",
    "build_encoder_from_checkpoint_payload",
    "make_random_patch_mask",
    "masked_patch_loss",
]
