"""One official-CNN morphology backbone with zero-output physiology/context residuals.

The same :class:`PCCNN` topology is used for both benchmarks. The official
NeuralCNN instance is supplied by the pinned upstream module and can be loaded
from a matched RawCNN checkpoint without a parameter-name translation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _gate_parameter(cap: float, effective_initial: float = 0.02) -> nn.Parameter:
    if not 0 < effective_initial < cap <= 0.5:
        raise ValueError("Residual gate must start small and within its cap")
    return nn.Parameter(torch.tensor(math.atanh(effective_initial / cap)))


def _channel_groups(selected: Tensor, maximum: int):
    """Bound memory without feeding a singleton to a training BatchNorm1d."""
    groups = list(selected.split(maximum))
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2] = torch.cat((groups[-2], groups[-1]))
        groups.pop()
    return groups


class PCCNN(nn.Module):
    """One-record, per-channel logits with permutation-equivariant set context.

    ``images`` is the pinned official-style spectral transform of the raw
    waveform, shaped [B,C,1,F,T]. Its generation is part of the benchmark
    input pipeline, not a learned second encoder. ``descriptors`` is
    [B,C,59,36], and ``descriptor_mask`` has the same shape. The mask marks
    physiologically unavailable coordinates (e.g. beyond native Nyquist).
    """

    def __init__(self, raw_cnn: nn.Module, film_cap: float = 0.5,
                 context_cap: float = 0.5, heads: int = 4):
        super().__init__()
        if heads not in (2, 4) or 32 % heads:
            raise ValueError("Only the predeclared 2/4-head configurations exist")
        if film_cap not in (0.25, 0.5) or context_cap not in (0.25, 0.5):
            raise ValueError("Only the predeclared residual caps exist")
        self.raw = raw_cnn
        self.film_cap = film_cap
        self.context_cap = context_cap
        self.physiology = nn.Sequential(
            nn.Linear(36, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64),
        )
        nn.init.zeros_(self.physiology[-1].weight)
        nn.init.zeros_(self.physiology[-1].bias)
        self.context_norm = nn.LayerNorm(32)
        self.context_mha = nn.MultiheadAttention(32, heads, batch_first=True)
        self.context_out = nn.Linear(32, 32)
        nn.init.zeros_(self.context_out.weight)
        nn.init.zeros_(self.context_out.bias)
        # Protocol amendment: zero output projections plus zero scalar gates
        # are a dead-gradient configuration. Effective gates 0.02 keep the
        # *function* exactly RawCNN at step 0 while enabling projector updates.
        self.a_gamma = _gate_parameter(film_cap)
        self.a_beta = _gate_parameter(film_cap)
        self.a_context = _gate_parameter(context_cap)

    def alphas(self) -> dict[str, Tensor]:
        return {
            "gamma": self.film_cap * torch.tanh(self.a_gamma),
            "beta": self.film_cap * torch.tanh(self.a_beta),
            "context": self.context_cap * torch.tanh(self.a_context),
        }

    def set_batch_norm_running_state(self, frozen: bool) -> None:
        """Freeze BN buffers while leaving affine parameters trainable."""
        for module in self.raw.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval() if frozen else module.train()

    def _classifier(self, embedding: Tensor) -> Tensor:
        # Exactly the order of the pinned upstream NeuralCNN.forward.
        x = self.raw.bn(self.raw.relu(self.raw.fc(embedding)))
        x = self.raw.bn1(self.raw.relu1(self.raw.fc1(x)))
        return self.raw.fc_out(x).squeeze(-1)

    def forward(self, images: Tensor, descriptors: Tensor, channel_mask: Tensor,
                descriptor_mask: Tensor, *, physiology: bool = True,
                context: bool = True) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 1:
            raise ValueError("images must be [B,C,1,F,T]")
        b, c = images.shape[:2]
        if descriptors.shape != (b, c, 59, 36):
            raise ValueError("descriptor trajectory must be [B,C,59,36]")
        if descriptor_mask.shape != descriptors.shape:
            raise ValueError("descriptor availability mask shape differs")
        if channel_mask.shape != (b, c) or not bool(channel_mask.any(dim=1).all()):
            raise ValueError("Every record needs at least one valid channel")
        keep = channel_mask.reshape(-1).bool()
        flat_images = images.reshape(b * c, *images.shape[2:])[keep]

        # Exact independent-channel RawCNN branch, including identical BN and
        # classifier call order. No score normalization or second classifier.
        if not physiology and not context:
            scores = self.raw(flat_images).squeeze(-1)
            out = images.new_zeros((b * c,))
            out[keep] = scores
            return out.reshape(b, c)

        hidden = self.raw.feature_extractor(flat_images)
        if physiology:
            d = (descriptors * descriptor_mask).reshape(b * c, 59, 36)[keep]
            d = F.interpolate(d.transpose(1, 2), size=hidden.shape[-1],
                              mode="linear", align_corners=False).transpose(1, 2)
            gamma, beta = self.physiology(d).chunk(2, dim=-1)
            gamma = gamma.transpose(1, 2).unsqueeze(2)
            beta = beta.transpose(1, 2).unsqueeze(2)
            scales = self.alphas()
            hidden = hidden * (1 + scales["gamma"] * torch.tanh(gamma)) + \
                scales["beta"] * torch.tanh(beta)
        embedding = self.raw.cnn(hidden)
        if embedding.shape[-1] != 32:
            raise RuntimeError("Pinned RawCNN must output a 32-D embedding")
        if context:
            all_embeddings = embedding.new_zeros((b * c, 32))
            all_embeddings[keep] = embedding
            all_embeddings = all_embeddings.reshape(b, c, 32)
            attended, _ = self.context_mha(self.context_norm(all_embeddings),
                                           self.context_norm(all_embeddings),
                                           self.context_norm(all_embeddings),
                                           key_padding_mask=~channel_mask.bool(),
                                           need_weights=False)
            embedding = embedding + self.alphas()["context"] * \
                self.context_out(attended.reshape(b * c, 32)[keep])
        scores = self._classifier(embedding)
        out = images.new_zeros((b * c,))
        out[keep] = scores
        return out.reshape(b, c)

    def forward_record(self, waveforms: Tensor, sampling_rate: float,
                       descriptors: Tensor, channel_mask: Tensor,
                       descriptor_mask: Tensor, preprocessor, *,
                       physiology: bool = True, context: bool = True,
                       channel_chunk: int = 4,
                       checkpoint_backbone: bool = False) -> Tensor:
        """Stream one raw record in bounded channel chunks, then set-context.

        This is numerically the same inference graph as ``forward`` when BN is
        in eval mode. The label-blind Morlet transform is computed per chunk
        and detached, avoiding a full C×224×60k image allocation.
        """
        if waveforms.ndim != 3 or waveforms.shape[0] != 1 or \
                waveforms.shape[:2] != channel_mask.shape or \
                descriptors.shape[:2] != channel_mask.shape or \
                descriptor_mask.shape != descriptors.shape:
            raise ValueError("Expected one [1,C,T] record and aligned descriptors")
        if channel_chunk < 1:
            raise ValueError("channel_chunk must be positive")
        selected = torch.nonzero(channel_mask[0].bool(), as_tuple=False).flatten()
        if selected.numel() == 0:
            raise ValueError("No valid channel")
        logits_chunks, embedding_chunks = [], []
        for indices in _channel_groups(selected, channel_chunk):
            image = preprocessor(waveforms[0, indices], sampling_rate)
            if not physiology and not context:
                logits_chunks.append(self.raw(image).squeeze(-1))
                continue
            hidden = self.raw.feature_extractor(image)
            if physiology:
                d = descriptors[0, indices] * descriptor_mask[0, indices]
                # Three source-short ictal records have fewer than 59 observed
                # windows. Never interpolate a fabricated zero-padded tail.
                active = descriptor_mask[0, indices].any(dim=(0, 2))
                n_window = int(active.sum())
                if n_window < 1 or not bool(active[:n_window].all()):
                    raise RuntimeError("Observed descriptor windows are not contiguous")
                d = d[:, :n_window]
                d = F.interpolate(d.transpose(1, 2), size=hidden.shape[-1],
                                  mode="linear", align_corners=False).transpose(1, 2)
                gamma, beta = self.physiology(d).chunk(2, dim=-1)
                scales = self.alphas()
                hidden = hidden * (1 + scales["gamma"] * torch.tanh(
                    gamma.transpose(1, 2).unsqueeze(2))) + scales["beta"] * \
                    torch.tanh(beta.transpose(1, 2).unsqueeze(2))
            if checkpoint_backbone and self.training and torch.is_grad_enabled():
                embedding = checkpoint(self.raw.cnn, hidden, use_reentrant=False)
            else:
                embedding = self.raw.cnn(hidden)
            embedding_chunks.append(embedding)
        if logits_chunks:
            scores = torch.cat(logits_chunks)
        else:
            embedding = torch.cat(embedding_chunks)
            if context:
                all_embeddings = embedding.new_zeros((1, channel_mask.shape[1], 32))
                all_embeddings[0, selected] = embedding
                normalized = self.context_norm(all_embeddings)
                attended, _ = self.context_mha(
                    normalized, normalized, normalized,
                    key_padding_mask=~channel_mask.bool(), need_weights=False)
                embedding = embedding + self.alphas()["context"] * \
                    self.context_out(attended[0, selected])
            scores = self._classifier(embedding)
        output = scores.new_zeros(channel_mask.shape)
        output[0, selected] = scores
        return output
