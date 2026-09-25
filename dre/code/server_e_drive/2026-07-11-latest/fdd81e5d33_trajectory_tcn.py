from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from ..feature_blocks import BlockScalers, PairFeatureBlocks
from ..utils import set_random_seed, write_json
from .training import PatientBalancedSampler, grouped_train_validation_split, resolve_device


class _ResidualTemporalBlock:
    def __init__(self, channels: int, dilation: int, dropout: float):
        import torch.nn as nn

        padding = dilation
        self.module = nn.Module()
        self.module.depthwise = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, groups=channels)
        self.module.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.module.norm = nn.GroupNorm(1, channels)
        self.module.activation = nn.GELU()
        self.module.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        return x + self.module.dropout(self.module.activation(self.module.norm(self.module.pointwise(self.module.depthwise(x)))))


class TrajectoryTCNEncoder:
    """Trainable masked temporal feature encoder with dilated residual depthwise convolutions."""

    def __init__(self, *, feature_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        import torch.nn as nn

        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.module = nn.Module()
        self.module.projection = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = [_ResidualTemporalBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4)]
        for index, block in enumerate(self.blocks):
            self.module.add_module(f"temporal_block_{index}", block.module)

    def forward(self, trajectories, window_mask, seizure_mask=None):
        import torch

        values = torch.as_tensor(trajectories, dtype=torch.float32) if not isinstance(trajectories, torch.Tensor) else trajectories
        window_mask = torch.as_tensor(window_mask, dtype=torch.bool, device=values.device) if not isinstance(window_mask, torch.Tensor) else window_mask.to(device=values.device, dtype=torch.bool)
        if values.ndim == 3:  # legacy [B,T,F] is one seizure, never a flattened seizure axis.
            values, window_mask = values[:, None, :, :], window_mask[:, None, :]
        if values.ndim != 4 or window_mask.shape != values.shape[:3]:
            raise ValueError("trajectories must be [B,S,T,F] with window mask [B,S,T]")
        if seizure_mask is None: seizure_mask = window_mask.any(dim=-1)
        elif not isinstance(seizure_mask, torch.Tensor): seizure_mask = torch.as_tensor(seizure_mask, dtype=torch.bool, device=values.device)
        if seizure_mask.shape != values.shape[:2]: raise ValueError("seizure_mask must be [B,S]")
        batch, seizures, windows, features = values.shape
        valid = window_mask.bool() & seizure_mask.bool().unsqueeze(-1)
        # The only structural folding is B*S. Each temporal sequence is trimmed
        # to valid windows before convolution so padded length cannot affect norm.
        packed_values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).reshape(batch * seizures, windows, features)
        packed_valid = valid.reshape(batch * seizures, windows)
        encoded = []
        for sequence, sequence_mask in zip(packed_values, packed_valid):
            if not bool(sequence_mask.any()):
                encoded.append(torch.zeros(self.hidden_dim * 6, device=values.device, dtype=values.dtype))
                continue
            x = sequence[sequence_mask].transpose(0, 1).unsqueeze(0)
            x = self.module.projection(x)
            for block in self.blocks:
                x = block(x)
            x = x.squeeze(0).transpose(0, 1)
            mean, maximum, first, last = x.mean(dim=0), x.max(dim=0).values, x[0], x[-1]
            variation = torch.abs(x[1:] - x[:-1]).mean(dim=0) if len(x) > 1 else torch.zeros_like(mean)
            encoded.append(torch.cat([mean, maximum, first, last, last - first, variation]))
        seizure_features = torch.stack(encoded).reshape(batch, seizures, -1)
        valid_seizures = seizure_mask.bool().unsqueeze(-1)
        n_seizures = seizure_mask.sum(dim=1, keepdim=True).clamp_min(1)
        sz_mean = (seizure_features * valid_seizures).sum(dim=1) / n_seizures
        variance = ((seizure_features - sz_mean[:, None, :]) ** 2 * valid_seizures).sum(dim=1) / n_seizures
        sz_std = torch.sqrt(variance.clamp_min(0.0) + 1e-8)
        sz_max_raw = seizure_features.masked_fill(~valid_seizures, -1e4).max(dim=1).values
        sz_max = torch.where(seizure_mask.any(dim=1, keepdim=True), sz_max_raw, torch.zeros_like(sz_max_raw))
        attention_logits = seizure_features.mean(dim=-1).masked_fill(~seizure_mask.bool(), -1e4)
        attention = torch.softmax(attention_logits, dim=1) * seizure_mask.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        gated = (seizure_features * attention.unsqueeze(-1)).sum(dim=1)
        coverage = seizure_mask.float().mean(dim=1, keepdim=True)
        pooled = torch.cat([sz_mean, sz_std, sz_max, gated, n_seizures.float(), coverage], dim=1)
        return torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)

    def encode(self, trajectories: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        import torch

        self.module.eval()
        with torch.no_grad():
            values = torch.as_tensor(np.asarray(trajectories, dtype=np.float32), dtype=torch.float32)
            valid = torch.as_tensor(np.asarray(mask, dtype=bool), dtype=torch.bool)
            # Public diagnostic API exposes the masked-mean channel embedding;
            # the pair model consumes the richer pooled representation in ``forward``.
            embedding = self.forward(values, valid)[:, : self.hidden_dim]
        return embedding.cpu().numpy(), {"valid_windows": int(valid.sum().item()), "padded_values_used": 0}


class _TrajectoryPairNetwork:
    def __init__(self, temporal_feature_dim: int, pair_dim: int, patient_dim: int, hidden_dim: int, dropout: float):
        import torch.nn as nn

        self.module = nn.Module()
        self.encoder = TrajectoryTCNEncoder(feature_dim=temporal_feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.module.encoder = self.encoder.module
        context_dim = pair_dim + patient_dim
        self.module.context = nn.Sequential(nn.Linear(max(1, context_dim), hidden_dim), nn.GELU())
        self.channel_embedding_dim = hidden_dim * 24 + 2
        self.module.head = nn.Sequential(nn.Linear(self.channel_embedding_dim * 5 + hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))
        self.context_dim = context_dim

    def __call__(self, eject_values, eject_mask, eject_seizure_mask, add_values, add_mask, add_seizure_mask, pair, patient):
        import torch

        z_eject = self.encoder.forward(eject_values, eject_mask, eject_seizure_mask)
        z_add = self.encoder.forward(add_values, add_mask, add_seizure_mask)
        extra = torch.cat([pair, patient], dim=1)
        if self.context_dim == 0:
            extra = torch.zeros((len(pair), 1), device=pair.device, dtype=pair.dtype)
        z_context = self.module.context(extra)
        # Encoder summaries each have 4*hidden dimensions; order is meaningful.
        interactions = torch.cat([z_eject, z_add, z_add - z_eject, torch.abs(z_add - z_eject), z_eject * z_add, z_context], dim=1)
        interactions = torch.nan_to_num(interactions, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.nan_to_num(self.module.head(interactions), nan=0.0, posinf=1e4, neginf=-1e4)


class TrajectoryTCNUtilityModel:
    """Real masked temporal TCN pair utility model; it does not inherit Siamese."""

    def __init__(self, *, hidden_dim: int = 32, dropout: float = 0.2, epochs: int = 40, learning_rate: float = 1e-3, weight_decay: float = 1e-4, risk_lambda: float = 0.10, random_seed: int = 42, device: str = "cpu", strict_device: bool = False, batch_size: int = 32, num_workers: int = 0, patience: int = 8, gradient_clip: float = 1.0, output_dir=None, trajectory_store=None) -> None:
        self.hidden_dim, self.dropout, self.epochs, self.learning_rate, self.weight_decay, self.risk_lambda, self.random_seed, self.device = hidden_dim, dropout, epochs, learning_rate, weight_decay, risk_lambda, random_seed, device
        self.strict_device, self.batch_size, self.num_workers, self.patience, self.gradient_clip, self.output_dir = strict_device, batch_size, num_workers, patience, gradient_clip, output_dir
        self.device_resolution = resolve_device(device, strict=strict_device)
        self.resolved_device = self.device_resolution.resolved_device
        self.network: _TrajectoryPairNetwork | None = None
        self.feature_columns: list[str] = []
        self.blocks: PairFeatureBlocks | None = None
        self.scalers: BlockScalers | None = None
        self.fit_subjects: set[str] = set()
        self.validation_subjects: set[str] = set()
        self.trajectory_store = trajectory_store
        self.best_epoch = 0
        self.training_log = pd.DataFrame()

    @staticmethod
    def _blocks(frame: pd.DataFrame, columns: list[str]) -> PairFeatureBlocks:
        if any(name.startswith(("eject_", "add_")) for name in columns):
            return PairFeatureBlocks.infer(frame, columns)
        return PairFeatureBlocks(tuple(columns), tuple(columns), tuple(), tuple())

    def _trajectory(self, frame: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value_column, mask_column = f"{prefix}_trajectory_values", f"{prefix}_trajectory_mask"
        if value_column in frame:
            values = [np.asarray(item, dtype=np.float32) for item in frame[value_column]]
            masks = [np.asarray(item, dtype=bool) if mask_column in frame else np.ones(np.asarray(item).shape[:2], dtype=bool) for item in frame.get(mask_column, values)]
            # A pair item is [seizure, windows, feature]; aggregate seizures only after TCN using masked mean.
            sequences, sequence_masks = [], []
            for value, mask in zip(values, masks):
                if value.ndim == 2: value, mask = value[None, :, :], mask[None, :]
                if value.ndim != 3 or mask.shape != value.shape[:2]: raise ValueError("trajectory values must be [S,T,F] with [S,T] mask")
                sequences.append(value); sequence_masks.append(mask)
            max_s, max_t, feature_dim = max(item.shape[0] for item in sequences), max(item.shape[1] for item in sequences), sequences[0].shape[2]
            output, output_mask, seizure_mask = np.full((len(frame), max_s, max_t, feature_dim), np.nan, np.float32), np.zeros((len(frame), max_s, max_t), bool), np.zeros((len(frame), max_s), bool)
            for index, (sequence, mask) in enumerate(zip(sequences, sequence_masks)):
                output[index, :sequence.shape[0], :sequence.shape[1]] = sequence; output_mask[index, :mask.shape[0], :mask.shape[1]] = mask; seizure_mask[index, :mask.shape[0]] = mask.any(axis=1)
            return output, output_mask, seizure_mask
        if self.trajectory_store is not None:
            channel_column = f"{prefix}_channel_norm" if f"{prefix}_channel_norm" in frame else f"{prefix}_channel"
            values, masks = [], []
            for _, row in frame.iterrows():
                value, mask, _ = self.trajectory_store.get_channel_trajectory(str(row["subject_id"]), str(row[channel_column]))
                values.append(value); masks.append(mask)
            max_s, max_t, feature_dim = max(item.shape[0] for item in values), max(item.shape[1] for item in values), values[0].shape[2]
            output, output_mask, seizure_mask = np.full((len(frame), max_s, max_t, feature_dim), np.nan, np.float32), np.zeros((len(frame), max_s, max_t), bool), np.zeros((len(frame), max_s), bool)
            for index, (value, mask) in enumerate(zip(values, masks)):
                output[index, :value.shape[0], :value.shape[1]] = value; output_mask[index, :mask.shape[0], :mask.shape[1]] = mask; seizure_mask[index, :mask.shape[0]] = mask.any(axis=1)
            return output, output_mask, seizure_mask
        raise RuntimeError("real TCN trajectories are required; static fallback is forbidden")

    def _tensors(self, frame: pd.DataFrame):
        import torch

        if self.scalers is None:
            raise RuntimeError("model scalers are unavailable")
        eject, eject_mask, eject_seizures = self._trajectory(frame, "eject")
        add, add_mask, add_seizures = self._trajectory(frame, "add")
        device = self.resolved_device
        return (torch.as_tensor(eject, dtype=torch.float32, device=device), torch.as_tensor(eject_mask, dtype=torch.bool, device=device), torch.as_tensor(eject_seizures, dtype=torch.bool, device=device), torch.as_tensor(add, dtype=torch.float32, device=device), torch.as_tensor(add_mask, dtype=torch.bool, device=device), torch.as_tensor(add_seizures, dtype=torch.bool, device=device), torch.as_tensor(self.scalers.transform(frame, "pair"), dtype=torch.float32, device=device), torch.as_tensor(self.scalers.transform(frame, "patient"), dtype=torch.float32, device=device))

    def fit(self, pairs: pd.DataFrame, feature_columns: list[str]) -> "TrajectoryTCNUtilityModel":
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset

        required = {"subject_id", "beneficial_label", "harmful_label", "delta_patient_macro_f1"}
        missing = required - set(pairs.columns)
        if missing: raise ValueError(f"pair labels missing: {sorted(missing)}")
        set_random_seed(self.random_seed)
        source = pairs.reset_index(drop=True)
        train_index, validation_index = grouped_train_validation_split(source, seed=self.random_seed)
        train, validation = source.iloc[train_index], source.iloc[validation_index]
        self.feature_columns = list(feature_columns)
        self.fit_subjects = set(train["subject_id"].astype(str)); self.validation_subjects = set(validation["subject_id"].astype(str))
        self.blocks = self._blocks(source, self.feature_columns); self.scalers = BlockScalers(self.blocks).fit(train)
        train_tensors = self._tensors(train)
        eject, _, _, _, _, _, pair, patient = train_tensors
        self.network = _TrajectoryPairNetwork(eject.shape[3], pair.shape[1], patient.shape[1], self.hidden_dim, self.dropout)
        self.network.module.to(self.resolved_device)
        optimizer = torch.optim.AdamW(self.network.module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        targets = (torch.as_tensor(train["beneficial_label"].astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(train["harmful_label"].astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(train["delta_patient_macro_f1"].astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(train.get("patient_pair_weight", pd.Series(1., index=train.index)).astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device))
        dataset = TensorDataset(*train_tensors, *targets)
        loader = DataLoader(dataset, batch_size=max(1, self.batch_size), sampler=PatientBalancedSampler(train["subject_id"], num_samples=len(train), seed=self.random_seed), num_workers=max(0, self.num_workers))
        validation_tensors = self._tensors(validation) if len(validation) else None
        best_loss, best_state, stale, rows = float("inf"), None, 0, []
        def loss_fn(output, benefit, harm, delta, weights):
            losses = functional.binary_cross_entropy_with_logits(output[:, 0], benefit, reduction="none") + functional.binary_cross_entropy_with_logits(output[:, 1], harm, reduction="none") + functional.huber_loss(output[:, 2], delta, reduction="none", delta=.05)
            return (losses * weights).sum() / weights.sum().clamp_min(1e-8)
        for epoch in range(1, max(1, self.epochs) + 1):
            self.network.module.train(); train_losses = []
            for batch in loader:
                output = self.network(*batch[:8]); loss = loss_fn(output, *batch[8:])
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), self.gradient_clip); optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            self.network.module.eval()
            if validation_tensors is not None:
                validation_targets = (torch.as_tensor(validation["beneficial_label"].astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(validation["harmful_label"].astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(validation["delta_patient_macro_f1"].astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(validation.get("patient_pair_weight", pd.Series(1., index=validation.index)).astype(float).to_numpy(), dtype=torch.float32, device=self.resolved_device))
                with torch.no_grad(): validation_loss = float(loss_fn(self.network(*validation_tensors), *validation_targets).detach().cpu())
            else:
                validation_loss = float(np.mean(train_losses))
            rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-9:
                best_loss, best_state, self.best_epoch, stale = validation_loss, copy.deepcopy(self.network.module.state_dict()), epoch, 0
            else:
                stale += 1
                if stale >= max(1, self.patience): break
        if best_state is not None: self.network.module.load_state_dict(best_state)
        if not all(torch.isfinite(parameter).all() for parameter in self.network.module.parameters()):
            raise RuntimeError("TCN training produced non-finite parameters")
        self.training_log = pd.DataFrame(rows)
        if self.output_dir is not None:
            output_dir = Path(self.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
            self.training_log.to_csv(output_dir / f"seed_{self.random_seed}_training_log.csv", index=False)
            torch.save({"state_dict": self.network.module.state_dict(), "feature_columns": self.feature_columns, "best_epoch": self.best_epoch}, output_dir / f"seed_{self.random_seed}_checkpoint.pt")
            write_json(output_dir / f"seed_{self.random_seed}_fit_manifest.json", {"fit_subjects": sorted(self.fit_subjects), "validation_subjects": sorted(self.validation_subjects), "best_epoch": self.best_epoch, "resolved_device": self.resolved_device, "n_parameters": sum(parameter.numel() for parameter in self.network.module.parameters()), "training_config": {"batch_size": self.batch_size, "max_epochs": self.epochs, "patience": self.patience, "learning_rate": self.learning_rate, "weight_decay": self.weight_decay, "gradient_clip": self.gradient_clip}, **self.device_resolution.to_dict()})
        return self

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        import torch

        if self.network is None: raise RuntimeError("model must be fit before predict")
        self.network.module.eval()
        with torch.no_grad(): output = self.network(*self._tensors(pairs)).detach().cpu().numpy()
        benefit, harm, delta = 1 / (1 + np.exp(-output[:, 0])), 1 / (1 + np.exp(-output[:, 1])), output[:, 2]
        return pd.DataFrame({"p_benefit": benefit, "p_harm": harm, "pred_delta": delta, "utility": benefit * np.maximum(delta, 0.) - self.risk_lambda * harm}, index=pairs.index)
