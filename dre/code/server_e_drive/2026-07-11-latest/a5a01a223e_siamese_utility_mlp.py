from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from ..feature_blocks import BlockScalers, PairFeatureBlocks
from ..utils import set_random_seed, write_json
from .training import PatientBalancedSampler, grouped_train_validation_split, resolve_device


class _ExplicitSiameseNetwork:
    def __init__(self, eject_dim: int, add_dim: int, pair_dim: int, patient_dim: int, hidden_dim: int, dropout: float):
        import torch.nn as nn
        if eject_dim != add_dim:
            raise ValueError("shared channel encoder needs matching eject/add dimensions")
        self.module = nn.Module()
        self.module.channel_encoder = nn.Sequential(nn.Linear(eject_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        extra_dim = pair_dim + patient_dim
        self.module.context_encoder = nn.Sequential(nn.Linear(max(1, extra_dim), hidden_dim), nn.GELU())
        self.module.head = nn.Sequential(nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))
        self.extra_dim = extra_dim

    def __call__(self, eject, add, pair, patient):
        import torch
        z_eject, z_add = self.module.channel_encoder(eject), self.module.channel_encoder(add)
        extra = torch.cat([pair, patient], dim=1)
        if self.extra_dim == 0:
            extra = torch.zeros((len(eject), 1), device=eject.device, dtype=eject.dtype)
        context = self.module.context_encoder(extra)
        return self.module.head(torch.cat([z_eject, z_add, z_add - z_eject, torch.abs(z_add - z_eject), z_eject * z_add, context], dim=1))


class SiameseUtilityMLP:
    """Patient-grouped mini-batch Siamese utility model with best restore."""
    def __init__(self, *, hidden_dim: int = 32, dropout: float = .2, epochs: int = 40,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 risk_lambda: float = .10, random_seed: int = 42, device: str = "cpu",
                 strict_device: bool = False, batch_size: int = 32, num_workers: int = 0,
                 patience: int = 8, gradient_clip: float = 1., output_dir: str | Path | None = None) -> None:
        self.hidden_dim, self.dropout, self.epochs = hidden_dim, dropout, epochs
        self.learning_rate, self.weight_decay, self.risk_lambda = learning_rate, weight_decay, risk_lambda
        self.random_seed, self.device, self.strict_device = random_seed, device, strict_device
        self.batch_size, self.num_workers, self.patience, self.gradient_clip = batch_size, num_workers, patience, gradient_clip
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.network: _ExplicitSiameseNetwork | None = None
        self.feature_columns: list[str] = []
        self.blocks: PairFeatureBlocks | None = None
        self.scalers: BlockScalers | None = None
        self.fit_subjects: set[str] = set(); self.validation_subjects: set[str] = set()
        self.best_epoch = 0; self.training_log = pd.DataFrame()
        self.device_resolution = resolve_device(device, strict=strict_device)
        self.resolved_device = self.device_resolution.resolved_device

    @staticmethod
    def _legacy_blocks(frame: pd.DataFrame, columns: list[str]) -> PairFeatureBlocks:
        return PairFeatureBlocks.infer(frame, columns) if any(name.startswith(("eject_", "add_")) for name in columns) else PairFeatureBlocks(tuple(columns), tuple(columns), tuple(), tuple())

    def _tensors(self, frame: pd.DataFrame, *, device: str | None = None):
        import torch
        if self.blocks is None or self.scalers is None:
            raise RuntimeError("model blocks/scalers are unavailable")
        target = device or self.resolved_device
        return tuple(torch.as_tensor(self.scalers.transform(frame, name), dtype=torch.float32, device=target) for name in ("eject", "add", "pair", "patient"))

    @staticmethod
    def _loss(output, benefit, harm, delta, weights):
        import torch.nn.functional as functional
        losses = functional.binary_cross_entropy_with_logits(output[:, 0], benefit, reduction="none") + functional.binary_cross_entropy_with_logits(output[:, 1], harm, reduction="none") + functional.huber_loss(output[:, 2], delta, reduction="none", delta=.05)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def fit(self, pairs: pd.DataFrame, feature_columns: list[str]) -> "SiameseUtilityMLP":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        required = {"subject_id", "beneficial_label", "harmful_label", "delta_patient_macro_f1"}
        if required - set(pairs):
            raise ValueError(f"pair labels missing: {sorted(required - set(pairs))}")
        set_random_seed(self.random_seed)
        train_index, validation_index = grouped_train_validation_split(pairs.reset_index(drop=True), seed=self.random_seed)
        source = pairs.reset_index(drop=True); train = source.iloc[train_index]; validation = source.iloc[validation_index]
        self.feature_columns = list(feature_columns)
        self.fit_subjects = set(train["subject_id"].astype(str)); self.validation_subjects = set(validation["subject_id"].astype(str))
        self.blocks = self._legacy_blocks(source, self.feature_columns); self.scalers = BlockScalers(self.blocks).fit(train)
        train_tensors = self._tensors(train)
        labels = (torch.as_tensor(train["beneficial_label"].to_numpy(float), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(train["harmful_label"].to_numpy(float), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(train["delta_patient_macro_f1"].to_numpy(float), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(train.get("patient_pair_weight", pd.Series(1., index=train.index)).to_numpy(float), dtype=torch.float32, device=self.resolved_device))
        self.network = _ExplicitSiameseNetwork(train_tensors[0].shape[1], train_tensors[1].shape[1], train_tensors[2].shape[1], train_tensors[3].shape[1], self.hidden_dim, self.dropout)
        self.network.module.to(self.resolved_device)
        dataset = TensorDataset(*train_tensors, *labels)
        sampler = PatientBalancedSampler(train["subject_id"], num_samples=len(train), seed=self.random_seed)
        loader = DataLoader(dataset, batch_size=max(1, self.batch_size), sampler=sampler, num_workers=max(0, self.num_workers))
        optimizer = torch.optim.AdamW(self.network.module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        validation_tensors = self._tensors(validation) if len(validation) else None
        best_loss, best_state, stale, rows = float("inf"), None, 0, []
        for epoch in range(1, max(1, self.epochs) + 1):
            self.network.module.train(); losses = []
            for batch in loader:
                output = self.network(*batch[:4]); loss = self._loss(output, *batch[4:])
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), self.gradient_clip); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.network.module.eval()
            if validation_tensors is not None:
                with torch.no_grad():
                    val_output = self.network(*validation_tensors)
                    val_labels = (torch.as_tensor(validation["beneficial_label"].to_numpy(float), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(validation["harmful_label"].to_numpy(float), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(validation["delta_patient_macro_f1"].to_numpy(float), dtype=torch.float32, device=self.resolved_device), torch.as_tensor(validation.get("patient_pair_weight", pd.Series(1., index=validation.index)).to_numpy(float), dtype=torch.float32, device=self.resolved_device))
                    validation_loss = float(self._loss(val_output, *val_labels).detach().cpu())
            else:
                validation_loss = float(np.mean(losses))
            rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-9:
                best_loss, best_state, self.best_epoch, stale = validation_loss, copy.deepcopy(self.network.module.state_dict()), epoch, 0
            else:
                stale += 1
                if stale >= max(1, self.patience):
                    break
        if best_state is not None:
            self.network.module.load_state_dict(best_state)
        self.training_log = pd.DataFrame(rows)
        self._write_artifacts()
        return self

    def _write_artifacts(self) -> None:
        if self.output_dir is None or self.network is None:
            return
        import torch
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_log.to_csv(self.output_dir / f"seed_{self.random_seed}_training_log.csv", index=False)
        torch.save({"state_dict": self.network.module.state_dict(), "feature_columns": self.feature_columns, "best_epoch": self.best_epoch}, self.output_dir / f"seed_{self.random_seed}_checkpoint.pt")
        write_json(self.output_dir / f"seed_{self.random_seed}_fit_manifest.json", {"fit_subjects": sorted(self.fit_subjects), "validation_subjects": sorted(self.validation_subjects), "best_epoch": self.best_epoch, "resolved_device": self.resolved_device, "n_parameters": sum(parameter.numel() for parameter in self.network.module.parameters()), "training_config": {"batch_size": self.batch_size, "max_epochs": self.epochs, "patience": self.patience, "learning_rate": self.learning_rate, "weight_decay": self.weight_decay, "gradient_clip": self.gradient_clip}, **self.device_resolution.to_dict()})

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        import torch
        if self.network is None:
            raise RuntimeError("model must be fit before predict")
        self.network.module.eval()
        with torch.no_grad():
            output = self.network(*self._tensors(pairs)).detach().cpu().numpy()
        benefit, harm, delta = 1. / (1. + np.exp(-output[:, 0])), 1. / (1. + np.exp(-output[:, 1])), output[:, 2]
        return pd.DataFrame({"p_benefit": benefit, "p_harm": harm, "pred_delta": delta, "utility": benefit * np.maximum(delta, 0.) - self.risk_lambda * harm}, index=pairs.index)
