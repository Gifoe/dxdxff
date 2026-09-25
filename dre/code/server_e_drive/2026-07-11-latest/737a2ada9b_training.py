from __future__ import annotations

import random
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import NPAMCollator
from .masks import apply_channel_dropout, apply_seizure_dropout
from .functional_graph import NETWORK_PHASES
from .p2_q10_npam_loss import compute_npam_loss

try:
    from tqdm.auto import trange
except ImportError:  # pragma: no cover - exercised only in minimal server envs
    trange = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class NPAMSystem(nn.Module):
    def __init__(self, p2_adapter: nn.Module, outcome_model: nn.Module) -> None:
        super().__init__()
        self.p2_adapter = p2_adapter
        self.outcome_model = outcome_model

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        p2 = self.p2_adapter(batch)
        graphs = None
        if "graph_adjacency" in batch:
            graphs = {
                "adjacency": batch["graph_adjacency"].to(p2["seizure_channel_embedding"].device),
                "phase_channel_mask": batch["graph_phase_channel_mask"].to(p2["seizure_channel_embedding"].device),
                "graph_valid": batch["graph_valid"].to(p2["seizure_channel_embedding"].device),
            }
        if self.outcome_model.__class__.__name__ == "P2TargetOutcomeModel":
            output = self.outcome_model(p2, graphs, batch.get("clinical_target_mask"))
        else:
            output = self.outcome_model(p2, graphs)
        output["final_nez_logit"] = p2["final_nez_logit"]
        return output


def prepare_examples(runtime: Any, cache: Mapping[str, Any], args: Any, *, normalizer_subjects: Sequence[str], output_subjects: Sequence[str]) -> tuple[list[dict[str, Any]], Any]:
    train_samples = runtime.flatten_window_samples(cache["run_records"], normalizer_subjects)
    normalizer = runtime.fit_normalizer(train_samples, args)
    output_samples = runtime.flatten_window_samples(cache["run_records"], output_subjects)
    examples = runtime.build_examples(output_samples, cache["patient_index"], normalizer=normalizer, subject_ids=output_subjects, args=args)
    if set(map(str, output_subjects)) != {str(example["subject_id"]) for example in examples}:
        missing = sorted(set(map(str, output_subjects)) - {str(example["subject_id"]) for example in examples})
        raise ValueError(f"Failed to build P2 patient examples for: {missing[:20]}")
    return examples, normalizer


def make_loader(examples: Sequence[dict[str, Any]], runtime: Any, outcome_by_subject: Mapping[str, int], graph_lookup: Mapping | None, *, batch_size: int, shuffle: bool, seed: int, clinical_target_lookup: Mapping | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(list(examples), batch_size=int(batch_size), shuffle=shuffle, generator=generator, collate_fn=NPAMCollator(runtime, outcome_by_subject, graph_lookup, clinical_target_lookup), num_workers=0)


def _move_training_fields(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for key in ("outcome_target", "labels_nez", "localization_mask", "clinical_target_mask", "graph_adjacency", "graph_phase_channel_mask", "graph_valid"):
        if torch.is_tensor(output.get(key)):
            output[key] = output[key].to(device)
    return output


def augment_batch(batch: dict[str, Any], *, channel_dropout: float, seizure_dropout: float, generator: torch.Generator) -> dict[str, Any]:
    output = dict(batch)
    seizure_mask = apply_seizure_dropout(batch["seizure_mask"], seizure_dropout, generator=generator)
    channel_mask = apply_channel_dropout(batch["seizure_channel_mask"], channel_dropout, generator=generator, minimum=4)
    channel_mask &= seizure_mask.unsqueeze(-1)
    output["seizure_mask"] = seizure_mask
    output["seizure_channel_mask"] = channel_mask
    if "graph_phase_channel_mask" in output:
        output["graph_phase_channel_mask"] = output["graph_phase_channel_mask"] & channel_mask.unsqueeze(2)
        output["graph_valid"] = output["graph_valid"] & (output["graph_phase_channel_mask"].sum(dim=-1) >= 4)
    return output


def class_pos_weight(labels: Sequence[int]) -> float:
    values = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(values == 1))
    negatives = int(np.sum(values == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Outcome training split must contain success and failure patients")
    return float(negatives / positives)


def fit_target_scalar_normalizer(system: NPAMSystem, loader: DataLoader, device: str | torch.device) -> None:
    if not hasattr(system.outcome_model, "fit_scalar_normalizer"):
        return
    target_device=torch.device(device); system.eval(); rows=[]
    with torch.no_grad():
        for batch in loader:
            output=system(_move_training_fields(batch,target_device)); rows.append(output["patient_scalar_features"].detach())
    system.outcome_model.fit_scalar_normalizer(torch.cat(rows,dim=0))


def train_seizure_probe_fixed_epochs(system: NPAMSystem, loader: DataLoader, *, device: str | torch.device, epochs: int, seed: int, progress_checkpoint: str | Path | None = None, resume: bool = True, resume_signature: str = "") -> pd.DataFrame:
    probe=getattr(system.outcome_model,"seizure_target_probe",None)
    if probe is None: return pd.DataFrame(columns=("epoch","train_loss","role"))
    for parameter in system.parameters(): parameter.requires_grad=False
    for parameter in probe.parameters(): parameter.requires_grad=True
    optimizer=torch.optim.AdamW(probe.parameters(),lr=5e-4,weight_decay=1e-4); target_device=torch.device(device); system.to(target_device)
    path=Path(progress_checkpoint) if progress_checkpoint else None; rows=[]; start=1; total=max(1,int(epochs))
    if resume and path is not None and path.exists():
        payload=torch.load(path,map_location=target_device,weights_only=False)
        if payload.get("resume_signature")==resume_signature and payload.get("total_epochs")==total:
            probe.load_state_dict(payload["probe_state_dict"]); optimizer.load_state_dict(payload["optimizer_state_dict"]); rows=list(payload.get("history",[])); start=int(payload.get("completed_epoch",0))+1
    iterator=trange(start,total+1,initial=start-1,total=total,desc="seizure target probe",dynamic_ncols=True,ascii=True) if trange is not None else range(start,total+1)
    from .seizure_target_probe import patient_balanced_probe_loss
    for epoch in iterator:
        probe.train(); losses=[]
        for batch in loader:
            moved=_move_training_fields(batch,target_device); optimizer.zero_grad(set_to_none=True); output=system(moved)
            loss=patient_balanced_probe_loss(output["seizure_target_logit"],moved["clinical_target_mask"],output["seizure_target_valid"])
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite seizure target probe loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(probe.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        mean=float(np.mean(losses)); rows.append({"epoch":epoch,"train_loss":mean,"role":"outer_train_seizure_target_probe"})
        if trange is not None: iterator.set_postfix(loss=f"{mean:.6f}")
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True); payload={"probe_state_dict":probe.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"completed_epoch":epoch,"total_epochs":total,"history":rows,"resume_signature":resume_signature}; temporary=path.with_suffix(path.suffix+".tmp"); torch.save(payload,temporary); temporary.replace(path)
    for parameter in system.outcome_model.parameters(): parameter.requires_grad=True
    for parameter in probe.parameters(): parameter.requires_grad=False
    for parameter in system.p2_adapter.parameters(): parameter.requires_grad=False
    return pd.DataFrame(rows)


def train_fixed_epochs(
    system: NPAMSystem,
    train_loader: DataLoader,
    *,
    device: str | torch.device,
    pos_weight: float,
    epochs: int,
    seed: int,
    head_lr: float = 1e-3,
    p2_lr: float = 5e-5,
    weight_decay: float = 1e-4,
    channel_dropout: float = 0.10,
    seizure_dropout: float = 0.20,
    lambda_loc: float = 0.20,
    lambda_residual: float = 1e-4,
    progress_prefix: str = "train",
    progress_checkpoint: str | Path | None = None,
    resume: bool = True,
    resume_signature: str = "",
) -> pd.DataFrame:
    target_device = torch.device(device)
    system.to(target_device)
    head = [parameter for name, parameter in system.named_parameters() if parameter.requires_grad and not name.startswith("p2_adapter.backbone")]
    p2 = [parameter for name, parameter in system.named_parameters() if parameter.requires_grad and name.startswith("p2_adapter.backbone")]
    p2_initial = {name: parameter.detach().clone() for name, parameter in system.named_parameters() if parameter.requires_grad and name.startswith("p2_adapter.backbone")}
    groups = [{"params": head, "lr": head_lr}]
    if p2:
        groups.append({"params": p2, "lr": p2_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    checkpoint_path = Path(progress_checkpoint) if progress_checkpoint else None
    total_epochs = max(1, int(epochs))
    rows: list[dict[str, Any]] = []
    start_epoch = 1
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
        if payload.get("resume_signature", "") == resume_signature and int(payload.get("total_epochs", -1)) == total_epochs:
            system.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            rows = list(payload.get("history", []))
            start_epoch = int(payload.get("completed_epoch", 0)) + 1
            if payload.get("augmentation_generator_state") is not None:
                generator.set_state(payload["augmentation_generator_state"].cpu())
            if payload.get("p2_initial") is not None:
                p2_initial={name:value.to(target_device) for name,value in payload["p2_initial"].items()}
            loader_generator = getattr(train_loader, "generator", None)
            if loader_generator is not None and payload.get("loader_generator_state") is not None:
                loader_generator.set_state(payload["loader_generator_state"].cpu())
            print(f"[{progress_prefix}] resume from epoch {start_epoch}/{total_epochs}", flush=True)
        else:
            print(f"[{progress_prefix}] stale progress checkpoint ignored", flush=True)
    if start_epoch > total_epochs:
        print(f"[{progress_prefix}] already complete ({total_epochs}/{total_epochs})", flush=True)
        return pd.DataFrame(rows)
    epoch_iterator = (
        trange(start_epoch, total_epochs + 1, initial=start_epoch - 1, total=total_epochs, desc=progress_prefix, dynamic_ncols=True, ascii=True)
        if trange is not None
        else range(start_epoch, total_epochs + 1)
    )
    for epoch in epoch_iterator:
        system.train()
        losses = []
        for batch in train_loader:
            moved = _move_training_fields(augment_batch(batch, channel_dropout=channel_dropout, seizure_dropout=seizure_dropout, generator=generator), target_device)
            optimizer.zero_grad(set_to_none=True)
            output = system(moved)
            if "patient_scalar_features" in output:
                outcome_loss = F.binary_cross_entropy_with_logits(output["outcome_logit_success"], moved["outcome_target"])
                probe_loss = outcome_loss.new_zeros(())
                if "seizure_target_logit" in output:
                    from .seizure_target_probe import patient_balanced_probe_loss
                    probe_loss = patient_balanced_probe_loss(output["seizure_target_logit"], moved["clinical_target_mask"], output["seizure_target_valid"])
                regularization = outcome_loss.new_zeros(())
                if "seizure_target_logit" in output:
                    regularization = sum((parameter.square().mean() for parameter in system.outcome_model.channel_projection.parameters()), start=regularization)
                    regularization = sum(((parameter-p2_initial[name]).square().mean() for name,parameter in system.named_parameters() if name in p2_initial),start=regularization)
                parts = {"loss": outcome_loss + (.20 * probe_loss + 1e-4 * regularization if "seizure_target_logit" in output else 0.0), "outcome_loss": outcome_loss, "probe_loss": probe_loss, "regularization": regularization}
            else:
                parts = compute_npam_loss(output, moved["outcome_target"], pos_weight=pos_weight, localization_target_nez=moved["labels_nez"], localization_mask=moved["localization_mask"], lambda_loc=lambda_loc, lambda_residual=lambda_residual)
            if not torch.isfinite(parts["loss"]):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            debug_context = torch.autograd.detect_anomaly(check_nan=True) if os.getenv("DRE_DEBUG_AUTOGRAD") == "1" else nullcontext()
            with debug_context:
                parts["loss"].backward()
            trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
            invalid_gradients = [name for name, parameter in system.named_parameters() if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if invalid_gradients:
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}: {invalid_gradients[:20]}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            losses.append(float(parts["loss"].detach().cpu()))
        mean_loss = float(np.mean(losses))
        rows.append({"epoch": epoch, "train_loss": mean_loss, "role": "outer_refit_fixed_epoch"})
        if trange is not None:
            epoch_iterator.set_postfix(loss=f"{mean_loss:.6f}")
        else:
            print(f"[{progress_prefix}] epoch {epoch}/{total_epochs} loss={mean_loss:.6f}", flush=True)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            loader_generator = getattr(train_loader, "generator", None)
            payload = {
                "model_state_dict": system.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_epoch": int(epoch),
                "total_epochs": total_epochs,
                "history": rows,
                "resume_signature": resume_signature,
                "augmentation_generator_state": generator.get_state(),
                "loader_generator_state": loader_generator.get_state() if loader_generator is not None else None,
                "p2_initial": p2_initial,
            }
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
    return pd.DataFrame(rows)


def predict(system: NPAMSystem, loader: DataLoader, device: str | torch.device) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    target_device = torch.device(device)
    system.eval()
    patient_rows: list[dict[str, Any]] = []
    audits: dict[str, list[dict[str, Any]]] = {"patient": [], "seizure": [], "channel": [], "network": [], "phase": [], "gate": [], "feature": [], "probe": [], "q10": []}
    with torch.no_grad():
        for batch in loader:
            moved = _move_training_fields(batch, target_device)
            output = system(moved)
            probabilities = output["outcome_probability_success"].detach().cpu().numpy()
            targets = batch["outcome_target"].numpy()
            for patient_idx, subject in enumerate(batch["subject_id"]):
                valid_seizure = batch["seizure_mask"][patient_idx].bool()
                valid_channel = batch["channel_mask"][patient_idx].bool()
                if "graph_valid" in batch:
                    patient_graph_valid = batch["graph_valid"][patient_idx].bool()
                    graph_denominator = max(int(valid_seizure.sum()) * patient_graph_valid.shape[-1], 1)
                    graph_coverage = float((patient_graph_valid & valid_seizure.unsqueeze(-1)).sum() / graph_denominator)
                else:
                    graph_coverage = 0.0
                probability_success = float(probabilities[patient_idx])
                base_row={"patient_key": str(subject), "patient_id": str(subject), "center": str(batch["center"][patient_idx]), "outcome_true": int(targets[patient_idx]), "outcome_group": "success" if int(targets[patient_idx]) == 1 else "failure", "outcome_probability_success": probability_success, "outcome_probability_failure": 1.0 - probability_success, "outcome_pred_05": int(probability_success >= 0.5), "n_seizures": int(valid_seizure.sum()), "n_channels": int(valid_channel.sum()), "graph_coverage": graph_coverage}
                if "patient_scalar_features" in output:
                    feature_row=dict(base_row)
                    for name in output.get("patient_scalar_feature_names", ()):
                        feature_row[name]=float(output[name][patient_idx].detach().cpu())
                    for name in ("target_count","target_fraction","abnormality_mean","abnormality_top10_mean","coverage_mass","residual_mass","target_precision_mass","inside_outside_gap","top10_target_coverage","top10_target_miss","target_count_jaccard","target_count_ndcg","within_patient_target_auroc","within_patient_target_auprc"):
                        if name in output: feature_row[name]=float(output[name][patient_idx].detach().cpu())
                        elif name in ("target_count","target_fraction") and "clinical_target_mask" in batch:
                            local_target=batch["clinical_target_mask"][patient_idx][valid_channel]; feature_row[name]=float(local_target.sum() if name=="target_count" else local_target.mean())
                        else: feature_row[name]=0.0
                    for name in ("stable_target_coverage","stable_target_residual","coverage_across_seizures_mean","coverage_across_seizures_std","residual_abnormal_network_preictal","residual_abnormal_network_onset","residual_abnormal_network_spread","network_residual_ratio_spread","abnormal_hub_coverage_spread"):
                        feature_row.setdefault(name,float(output[name][patient_idx].detach().cpu()) if name in output else 0.0)
                    feature_row["target_mask_available"]=bool("clinical_target_mask" in batch)
                    audits["feature"].append(feature_row); patient_rows.append(feature_row)
                    if "seizure_target_probability" in output:
                        values=output["seizure_target_probability"][patient_idx][output["seizure_target_valid"][patient_idx]].detach().cpu()
                        valid_probe=output["seizure_target_valid"][patient_idx].detach().cpu(); target_probe=batch["clinical_target_mask"][patient_idx].unsqueeze(0).expand_as(valid_probe)[valid_probe].numpy(); score_probe=values.numpy()
                        if np.unique(target_probe).size==2:
                            from sklearn.metrics import roc_auc_score
                            probe_auroc=float(roc_auc_score(target_probe,score_probe))
                        else: probe_auroc=np.nan
                        audits["probe"].append({"patient_key":str(subject),"n_values":int(values.numel()),"probability_mean":float(values.mean()),"probability_std":float(values.std(unbiased=False)),"target_auroc":probe_auroc})
                    audits["q10"].append({"patient_key":str(subject),"old_q10_std":float(output["old_q10_std"][patient_idx].cpu()),"old_q10_min":float(output["old_q10_min"][patient_idx].cpu()),"old_q10_max":float(output["old_q10_max"][patient_idx].cpu()),"new_q10_std":float(output["new_q10_std"][patient_idx].cpu()) if "new_q10_std" in output else 0.0,"new_q10_min":float(output["new_q10_min"][patient_idx].cpu()) if "new_q10_min" in output else 0.0,"new_q10_max":float(output["new_q10_max"][patient_idx].cpu()) if "new_q10_max" in output else 0.0})
                    if "abnormal_network_preictal" in output:
                        from .target_network import NETWORK_FEATURES
                        audits["network"].append({"patient_key":str(subject),"graph_coverage":graph_coverage,**{name:float(output[name][patient_idx].detach().cpu()) for name in NETWORK_FEATURES}})
                    continue
                patient_rows.append(base_row)
                audits["patient"].append({"patient_key": str(subject), "embedding_norm": float(output["patient_outcome_embedding"][patient_idx].norm().cpu()), "n_seizures": int(valid_seizure.sum())})
                for seizure_idx, run_id in enumerate(batch["run_ids"][patient_idx]):
                    if not bool(valid_seizure[seizure_idx]):
                        continue
                    audits["seizure"].append({"patient_key": str(subject), "seizure_id": str(run_id), "seizure_outcome_logit_success": float(output["seizure_outcome_logit_success"][patient_idx, seizure_idx].cpu())})
                    for channel_idx, channel in enumerate(batch["canonical_channels"][patient_idx]):
                        channel_valid = bool(batch["seizure_channel_mask"][patient_idx, seizure_idx, channel_idx])
                        audits["channel"].append({"patient_key": str(subject), "seizure_id": str(run_id), "channel_id": str(channel), "base_failure_evidence": float(output["base_failure_evidence"][patient_idx, seizure_idx, channel_idx].cpu()), "q10_adjustment": float(output["q10_adjustment"][patient_idx, seizure_idx, channel_idx].cpu()), "failure_risk_logit": float(output["failure_risk_logit"][patient_idx, seizure_idx, channel_idx].cpu()), "risk_membership": float(output["risk_membership"][patient_idx, seizure_idx, channel_idx].cpu()), "q10_nez_probability": float(output["simple_q10_nez_probability"][patient_idx, channel_idx].cpu()), "q10_nez_z": float(output["simple_q10_nez_z"][patient_idx, channel_idx].cpu()), "channel_valid": channel_valid})
                    if "network_burden" in output:
                        for phase_idx, phase in enumerate(NETWORK_PHASES):
                            values = output["network_burden"][patient_idx, seizure_idx, phase_idx].cpu().numpy()
                            audits["network"].append({"patient_key": str(subject), "seizure_id": str(run_id), "phase": phase, "B_edge": values[0], "B_cross": values[1], "E_lap": values[2], "H_risk": values[3], "B_hub": values[4], "graph_valid": bool(output["graph_valid"][patient_idx, seizure_idx, phase_idx])})
                            phase_row = {"patient_key": str(subject), "seizure_id": str(run_id), "phase": phase, "phase_index": phase_idx, "graph_valid": bool(output["graph_valid"][patient_idx, seizure_idx, phase_idx])}
                            if "phase_delta" in output and phase_idx > 0:
                                delta = output["phase_delta"][patient_idx, seizure_idx, phase_idx - 1].cpu().numpy()
                                phase_row.update({f"delta_{name}": float(value) for name, value in zip(("B_edge", "B_cross", "E_lap", "H_risk", "B_hub"), delta)})
                                phase_row["delta_valid"] = bool(output["delta_valid"][patient_idx, seizure_idx, phase_idx - 1])
                            audits["phase"].append(phase_row)
                audits["gate"].append({"patient_key": str(subject), "graph_gate": float(output.get("graph_gate", torch.tensor(0.0)).cpu()), "beta_q": float(output["beta_q"].cpu()), "tau_burden": float(output["tau_burden"].cpu())})
    return pd.DataFrame(patient_rows), audits


__all__ = ["NPAMSystem", "class_pos_weight", "fit_target_scalar_normalizer", "make_loader", "predict", "prepare_examples", "seed_everything", "train_fixed_epochs", "train_seizure_probe_fixed_epochs"]
