"""Frozen-B0 CalibRank adapters on the historical Task-1 fixed 80-patient folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


VARIANTS = ("B0", "B1", "B2", "B3")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_groups(archive: np.lib.npyio.NpzFile, part: str, device: torch.device) -> list[dict]:
    ids = archive[f"{part}_subject_id"].astype(str)
    groups = []
    for patient in np.unique(ids):
        indices = np.flatnonzero(ids == patient)
        a = torch.from_numpy(archive[f"{part}_logit_ez"][indices]).to(device)
        h = torch.from_numpy(archive[f"{part}_hidden"][indices]).to(device)
        counts = torch.from_numpy(archive[f"{part}_seizure_count"][indices]).to(device)
        labels = torch.from_numpy(archive[f"{part}_label_nez"][indices].astype(np.int64)).to(device)
        groups.append({"patient_id": patient, "a": a.detach(), "h": h.detach(), "seizures": counts.detach(),
                       "y_nez": labels, "fold": None})
    return groups


def logit(value: torch.Tensor) -> torch.Tensor:
    p = value.clamp(1e-5, 1 - 1e-5)
    return torch.log(p) - torch.log1p(-p)


def patient_statistics(a: torch.Tensor, seizures: torch.Tensor) -> torch.Tensor:
    """Label-blind B0-score distribution, detached before any adapter."""
    a = a.detach()
    seizures = seizures.detach()
    p = torch.sigmoid(a)
    entropy = -(p * torch.log(p.clamp_min(1e-8)) + (1 - p) * torch.log((1 - p).clamp_min(1e-8)))
    q = torch.quantile(a, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=a.device))
    return torch.stack((a.mean(), a.std(unbiased=False), q[2], q[0], q[1], q[3], q[4], q[3] - q[1],
                        entropy.mean(), entropy.std(unbiased=False), torch.log1p(torch.tensor(float(len(a)), device=a.device)),
                        seizures.mean(), seizures.std(unbiased=False)))


def calibrated_bias(a: torch.Tensor, target_fraction: torch.Tensor) -> torch.Tensor:
    """Monotone bisection root with the implicit gradient via one Newton correction."""
    with torch.no_grad():
        target = target_fraction.detach().clamp(1e-5, 1 - 1e-5)
        low = float((-a.detach().max() - 20).item())
        high = float((-a.detach().min() + 20).item())
        for _ in range(32):
            mid = (low + high) / 2
            if float(torch.sigmoid(a.detach() + mid).mean().item()) < float(target.item()):
                low = mid
            else:
                high = mid
        root = torch.tensor((low + high) / 2, dtype=a.dtype, device=a.device)
    probability = torch.sigmoid(a + root)
    slope = (probability * (1 - probability)).mean().clamp_min(1e-6)
    return root + (target_fraction - probability.mean()) / slope


class CalibRank(torch.nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS[1:]:
            raise ValueError(variant)
        self.variant = variant
        self.use_calibration = variant in {"B1", "B3"}
        self.use_rank = variant in {"B2", "B3"}
        if self.use_calibration:
            self.calibration = torch.nn.Sequential(torch.nn.LayerNorm(13), torch.nn.Linear(13, 16),
                                                   torch.nn.GELU(), torch.nn.Linear(16, 1))
            torch.nn.init.zeros_(self.calibration[-1].weight)
            torch.nn.init.zeros_(self.calibration[-1].bias)
        if self.use_rank:
            self.projection = torch.nn.Linear(96, 16)
            self.rank = torch.nn.Sequential(torch.nn.LayerNorm(20), torch.nn.Linear(20, 16),
                                            torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(16, 1))
            torch.nn.init.zeros_(self.rank[-1].weight)
            torch.nn.init.zeros_(self.rank[-1].bias)

    def forward(self, group: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = group["a"].detach()
        h = group["h"].detach()
        counts = group["seizures"].detach()
        zero = a.sum() * 0
        delta = torch.zeros_like(a)
        if self.use_rank:
            p = torch.sigmoid(a)
            ranks = torch.argsort(torch.argsort(a, stable=True), stable=True).to(a.dtype) / max(len(a) - 1, 1)
            entropy = -(p * torch.log(p.clamp_min(1e-8)) + (1 - p) * torch.log((1 - p).clamp_min(1e-8)))
            count_norm = counts / counts.max().clamp_min(1)
            features = torch.cat((self.projection(h), a[:, None], ranks[:, None], entropy[:, None], count_norm[:, None]), dim=1)
            raw = self.rank(features).squeeze(-1)
            delta = 0.10 * torch.tanh(raw)
            delta = delta - delta.mean()
            gate = 4 * p * (1 - p)
            a = a + gate * delta
        target_fraction = zero
        if self.use_calibration:
            stats = patient_statistics(group["a"], counts)
            baseline_fraction = torch.sigmoid(group["a"].detach()).mean()
            target_logit = logit(baseline_fraction) + self.calibration(stats).squeeze()
            target_fraction = torch.sigmoid(target_logit)
            a = a + calibrated_bias(a, target_fraction)
        return a, delta, target_fraction, zero


def patient_objective(model: CalibRank, group: dict) -> tuple[torch.Tensor, dict[str, float]]:
    logits, delta, fraction, zero = model(group)
    y = (1 - group["y_nez"]).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    count_loss = zero
    if model.use_calibration:
        true_fraction = y.mean()
        count_loss = torch.nn.functional.smooth_l1_loss(logit(fraction), logit(true_fraction))
    rank_loss = zero
    if model.use_rank:
        ez, nez = logits[y == 1], logits[y == 0]
        if len(ez) and len(nez):
            weak_count = max(1, math.ceil(0.3 * len(ez)))
            hard_count = min(len(nez), max(1, min(16, len(ez))))
            rank_loss = torch.nn.functional.softplus(0.05 + nez.topk(hard_count).values.mean() -
                                                      ez.topk(weak_count, largest=False).values.mean())
    penalty = delta.square().mean()
    loss = bce + 0.10 * count_loss + 0.05 * rank_loss + 0.01 * penalty
    return loss, {"bce": float(bce.detach()), "count": float(count_loss.detach()),
                  "rank": float(rank_loss.detach()), "residual": float(penalty.detach())}


@torch.no_grad()
def predict(groups: list[dict], model: CalibRank | None, fold: int) -> pd.DataFrame:
    if model is not None:
        model.eval()
    rows = []
    for group in groups:
        logits = group["a"] if model is None else model(group)[0]
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = group["y_nez"].detach().cpu().numpy()
        for index, (score, label) in enumerate(zip(scores, labels)):
            rows.append({"subject_id": group["patient_id"], "outer_fold": fold, "channel_index": index,
                         "label_nez": int(label), "score_nez_probability": float(1 - score)})
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, threshold_nez: float) -> pd.DataFrame:
    rows = []
    for patient, group in frame.groupby("subject_id", sort=True):
        y = group.label_nez.to_numpy(int)
        score_ez = 1 - group.score_nez_probability.to_numpy(float)
        pred = (group.score_nez_probability.to_numpy(float) >= threshold_nez).astype(int)
        ez = (y == 0).astype(int)
        order = np.argsort(-score_ez, kind="mergesort")
        hits = np.flatnonzero(ez[order] == 1)
        rows.append({"subject_id": patient, "outer_fold": int(group.outer_fold.iloc[0]),
                     "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
                     "ez_f1": float(f1_score(ez, pred == 0, zero_division=0)),
                     "nez_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                     "accuracy": float((y == pred).mean()),
                     "ez_auprc": float(average_precision_score(ez, score_ez)) if ez.sum() else 0.0,
                     "ez_auroc": float(roc_auc_score(ez, score_ez)) if 0 < ez.sum() < len(ez) else float("nan"),
                     "ez_mrr": float(1 / (hits[0] + 1)) if len(hits) else 0.0,
                     "top1_is_ez": float(ez[order[0]]),
                     "true_k_recall": float(ez[order[:int(ez.sum())]].sum() / ez.sum()) if ez.sum() else float("nan")})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    columns = ("macro_f1", "ez_f1", "nez_f1", "balanced_accuracy", "accuracy", "ez_auprc", "ez_auroc",
               "ez_mrr", "top1_is_ez", "true_k_recall")
    return {column: float(frame[column].mean()) for column in columns}


def bootstrap(patient: pd.DataFrame, reference: pd.DataFrame, seed: int = 20260926) -> dict[str, object]:
    merged = patient.merge(reference, on="subject_id", how="inner", validate="one_to_one", suffixes=("_method", "_b0"))
    if len(merged) != 80:
        raise RuntimeError("Paired bootstrap requires exactly 80 patients")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(merged), size=(2000, len(merged)))
    report = {}
    for metric in ("macro_f1", "ez_f1", "balanced_accuracy", "ez_auprc", "ez_mrr"):
        delta = (merged[f"{metric}_method"] - merged[f"{metric}_b0"]).to_numpy(float)
        draws = np.nanmean(delta[samples], axis=1)
        report[metric] = {"delta": float(np.nanmean(delta)), "ci_low": float(np.nanquantile(draws, 0.025)),
                          "ci_high": float(np.nanquantile(draws, 0.975))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--replay-audit", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--b0-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    replay = json.loads(args.replay_audit.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    if replay.get("status") != "PASS" or not diagnostic.get("constructive_gate", {}).get("pass"):
        raise RuntimeError("B0 replay or predeclared diagnostic gate did not pass")
    if args.preflight_only:
        with np.load(args.embeddings / "fold1.npz", allow_pickle=True) as archive:
            fit = patient_groups(archive, "fit", device)
            validation = patient_groups(archive, "validation", device)
        for variant in VARIANTS[1:]:
            set_seed(42 + 1009)
            model = CalibRank(variant).to(device)
            original = predict(validation, None, 1)
            epoch0 = predict(validation, model, 1)
            difference = float(np.max(np.abs(original.score_nez_probability - epoch0.score_nez_probability)))
            if difference > 3e-6:
                raise RuntimeError(f"Preflight epoch0 mismatch for {variant}: {difference}")
            loss, _ = patient_objective(model, fit[0])
            loss.backward()
            gradient = sum(float(parameter.grad.abs().sum()) for parameter in model.parameters()
                           if parameter.grad is not None)
            if not math.isfinite(gradient) or gradient <= 0:
                raise RuntimeError(f"Preflight gradient failed for {variant}: {gradient}")
            print(f"PREFLIGHT_PASS {variant} epoch0_max_difference={difference:.9g} gradient_l1={gradient:.6g}", flush=True)
        return
    protocol = {"seed": 42, "folds": [1, 2, 3, 4, 5], "variants": list(VARIANTS),
                "status": "EXPLORATORY_OUTER_PREVIOUSLY_VIEWED", "backbone": "frozen patient_z_mlp, 88 input, 96 hidden",
                "ranking_residual_bound": 0.10, "rank_projection_dimension": 16, "rank_dropout": 0.1,
                "calibration_hidden_dimension": 16, "calibration_statistics_dimension": 13,
                "loss": {"final_bce": 1.0, "count": 0.10, "rank": 0.05, "delta_squared": 0.01},
                "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                "max_epochs": 30, "patience": 6, "minimum_stop_epoch": 6, "gradient_clip": 1.0,
                "selection": "validation patient Macro-F1, then EZ-F1, then earliest epoch; epoch0 B0 legal",
                "threshold": "original frozen patient-Macro-F1 selector, validation only, 0.005 grid",
                "adapter_input": "detached B0 hidden and logits; no patient/center identifiers or labels in forward",
                "oracle_used_for_training": False, "outer_test_used_for_tuning": False,
                "code_sha256": sha256(Path(__file__)), "replay_audit_sha256": sha256(args.replay_audit),
                "diagnostics_sha256": sha256(args.diagnostics), "b0_ledger_sha256": sha256(args.b0_ledger),
                "embedding_hashes": {str(fold): sha256(args.embeddings / f"fold{fold}.npz") for fold in range(1, 6)}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / "PROTOCOL_LOCK.json"
    if lock.exists():
        if json.loads(lock.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Protocol lock differs on resume")
    else:
        atomic_json(lock, protocol)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from task1_baselines.thresholds import select_patient_macro_threshold
    b0_ledger = pd.read_csv(args.b0_ledger)
    fold_rows = []
    development = []
    all_patient = []
    for fold in range(1, 6):
        with np.load(args.embeddings / f"fold{fold}.npz", allow_pickle=True) as archive:
            groups = {part: patient_groups(archive, part, device) for part in ("fit", "validation", "test")}
            b0_threshold = float(archive["b0_selected_threshold_nez"])
            b0_epoch = int(archive["b0_selected_epoch"])
        if [len(groups[name]) for name in ("fit", "validation", "test")] != [[51, 13, 16], [51, 13, 16],
                                                                                [50, 13, 17], [52, 13, 15],
                                                                                [51, 13, 16]][fold - 1]:
            raise RuntimeError(f"Frozen subject membership changed in fold {fold}")
        base_val = predict(groups["validation"], None, fold)
        base_score = summarize(evaluate(base_val, b0_threshold))
        for variant in VARIANTS:
            location = args.output_dir / variant / f"fold{fold}"
            complete = location / "COMPLETE.json"
            if complete.exists():
                saved = json.loads(complete.read_text(encoding="utf-8"))
                fold_rows.append(saved["fold_result"])
                development.append(saved["development_result"])
                private = pd.read_csv(location / "TEST_PATIENT_PRIVATE.csv")
                all_patient.append(private)
                print(f"SKIP_COMPLETE variant={variant} fold={fold}", flush=True)
                continue
            if location.exists() and any(location.iterdir()):
                raise RuntimeError(f"Incomplete nonempty cell; preserve for diagnosis: {location}")
            location.mkdir(parents=True)
            if variant == "B0":
                model = None
                best_epoch, threshold = b0_epoch, b0_threshold
                selected_validation = base_score
                history = []
            else:
                set_seed(42 + 1009 * fold)
                model = CalibRank(variant).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                best_epoch, threshold = 0, b0_threshold
                best_key = (base_score["macro_f1"], base_score["ez_f1"], 0)
                selected_validation = base_score
                best_state = None
                stale = 0
                history = []
                # Zero-init residual paths must reproduce B0 at epoch0.
                epoch0 = predict(groups["validation"], model, fold)
                if np.max(np.abs(epoch0.score_nez_probability - base_val.score_nez_probability)) > 3e-6:
                    raise RuntimeError(f"{variant} fold{fold} epoch0 does not reproduce B0")
                for epoch in range(1, 31):
                    model.train()
                    losses = []
                    ordered = np.random.default_rng(42 + 1009 * fold + epoch).permutation(len(groups["fit"]))
                    for index in ordered:
                        optimizer.zero_grad(set_to_none=True)
                        loss, _ = patient_objective(model, groups["fit"][int(index)])
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite loss {variant} fold{fold} epoch{epoch}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    val = predict(groups["validation"], model, fold)
                    chosen = select_patient_macro_threshold(val, source="outer_validation_patient_macro_f1")
                    key = (chosen.patient_macro_f1, chosen.patient_ez_f1, -epoch)
                    improved = key > best_key
                    history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                                    "validation_macro_f1": chosen.patient_macro_f1,
                                    "validation_ez_f1": chosen.patient_ez_f1, "validation_threshold_nez": chosen.threshold,
                                    "improved": improved})
                    if improved:
                        best_key, best_epoch, threshold, stale = key, epoch, chosen.threshold, 0
                        selected_validation = summarize(evaluate(val, threshold))
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        stale += 1
                    print(f"EPOCH {variant} fold={fold} epoch={epoch} val={chosen.patient_macro_f1:.6f} best={best_key[0]:.6f} stale={stale}", flush=True)
                    if epoch >= 6 and stale >= 6:
                        break
                pd.DataFrame(history).to_csv(location / "HISTORY.csv", index=False)
                if best_epoch == 0:
                    model = None
                else:
                    assert best_state is not None
                    model.load_state_dict(best_state)
                    torch.save({"state_dict": best_state, "epoch": best_epoch, "threshold": threshold,
                                "protocol_sha256": sha256(lock)}, location / "ADAPTER_PRIVATE.pt")
            # Outer test is first consulted only after this cell's validation selection.
            test = predict(groups["test"], model, fold)
            test_patient = evaluate(test, threshold)
            test_patient["variant"] = variant
            test_patient.to_csv(location / "TEST_PATIENT_PRIVATE.csv", index=False)
            test_summary = summarize(test_patient)
            fold_result = {"variant": variant, "fold": fold, "seed": 42, "test_patients": len(test_patient),
                           "selected_epoch": best_epoch, "threshold_nez": threshold, **test_summary}
            development_result = {"variant": variant, "fold": fold, "seed": 42, "validation_patients": len(groups["validation"]),
                                  "selected_epoch": best_epoch, "threshold_nez": threshold, **selected_validation}
            atomic_json(complete, {"fold_result": fold_result, "development_result": development_result})
            fold_rows.append(fold_result)
            development.append(development_result)
            all_patient.append(test_patient)
            print("CELL_COMPLETE " + json.dumps(fold_result), flush=True)
    if len(fold_rows) != 20 or len(development) != 20:
        raise RuntimeError("Expected all four variants times five folds")
    pd.DataFrame(development).sort_values(["variant", "fold"]).to_csv(args.output_dir / "DEVELOPMENT_RESULTS.csv", index=False)
    pd.DataFrame(fold_rows).sort_values(["variant", "fold"]).to_csv(args.output_dir / "OUTER_FOLD_RESULTS.csv", index=False)
    patients = pd.concat(all_patient, ignore_index=True)
    if any(patients.loc[patients.variant == v, "subject_id"].nunique() != 80 for v in VARIANTS):
        raise RuntimeError("Each variant must cover exactly 80 patients")
    patients.to_csv(args.output_dir / "ALL_TEST_PATIENT_PRIVATE.csv", index=False)
    summary_rows, paired_rows = [], []
    b0_patients = patients[patients.variant == "B0"]
    for variant in VARIANTS:
        part = patients[patients.variant == variant]
        summary_rows.append({"variant": variant, "seed": 42, "patients": len(part), **summarize(part),
                             "positive_folds_vs_b0": int(sum(
                                 pd.DataFrame(fold_rows).query("variant == @variant and fold == @fold").iloc[0].macro_f1 >
                                 pd.DataFrame(fold_rows).query("variant == 'B0' and fold == @fold").iloc[0].macro_f1
                                 for fold in range(1, 6))) if variant != "B0" else 0})
        if variant != "B0":
            paired = bootstrap(part, b0_patients)
            for metric, result in paired.items():
                paired_rows.append({"variant": variant, "metric": metric, "seed": 42, "patients": 80, **result})
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "OUTER_SUMMARY.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(args.output_dir / "PAIRED_BOOTSTRAP.csv", index=False)
    print("ALL_COMPLETE " + pd.DataFrame(summary_rows).to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
