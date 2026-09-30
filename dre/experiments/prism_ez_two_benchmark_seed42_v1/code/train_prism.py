"""Train/validate PRiSM-EZ without opening any outer or official-test record.

The private feature cache stores only label-independent 68-D tokens.  Labels
are loaded separately from the frozen source bank only for the patient-equal
loss and metrics.  This script intentionally has no test-cache argument.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
PC_CODE = HERE.parents[2] / "pc_cnn_two_benchmark_seed42_v1" / "code"
sys.path.insert(0, str(PC_CODE))
from patient_bank import IctalBank, OmniTrainBank  # noqa: E402
from raw_metrics import ictal_validation, omni_validation  # noqa: E402

from prism_ez import FEATURE_DIM, PHYSIOLOGY_DIM, PRiSMEZ, spectral_sketch


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _hash(value: str, size: int = 24) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:size]


def build_bank(args, lock):
    if args.benchmark == "ictal":
        if args.fold not in range(1, 6) or digest(args.ictal_manifest) != lock["ictal_fold_manifest_sha256"]:
            raise RuntimeError("ictal fold manifest differs from PRiSM protocol lock")
        bank = IctalBank(args.ictal_cache, args.ictal_manifest)
        return bank, bank.folds[args.fold]["fit"], bank.folds[args.fold]["validation"]
    if args.fold != 1 or digest(args.omni_official_split) != lock["omni_official_split_sha256"] or \
            digest(args.omni_inner_split) != lock["omni_inner_train_val_split_sha256"]:
        raise RuntimeError("Omni frozen train/validation split differs from PRiSM protocol lock")
    bank = OmniTrainBank(args.omni_cache, args.omni_inner_split, args.omni_official_split)
    return bank, bank.patients("inner_train"), bank.patients("inner_val")


def source_groups(bank, patient: str):
    """Return groups of raw records without consulting any label for grouping.

    Ictal aggregates all seizures for a patient-channel.  Omni aggregates the
    deterministic up-to-five training clips only within an EDF, preserving the
    official EDF-channel prediction unit and never mixing scores across EDFs.
    """
    if isinstance(bank, IctalBank):
        return {"patient": list(bank.records(patient))}
    grouped = defaultdict(list)
    for record in bank.records(patient, 0, all_clips=True):
        grouped[str(record["edf"])].append(record)
    result = {}
    for edf, records in grouped.items():
        # This is deterministic from metadata only.  The cache contains the
        # frozen clip starts; no class label can alter the chosen five clips.
        ordered = sorted(records, key=lambda r: (_hash(f"42|{patient}|{edf}|{r['record_index']}"),
                                                 int(r["record_index"])))[:5]
        if not ordered:
            raise RuntimeError("Omni EDF has no frozen train clip")
        result[edf] = ordered
    return result


class TokenStore:
    """Private label-independent raw-token cache shared across inner folds."""
    def __init__(self, root: Path, benchmark: str, bank, protocol_sha: str):
        self.root, self.benchmark, self.bank, self.protocol_sha = Path(root), benchmark, bank, protocol_sha
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, patient: str, group: str) -> Path:
        return self.root / self.benchmark / f"{_hash(patient, 20)}_{_hash(group, 20)}.npz"

    @staticmethod
    def _record_tokens(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        descriptor = np.asarray(record["descriptors"], dtype=np.float32)
        descriptor_mask = np.asarray(record["descriptor_mask"], dtype=bool)
        channels, windows, dimensions = descriptor.shape
        if dimensions != PHYSIOLOGY_DIM:
            raise RuntimeError("A1 descriptor contract changed from 36D")
        window_mask = descriptor_mask.any(axis=-1) & np.asarray(record["channel_mask"], dtype=bool)[:, None]
        sketch, sketch_mask = spectral_sketch(record["waveforms"], record["sampling_rate_hz"], window_mask)
        values = np.concatenate((np.where(descriptor_mask, descriptor, 0.0), sketch), axis=-1)
        feature_mask = np.concatenate((descriptor_mask, sketch_mask), axis=-1)
        if values.shape != (channels, windows, FEATURE_DIM) or feature_mask.shape != values.shape:
            raise RuntimeError("68-D feature construction changed shape")
        return values.astype(np.float32), feature_mask.astype(bool), window_mask.astype(bool)

    def ensure(self, patient: str, group: str, records: list[dict]) -> Path:
        path = self._path(patient, group)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["protocol_sha256"]) != self.protocol_sha or str(saved["patient"]) != patient or \
                        str(saved["group"]) != group:
                    raise RuntimeError("raw-token cache provenance mismatch")
            return path
        if not records:
            raise RuntimeError("cannot build a token cache for an empty record group")
        names = [str(value) for value in records[0]["channel_names"]]
        labels = np.asarray(records[0]["labels"], dtype=np.int8)
        channel_count = len(names)
        xs, feature_masks, window_masks, channel_masks = [], [], [], []
        for record in records:
            current_names = [str(value) for value in record["channel_names"]]
            if current_names != names or not np.array_equal(np.asarray(record["labels"], dtype=np.int8), labels):
                raise RuntimeError("channel names or labels differ inside a record-pooling group")
            x, feature_mask, window_mask = self._record_tokens(record)
            xs.append(x); feature_masks.append(feature_mask); window_masks.append(window_mask)
            channel_masks.append(np.asarray(record["channel_mask"], dtype=bool))
        temporary = path.with_suffix(".tmp.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(temporary, protocol_sha256=np.asarray(self.protocol_sha), patient=np.asarray(patient),
                            group=np.asarray(group), values=np.stack(xs), feature_mask=np.stack(feature_masks),
                            window_mask=np.stack(window_masks), channel_mask=np.stack(channel_masks),
                            labels=labels, channel_names=np.asarray(names, dtype="U"),
                            group_is_edf=np.asarray(self.benchmark == "omni"), source_records=np.asarray(len(records)))
        temporary.replace(path)
        return path

    def prepare_patient(self, patient: str) -> list[Path]:
        return [self.ensure(patient, group, records) for group, records in source_groups(self.bank, patient).items()]

    def groups(self, patient: str) -> list[Path]:
        paths = sorted((self.root / self.benchmark).glob(f"{_hash(patient, 20)}_*.npz"))
        if not paths:
            raise RuntimeError("requested patient has not been precomputed under allowed scope")
        return paths


class RobustScaler:
    """Exact train-only per-feature median/IQR over cached label-free tokens."""
    def __init__(self, payload: dict):
        self.payload = payload
        self.median = np.asarray(payload["median"], dtype=np.float32)
        self.iqr = np.asarray(payload["iqr"], dtype=np.float32)

    @classmethod
    def fit(cls, store: TokenStore, patients: list[str], *, benchmark: str, fold: int,
            protocol_sha: str, out: Path):
        if out.exists():
            payload = json.loads(out.read_text(encoding="utf-8"))
            if payload["protocol_sha256"] != protocol_sha or payload["fit_patients"] != len(patients) or \
                    payload["benchmark"] != benchmark or int(payload["fold"]) != fold:
                raise RuntimeError("robust scaler resume provenance mismatch")
            return cls(payload)
        buckets = [[] for _ in range(FEATURE_DIM)]
        for patient in patients:
            for path in store.groups(patient):
                with np.load(path, allow_pickle=False) as item:
                    values, mask = item["values"], item["feature_mask"].astype(bool)
                for feature in range(FEATURE_DIM):
                    observed = values[..., feature][mask[..., feature]]
                    if len(observed):
                        buckets[feature].append(observed.astype(np.float32, copy=False))
        median, iqr, count = np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM), np.zeros(FEATURE_DIM, dtype=int)
        for feature, pieces in enumerate(buckets):
            if not pieces:
                continue
            all_values = np.concatenate(pieces)
            median[feature] = np.median(all_values)
            iqr[feature] = max(float(np.percentile(all_values, 75) - np.percentile(all_values, 25)), 1e-5)
            count[feature] = len(all_values)
        payload = {"benchmark": benchmark, "fold": fold, "fit_patients": len(patients),
                   "protocol_sha256": protocol_sha, "method": "train-only exact median/IQR",
                   "median": median.tolist(), "iqr": iqr.tolist(), "observed_count": count.tolist(),
                   "validation_or_test_used_for_fit": False, "clip": [-8.0, 8.0]}
        atomic_json(out, payload)
        return cls(payload)

    def apply(self, item: np.lib.npyio.NpzFile) -> list[dict]:
        raw, feature_mask = item["values"].astype(np.float32), item["feature_mask"].astype(bool)
        normalized = np.clip((raw - self.median) / self.iqr, -8.0, 8.0).astype(np.float32)
        normalized[~feature_mask] = 0.0
        if not np.isfinite(normalized).all():
            raise RuntimeError("nonfinite robustly normalized token")
        return [{"features": torch.from_numpy(normalized[index]),
                 "channel_mask": torch.from_numpy(item["channel_mask"][index].astype(bool)),
                 "window_mask": torch.from_numpy(item["window_mask"][index].astype(bool))}
                for index in range(len(normalized))]


def load_group(path: Path, scaler: RobustScaler, device: torch.device):
    with np.load(path, allow_pickle=False) as item:
        labels = item["labels"].astype(np.int8); names = [str(value) for value in item["channel_names"]]
        group, is_edf = str(item["group"]), bool(item["group_is_edf"])
        records = scaler.apply(item)
    converted = [{key: value.to(device) for key, value in record.items()} for record in records]
    return {"records": converted, "labels": labels, "names": names, "group": group, "is_edf": is_edf}


def patient_groups(store: TokenStore, patient: str, scaler: RobustScaler, device: torch.device):
    return [load_group(path, scaler, device) for path in store.groups(patient)]


def class_weight(store: TokenStore, patients: list[str]) -> float:
    positive = negative = 0
    for patient in patients:
        # One label per group/channel gives the same patient-equal loss support
        # as the actual optimizer aggregation below.
        for path in store.groups(patient):
            with np.load(path, allow_pickle=False) as item:
                labels, present = item["labels"].astype(int), item["channel_mask"].any(axis=0).astype(bool)
            valid = labels[present] >= 0
            positive += int((labels[present][valid] == 1).sum())
            negative += int((labels[present][valid] == 0).sum())
    if positive < 1 or negative < 1:
        raise RuntimeError("train partition lacks a class for positive-weight calculation")
    return float(np.clip(math.sqrt(negative / positive), 1.0, 4.0))


def patient_loss(model: PRiSMEZ, groups, positive_weight: float):
    logits, labels = [], []
    for group in groups:
        score = model.forward_group(group["records"])
        channel_mask = torch.stack([record["channel_mask"] for record in group["records"]]).any(0)
        y = torch.from_numpy(group["labels"].astype(np.float32)).to(score.device)
        selected = channel_mask & (y >= 0)
        if bool(selected.any()):
            logits.append(score[selected]); labels.append(y[selected])
    if not logits:
        raise RuntimeError("fit patient has no supervised channel")
    p, y = torch.cat(logits), torch.cat(labels)
    weights = torch.where(y > 0, float(positive_weight), 1.0)
    return torch.nn.functional.binary_cross_entropy_with_logits(p, y, weight=weights, reduction="mean"), int(len(y))


def prediction(model: PRiSMEZ, store: TokenStore, patients: list[str], scaler: RobustScaler,
               device: torch.device, *, diagnostics: bool = False):
    model.eval(); result, gate_values, spectral_rank = {}, [], []
    with torch.inference_mode():
        for patient in patients:
            rows = []
            for group in patient_groups(store, patient, scaler, device):
                if diagnostics:
                    logits, info = model.forward_group(group["records"], return_diagnostics=True)
                    gate_values.append(info["gate"].float().cpu().numpy())
                    spectral_rank.append(info["rank"][..., PHYSIOLOGY_DIM:].float().cpu().numpy())
                else:
                    logits = model.forward_group(group["records"])
                available = torch.stack([record["channel_mask"] for record in group["records"]]).any(0).cpu().numpy()
                chosen = np.flatnonzero(available)
                rows.append({"channel": [group["names"][int(index)] for index in chosen],
                             "label": group["labels"][chosen].astype(int).tolist(),
                             "score": torch.sigmoid(logits[chosen]).cpu().numpy().astype(float).tolist(),
                             "edf": group["group"] if group["is_edf"] else ""})
            result[patient] = rows
    metrics = ictal_validation(result) if store.benchmark == "ictal" else omni_validation(result)
    diag = {"gate": np.concatenate(gate_values) if gate_values else np.empty((0, 48)),
            "spectral_rank": np.concatenate(spectral_rank) if spectral_rank else np.empty((0, 32))}
    return metrics, result, diag


def selection_rank(metrics: dict, epoch: int, benchmark: str):
    # None cannot win a selection tie and is explicitly carried as -inf.
    safe = lambda value: float(value) if value is not None else float("-inf")
    if benchmark == "ictal":
        return (safe(metrics["auroc"]), safe(metrics["ap"]), safe(metrics["mrr"]), -epoch)
    return (safe(metrics["auroc"]), safe(metrics["ap"]), safe(metrics["patient_ap"]), safe(metrics["mrr"]), -epoch)


def macro_f1_threshold(private: dict, benchmark: str):
    ys, scores = [], []
    if benchmark == "ictal":
        for rows in private.values():
            for row in rows:
                ys.extend(value for value in row["label"] if value >= 0)
                scores.extend(value for value, label in zip(row["score"], row["label"]) if label >= 0)
    else:
        by_key, labels = defaultdict(list), {}
        for rows in private.values():
            for row in rows:
                for channel, y, score in zip(row["channel"], row["label"], row["score"]):
                    if y >= 0:
                        key = (row["edf"], channel); labels[key] = int(y); by_key[key].append(float(score))
        ys = [labels[key] for key in sorted(by_key)]
        scores = [float(np.mean(by_key[key])) for key in sorted(by_key)]
    y, p = np.asarray(ys, dtype=int), np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    value = np.asarray([f1_score(y, p >= threshold, average="macro", zero_division=0) for threshold in candidates])
    index = int(np.flatnonzero(value == value.max())[0])
    return {"threshold": float(candidates[index]), "macro_f1": float(value[index]),
            "candidate_count": int(len(candidates)), "source": "validation only"}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one(args, lock, bank, fit, val, store: TokenStore, protocol_sha: str):
    work = args.runtime / args.benchmark / f"fold{args.fold}"
    completion = work / "PRISM_VALIDATION_SELECTION.json"
    if completion.exists():
        saved = json.loads(completion.read_text(encoding="utf-8"))
        if saved.get("status") == "TRAIN_VALIDATION_COMPLETE" and saved.get("protocol_sha256") == protocol_sha and not saved.get("test_accessed"):
            return saved
        raise RuntimeError("existing PRiSM validation marker has incompatible provenance")
    # Explicitly prepare only fit and inner-validation identities.  Outer/test
    # patient files cannot enter either token cache or scaler construction.
    for patient in [*fit, *val]:
        store.prepare_patient(patient)
    scaler = RobustScaler.fit(store, fit, benchmark=args.benchmark, fold=args.fold,
                              protocol_sha=protocol_sha, out=work / "feature_normalization.json")
    positive_weight = class_weight(store, fit)
    set_seed(420000 + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRiSMEZ().to(device)
    parameter_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if parameter_count >= 100000:
        raise RuntimeError("PRiSM parameter gate failed before training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lock["training"]["lr"]),
                                  weight_decay=float(lock["training"]["weight_decay"]))
    last, best = work / "last.pt", work / "selected_best.pt"
    selected, start, stale = None, 1, 0
    runtime_rows = []
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["protocol_sha256"] != protocol_sha or state["benchmark"] != args.benchmark or state["fold"] != args.fold:
            raise RuntimeError("resume checkpoint provenance mismatch")
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].detach().cpu())
        if device.type == "cuda": torch.cuda.set_rng_state(state["cuda_rng"].detach().cpu())
        selected, start, stale = state["selected"], int(state["epoch"]) + 1, int(state["stale"])
    max_epochs, patience = int(lock["training"]["max_epochs"]), int(lock["training"]["patience"])
    for epoch in range(start, max_epochs + 1):
        if epoch <= 2:
            factor = epoch / 2.0
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * (epoch - 2) / max(max_epochs - 2, 1)))
        for group in optimizer.param_groups: group["lr"] = float(lock["training"]["lr"]) * factor
        model.train(); started = time.perf_counter(); patient_seconds, observed = 0.0, 0
        shuffled = list(fit); random.shuffle(shuffled)
        for patient in shuffled:
            optimizer.zero_grad(set_to_none=True)
            loss, count = patient_loss(model, patient_groups(store, patient, scaler, device), positive_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(lock["training"]["gradient_clip"]))
            optimizer.step(); observed += count
        metrics, private, _ = prediction(model, store, val, scaler, device)
        current_rank = selection_rank(metrics, epoch, args.benchmark)
        improved = selected is None or tuple(current_rank) > tuple(selected["rank"])
        if improved:
            selected, stale = {"epoch": epoch, "rank": list(current_rank), "metrics": metrics}, 0
            atomic_torch(best, {"protocol_sha256": protocol_sha, "benchmark": args.benchmark, "fold": args.fold,
                                "selected": selected, "model_state": model.state_dict(), "test_accessed": False})
        else:
            stale += 1
        seconds = time.perf_counter() - started
        runtime_rows.append({"benchmark": args.benchmark, "fold": args.fold, "epoch": epoch,
                             "seconds_per_epoch": seconds, "train_patients": len(fit),
                             "supervised_train_group_channels": observed, "peak_gpu_memory_bytes":
                             int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                             "parameter_count": parameter_count})
        atomic_json(work / f"validation_epoch_{epoch:02d}_private.json", {"epoch": epoch, "private": private})
        atomic_torch(last, {"protocol_sha256": protocol_sha, "benchmark": args.benchmark, "fold": args.fold,
                            "epoch": epoch, "stale": stale, "selected": selected, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(), "python_rng": random.getstate(),
                            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state().detach().cpu() if device.type == "cuda" else None,
                            "test_accessed": False})
        print(json.dumps({"status": "EPOCH_COMPLETE", "benchmark": args.benchmark, "fold": args.fold,
                          "epoch": epoch, "validation": metrics, "selected_epoch": selected["epoch"],
                          "early_stop_stale": stale, "seconds": seconds, "test_accessed": False}), flush=True)
        if epoch >= 2 and stale >= patience:
            break
    if selected is None or not best.exists():
        raise RuntimeError("no PRiSM checkpoint selected")
    state = torch.load(best, map_location=device, weights_only=False); model.load_state_dict(state["model_state"])
    final_metrics, final_private, diagnostics = prediction(model, store, val, scaler, device, diagnostics=True)
    threshold = macro_f1_threshold(final_private, args.benchmark)
    gate = diagnostics["gate"].reshape(-1)
    selection = {"status": "TRAIN_VALIDATION_COMPLETE", "protocol_sha256": protocol_sha,
                 "benchmark": args.benchmark, "fold": args.fold, "fit_patients": len(fit),
                 "validation_patients": len(val), "selected": selected, "selected_replayed_metrics": final_metrics,
                 "validation_threshold": threshold, "positive_weight": positive_weight, "parameter_count": parameter_count,
                 "early_stopping": {"max_epochs": max_epochs, "patience": patience,
                                    "completed_epochs": len(runtime_rows), "last_stale": stale},
                 "gate_activation": {"mean": float(gate.mean()), "std": float(gate.std()),
                                     "p05": float(np.quantile(gate, .05)), "p25": float(np.quantile(gate, .25)),
                                     "p50": float(np.quantile(gate, .50)), "p75": float(np.quantile(gate, .75)),
                                     "p95": float(np.quantile(gate, .95))}, "test_accessed": False}
    atomic_json(completion, selection)
    with (work / "runtime_private.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runtime_rows[0])); writer.writeheader(); writer.writerows(runtime_rows)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=("ictal", "omni"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--ictal-cache", type=Path); parser.add_argument("--ictal-manifest", type=Path)
    parser.add_argument("--omni-cache", type=Path); parser.add_argument("--omni-official-split", type=Path)
    parser.add_argument("--omni-inner-split", type=Path)
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8")); protocol_sha = digest(args.protocol)
    bank, fit, val = build_bank(args, lock)
    if set(fit) & set(val): raise RuntimeError("fit/validation patient overlap")
    store = TokenStore(args.feature_cache, args.benchmark, bank, protocol_sha)
    print(json.dumps(train_one(args, lock, bank, fit, val, store, protocol_sha), indent=2), flush=True)


if __name__ == "__main__":
    main()
