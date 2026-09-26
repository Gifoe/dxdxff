"""Fit fixed C1/C2/C3 optimal-cardinality corrections on outer-fit OOF patients."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def delta_intervals(row: dict) -> list[tuple[float, float]]:
    c = len(row["a"])
    return [(lo / c - row["q0"], hi / c - row["q0"]) for lo, hi in row["optimal_runs"]]


def nearest_zero(intervals: list[tuple[float, float]]) -> float:
    candidates = [min(max(0.0, lo), hi) for lo, hi in intervals]
    return min(candidates, key=lambda x: (abs(x), x))


def fit_linear(x: np.ndarray, targets: list[list[tuple[float, float]]]) -> tuple[dict, dict]:
    scaler = StandardScaler().fit(x)
    x_tensor = torch.from_numpy(scaler.transform(x).astype(np.float32))
    torch.manual_seed(42)
    layer = torch.nn.Linear(x.shape[1], 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01, weight_decay=0.01)
    first, last = None, None
    for step in range(500):
        optimizer.zero_grad(set_to_none=True)
        prediction = layer(x_tensor).squeeze(1)
        distances = []
        for value, union in zip(prediction, targets):
            candidates = [torch.maximum(torch.maximum(torch.as_tensor(lo, dtype=value.dtype) - value,
                                                       value - torch.as_tensor(hi, dtype=value.dtype)),
                                        torch.zeros_like(value)) for lo, hi in union]
            distances.append(torch.stack(candidates).min())
        dist = torch.stack(distances)
        loss = torch.nn.functional.smooth_l1_loss(dist, torch.zeros_like(dist)) + 0.02 * prediction.square().mean()
        if first is None:
            first = float(loss.detach())
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
    model = {"scaler": scaler, "weight": layer.weight.detach().numpy().reshape(-1).astype(float),
             "bias": float(layer.bias.detach().item())}
    audit = {"first_objective": first, "last_objective": last,
             "weight_l2": float(np.linalg.norm(model["weight"])), "bias": model["bias"]}
    return model, audit


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    return model["scaler"].transform(x) @ model["weight"] + model["bias"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("context-dir", "protocol-lock", "output-dir"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    audit = json.loads((args.context_dir / "PATIENT_CONTEXT_AUDIT.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["protocol_lock_sha256"] != sha(args.protocol_lock) or lock["variants"] != ["C0", "C1", "C2", "C3"]:
        raise RuntimeError("Context/variant protocol integrity failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold in range(1, 6):
        path = args.context_dir / f"fold{fold}_CONTEXT_PRIVATE.pkl"
        if sha(path) != audit["folds"][fold - 1]["private_context_sha256"]:
            raise RuntimeError("Context archive hash changed")
        with path.open("rb") as f:
            rows = pickle.load(f)
        train = [r for r in rows if r["role"] == "fit"]
        if len(train) != audit["folds"][fold - 1]["fit_patients"]:
            raise RuntimeError("Outer-fit cardinality target count mismatch")
        targets = [delta_intervals(r) for r in train]
        c1 = float(np.median([nearest_zero(union) for union in targets]))
        score_x = np.stack([r["z_score"] for r in train])
        abs_x = np.stack([r["z_abs"] for r in train])
        c2, c2_audit = fit_linear(score_x, targets)
        c3, c3_audit = fit_linear(np.c_[score_x, abs_x], targets)
        models = {"C1_constant_delta": c1, "C2": c2, "C3": c3}
        model_path = args.output_dir / f"fold{fold}_MODELS_PRIVATE.joblib"
        joblib.dump(models, model_path)
        predictions = []
        for r in rows:
            s = r["z_score"].reshape(1, -1)
            sa = np.c_[s, r["z_abs"].reshape(1, -1)]
            predictions.append({"patient_id": r["patient_id"], "role": r["role"],
                                "C0": 0.0, "C1": c1, "C2": float(predict(c2, s)[0]),
                                "C3": float(predict(c3, sa)[0])})
        pred_path = args.output_dir / f"fold{fold}_PREDICTIONS_PRIVATE.pkl"
        with pred_path.open("wb") as f:
            pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
        folds.append({"fold": fold, "fit_patients": len(train), "constant_delta": c1,
                      "C2": c2_audit, "C3": c3_audit,
                      "private_model_sha256": sha(model_path), "private_prediction_sha256": sha(pred_path)})
        print(f"CARDINALITY_TRAIN_FOLD_PASS fold={fold} constant_delta={c1:.6f}", flush=True)
    result = {"status": "PASS", "protocol_lock_sha256": sha(args.protocol_lock),
              "training_scope": "outer-fit OOF only", "folds": folds}
    (args.output_dir / "TRAINING_AUDIT_PRIVATE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("CARDINALITY_TRAIN_PASS", flush=True)


if __name__ == "__main__":
    main()
