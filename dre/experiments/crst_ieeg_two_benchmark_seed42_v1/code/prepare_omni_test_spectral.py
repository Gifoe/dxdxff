"""One-shot official Omni test extraction after both models/thresholds freeze.

No prediction or outcome is inspected here. EDF-channel labels are retained
privately in the cache for the one official evaluation. Runtime artifacts must
not be committed to GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spectral_cache import connectivity_edges, spectral_patches
from train_crst import sha


def one(value):
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def zero(value):
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def label_of(outcome, resection, soz, good):
    if not one(good):
        return -1
    if one(outcome) and zero(resection):
        return 0  # Official normal branch has precedence.
    if one(soz):
        return 1  # Official pathological branch.
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-lock", type=Path, required=True)
    p.add_argument("--official-split", type=Path, required=True)
    p.add_argument("--train-val-split", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--signal-cache", type=Path, required=True)
    p.add_argument("--prior-code", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise RuntimeError("Invalid disjoint test shard")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (freeze["status"] != "FROZEN_BEFORE_OFFICIAL_TEST" or
            freeze["protocol_sha256"] != sha(args.protocol) or
            freeze["training_lock_sha256"] != sha(args.training_lock) or
            set(freeze["models"]) != {"CRST-0", "CRST-FULL"} or
            freeze["official_test_accessed_before_freeze"]):
        raise RuntimeError("Both checkpoints and thresholds must be frozen before test")
    if sha(args.official_split) != lock["splits"]["omni_official_split_sha256"]:
        raise RuntimeError("Official split SHA differs")
    for obj in freeze["models"].values():
        path = Path(obj["checkpoint_path"])
        if not path.is_file() or sha(path) != obj["checkpoint_sha256"]:
            raise RuntimeError("Frozen model checkpoint changed")
    train_roles = pd.read_csv(args.train_val_split)
    train_ids = set(train_roles["patient"].astype(str))
    if len(train_ids) != 141:
        raise RuntimeError("Frozen train patient membership changed")
    sys.path.insert(0, str(args.prior_code))
    from prepare_official_cnn_waveforms import physical_signal, windows, flag

    rows = pd.read_csv(args.official_split)
    rows = rows.loc[(rows["split"] == "test") & (rows["dataset"] != "Multicenter") &
                    (pd.to_numeric(rows["frequency"], errors="coerce") > 900) &
                    rows["interictal"].map(flag) &
                    (pd.to_numeric(rows["length"], errors="coerce") >= 62)].copy()
    if set(rows["patient_name"].astype(str)) & train_ids:
        raise RuntimeError("Official test and train patient overlap")
    # These counts are an identity gate from the frozen historical comparator.
    if rows["patient_name"].nunique() != 96:
        raise RuntimeError("Historical 96-patient official test membership changed")
    args.output.mkdir(parents=True, exist_ok=True)
    converted, skipped = 0, 0
    for ordinal, row in enumerate(rows.iloc[args.shard_index::args.shards].itertuples(index=False), 1):
        rel = Path(str(row.edf_name))
        if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() != ".edf":
            raise RuntimeError("Unsafe official relative EDF path")
        edf_hash = hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24]
        path = args.output / f"edf_{edf_hash}.npz"
        marker = args.output / f"edf_{edf_hash}.json"
        if path.exists() and marker.exists():
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if (prior.get("protocol_sha256") != sha(args.protocol) or
                    prior.get("freeze_sha256") != sha(args.freeze) or
                    prior.get("sha256") != sha(path)):
                raise RuntimeError("Private test cache resume identity mismatch")
            converted += 1
            continue
        if path.exists() or marker.exists():
            raise RuntimeError("Partial private test cache file needs inspection")
        sidecar = args.source / Path(str(rel).replace("_ieeg.edf", "_channels.tsv"))
        h5path = args.signal_cache / rel.with_suffix(".edf.h5")
        if not sidecar.is_file() or not h5path.is_file():
            raise RuntimeError("Missing frozen Omni sidecar/cache")
        channels = pd.read_csv(sidecar, sep="\t")
        with h5py.File(h5path, "r") as h5:
            if str(h5.attrs["dataset_revision"]) != lock["splits"]["omni_dataset_revision"]:
                raise RuntimeError("Frozen source revision mismatch")
            meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
            headers = meta["signal_headers"]
            by_name = {str(h["label"]): i for i, h in enumerate(headers)}
            good = channels.loc[channels["good"].map(flag) &
                                channels["name"].astype(str).isin(by_name)].copy()
            if good.empty or good["name"].duplicated().any():
                raise RuntimeError("No unique official good signal channels")
            names = good["name"].astype(str).tolist()
            labels = np.asarray([label_of(row.outcome, ch.resection, ch.soz, ch.good)
                                 for ch in good.itertuples(index=False)], dtype=np.int8)
            if np.all(labels < 0):
                skipped += 1
                print(f"test shard={args.shard_index} ordinal={ordinal} no_label", flush=True)
                continue
            rates = {float(headers[by_name[name]]["sample_frequency"]) for name in names}
            if len(rates) != 1:
                raise RuntimeError("Mixed source rates in an EDF")
            data = np.stack([physical_signal(h5, by_name[name], headers[by_name[name]])
                             for name in names])
        starts = windows(data.shape[1], train=False, rng=random.Random(0))
        if not starts:
            raise RuntimeError("No complete official test clip")
        n, c = len(starts), len(names)
        patches = np.zeros((n, c, 59, 64, 8), dtype=np.float16)
        edges = np.zeros((n, c, c, 15), dtype=np.float16)
        bands = np.zeros((n, 5), dtype=np.uint8)
        for clip_idx, start in enumerate(starts):
            wave = data[:, start:start+60000]
            if wave.shape != (c, 60000):
                raise RuntimeError("Incomplete official 60s test clip")
            spec, fmask, _ = spectral_patches(wave, 1000)
            relation, available = connectivity_edges(wave, 1000)
            patches[clip_idx] = spec.astype(np.float16)
            edges[clip_idx] = relation.astype(np.float16)
            bands[clip_idx] = available.astype(np.uint8)
        partial = path.with_suffix(".npz.partial")
        with partial.open("wb") as f:
            np.savez(f, patches=patches, edges=edges,
                     window_mask=np.ones((n, c, 59), bool), frequency_mask=fmask,
                     connectivity_band_mask=bands, labels=labels,
                     channel_names=np.asarray(names), starts=np.asarray(starts, np.int64),
                     edf=rel.as_posix(), patient=str(row.patient_name),
                     dataset=str(row.dataset), official_split="test")
        os.replace(partial, path)
        marker.write_text(json.dumps({"protocol_sha256": sha(args.protocol),
                                      "freeze_sha256": sha(args.freeze),
                                      "sha256": sha(path), "bytes": path.stat().st_size,
                                      "clips": n, "good_channels": c,
                                      "labeled_channels": int((labels >= 0).sum())},
                                     sort_keys=True) + "\n", encoding="utf-8")
        converted += 1
        print(f"test shard={args.shard_index} ordinal={ordinal} converted clips={n} channels={c}",
              flush=True)
    print(json.dumps({"shard": args.shard_index, "converted_edfs": converted,
                      "skipped_unlabeled": skipped,
                      "freeze_sha256": sha(args.freeze)}), flush=True)


if __name__ == "__main__":
    main()
