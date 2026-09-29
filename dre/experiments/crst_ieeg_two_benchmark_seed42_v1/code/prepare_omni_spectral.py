"""Private Omni TRAIN-only 60-s all-good-channel CRST spectral cache.

The official test split is rejected by this extraction entry point. A separate
frozen-test entry point may be used only after checkpoint/threshold freeze.
No private EDF-channel records or spectral arrays may be committed to GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spectral_cache import connectivity_edges, spectral_patches


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-split", type=Path, required=True)
    p.add_argument("--train-val-split", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--signal-cache", type=Path, required=True)
    p.add_argument("--prior-code", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid disjoint shard")
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    if digest(args.official_split) != lock["splits"]["omni_official_split_sha256"]:
        raise RuntimeError("Frozen official split SHA changed")
    sys.path.insert(0, str(args.prior_code))
    from prepare_official_cnn_waveforms import physical_signal, windows, flag  # noqa: E402
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

    def official_pathology_label(outcome, resection, soz, good):
        if not one(good):
            return None
        if one(outcome) and zero(resection):
            return 0
        if one(soz):
            return 1
        return None
    roles = pd.read_csv(args.train_val_split).set_index("patient")["role"].to_dict()
    if len(roles) != 141 or set(roles.values()) != {"inner_train", "inner_val"}:
        raise RuntimeError("Frozen 141-patient inner role file changed")
    rows = pd.read_csv(args.official_split)
    rows = rows.loc[(rows["split"] == "train") & (rows["dataset"] != "Multicenter") &
                    (pd.to_numeric(rows["frequency"], errors="coerce") > 900) &
                    rows["interictal"].map(flag) &
                    (pd.to_numeric(rows["length"], errors="coerce") >= 62)].copy()
    rows = rows.loc[rows["patient_name"].astype(str).isin(roles)].copy()
    if rows["patient_name"].nunique() != 141:
        raise RuntimeError("Official eligible train-patient count changed")
    # The historical v2 eligible EDF count excludes zero-labeled runs. Preserve
    # that distinction while all good signal channels remain model context.
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_sha = digest(args.protocol)
    done = 0
    for ordinal, row in enumerate(rows.iloc[args.shard_index::args.shards].itertuples(index=False), 1):
        rel = Path(str(row.edf_name))
        if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() != ".edf":
            raise RuntimeError("Unsafe official relative EDF path")
        edf_hash = hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24]
        path = args.output / f"edf_{edf_hash}.npz"
        marker = args.output / f"edf_{edf_hash}.json"
        if path.exists() and marker.exists():
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("protocol_sha256") != protocol_sha or prior.get("sha256") != digest(path):
                raise RuntimeError("Private Omni resume mismatch")
            done += 1
            print(f"omni train shard={args.shard_index} ordinal={ordinal} reused", flush=True)
            continue
        if path.exists() or marker.exists():
            raise RuntimeError("Partial private Omni file needs inspection")
        sidecar = args.source / Path(str(rel).replace("_ieeg.edf", "_channels.tsv"))
        h5path = args.signal_cache / rel.with_suffix(".edf.h5")
        if not sidecar.is_file() or not h5path.is_file():
            raise RuntimeError("Missing frozen Omni sidecar/cache")
        channels = pd.read_csv(sidecar, sep="\t")
        with h5py.File(h5path, "r") as h5:
            if str(h5.attrs["dataset_revision"]) != lock["splits"]["omni_dataset_revision"]:
                raise RuntimeError("Omni source revision mismatch")
            meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
            headers = meta["signal_headers"]
            by_name = {str(h["label"]): i for i, h in enumerate(headers)}
            good = channels.loc[channels["good"].map(flag) &
                                channels["name"].astype(str).isin(by_name)].copy()
            if good.empty or good["name"].duplicated().any():
                raise RuntimeError("No unique official good signal channels")
            names = good["name"].astype(str).tolist()
            labels = np.asarray([official_pathology_label(row.outcome, ch.resection,
                                                           ch.soz, ch.good)
                                 for ch in good.itertuples(index=False)], dtype=object)
            labels = np.asarray([-1 if x is None else x for x in labels], dtype=np.int8)
            if np.all(labels < 0):
                print(f"omni train shard={args.shard_index} ordinal={ordinal} no_labeled_channel", flush=True)
                continue
            rates = {float(headers[by_name[name]]["sample_frequency"]) for name in names}
            if len(rates) != 1:
                raise RuntimeError("Mixed source rates in an EDF")
            data = np.stack([physical_signal(h5, by_name[name], headers[by_name[name]])
                             for name in names])
        rng_seed = int.from_bytes(hashlib.sha256(("42|" + rel.as_posix()).encode()).digest()[:8], "big")
        import random
        starts = windows(data.shape[1], train=True, rng=random.Random(rng_seed))
        if not starts:
            raise RuntimeError("No complete frozen 60-s train clips")
        n, c = len(starts), len(names)
        patches = np.zeros((n, c, 59, 64, 8), dtype=np.float16)
        edges = np.zeros((n, c, c, 15), dtype=np.float16)
        bands = np.zeros((n, 5), dtype=np.uint8)
        for clip_idx, start in enumerate(starts):
            wave = data[:, start:start + 60000]
            if wave.shape != (c, 60000):
                raise RuntimeError("Incomplete 60-s train clip")
            spec, fmask, wmask = spectral_patches(wave, 1000)
            relation, available = connectivity_edges(wave, 1000)
            patches[clip_idx] = spec.astype(np.float16)
            edges[clip_idx] = relation.astype(np.float16)
            bands[clip_idx] = available.astype(np.uint8)
        partial = path.with_suffix(".npz.partial")
        with partial.open("wb") as f:
            np.savez(f, patches=patches, edges=edges, window_mask=np.ones((n, c, 59), bool),
                     frequency_mask=fmask, connectivity_band_mask=bands,
                     labels=labels, channel_names=np.asarray(names),
                     starts=np.asarray(starts, dtype=np.int64), edf=rel.as_posix(),
                     patient=str(row.patient_name), dataset=str(row.dataset),
                     validation_role=roles[str(row.patient_name)])
        os.replace(partial, path)
        marker.write_text(json.dumps({"protocol_sha256": protocol_sha,
                                      "sha256": digest(path), "bytes": path.stat().st_size,
                                      "clips": n, "good_channels": c,
                                      "labeled_channels": int((labels >= 0).sum())},
                                     sort_keys=True) + "\n", encoding="utf-8")
        done += 1
        print(f"omni train shard={args.shard_index} ordinal={ordinal} converted clips={n} channels={c}", flush=True)
    print(json.dumps({"shard": args.shard_index, "complete_edfs": done}), flush=True)


if __name__ == "__main__":
    main()
