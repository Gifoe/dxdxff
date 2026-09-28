"""One complete official-train EDF through frozen A1 forward/backward only."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from train_v2 import (V2Bank, check_source, make_model,
                      patient_equal_weighted_bce, to_device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    full = pd.read_csv(args.cohort)
    eligible = full.loc[(full["official_split"] == "train") &
                        (full["official_labeled_channels"] > 0)]
    ready = eligible.loc[eligible["edf"].map(
        lambda value: (args.features / Path(value).with_suffix(".npz")).is_file())]
    if ready.empty:
        raise RuntimeError("No completed official-train EDF for smoke")
    with TemporaryDirectory() as temp:
        subset = Path(temp) / "cohort.csv"
        ready.head(1).to_csv(subset, index=False)
        bank = V2Bank(subset, args.features, None, protocol=args.protocol)
        patient = bank.patients[0]
        mean, std = bank.fit_normalizer([patient])
        collate, Model = check_source()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = make_model(Model, collate, bank, patient, mean, std, device)
        batch = to_device(collate([bank.example(patient, mean, std, epoch=1)]), device)
        output = model(batch)
        loss = patient_equal_weighted_bce(output["logits"], batch["labels_nez"],
                                          batch["labels_ez"], batch["channel_mask"])
        loss.backward()
        gradients = [bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
                     if parameter.grad is not None]
        if not np.isfinite(float(loss.detach().cpu())) or not gradients or not all(gradients):
            raise RuntimeError("A1 v2 smoke loss/gradients invalid")
        print(f"v2 official-train smoke PASS patient={patient} channels={len(bank.canonical[patient])} "
              f"loss={float(loss.detach().cpu()):.6f} finite_gradient_tensors={len(gradients)}")


if __name__ == "__main__":
    main()
