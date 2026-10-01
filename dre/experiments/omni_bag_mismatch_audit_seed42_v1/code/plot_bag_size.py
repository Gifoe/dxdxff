from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
frame = pd.read_csv(root / "PERFORMANCE_VS_BAG_SIZE.csv")
order = ["1", "2", "3", "5", "10", "20", "ALL"]
frame["k"] = frame.k.astype(str)
summary = frame.groupby("k").agg(
    auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
    ap_mean=("ap", "mean"), ap_sd=("ap", "std")).reindex(order)
x = np.arange(len(order))
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
for ax, metric, label, color in [(axes[0], "auroc", "AUROC", "#2864a6"), (axes[1], "ap", "Average precision", "#b44b3e")]:
    mean = summary[f"{metric}_mean"].to_numpy(); sd = summary[f"{metric}_sd"].fillna(0).to_numpy()
    ax.plot(x, mean, marker="o", lw=2, color=color)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.18, linewidth=0)
    ax.set_xticks(x, order); ax.set_xlabel("Segments sampled per EDF (K)"); ax.set_ylabel(label)
    ax.grid(axis="y", color="#dddddd", linewidth=.7); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Frozen mean-pooling performance versus bag size", fontsize=12)
fig.savefig(root / "PERFORMANCE_VS_BAG_SIZE.png", dpi=180)
