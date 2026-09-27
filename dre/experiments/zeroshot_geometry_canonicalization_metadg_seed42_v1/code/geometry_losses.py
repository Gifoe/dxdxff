"""Training-only R4 geometry losses. No target-patient input is accepted here."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _per_patient(h: torch.Tensor, y_ez: torch.Tensor, mask: torch.Tensor, *, normalized: bool):
    rows = []
    for i in range(h.shape[0]):
        valid = mask[i] & (y_ez[i] >= 0)
        ez, nez = valid & (y_ez[i] > .5), valid & (y_ez[i] <= .5)
        if not bool(ez.any()) or not bool(nez.any()):
            continue
        x = F.normalize(h[i], dim=-1, eps=1e-8) if normalized else h[i]
        rows.append((x[ez].mean(0), x[nez].mean(0), x[valid], y_ez[i][valid]))
    return rows


def direction_loss(h, y_ez, mask):
    rows = _per_patient(h, y_ez, mask, normalized=False)
    if len(rows) < 2:
        return h.sum() * 0
    directions = torch.stack([F.normalize(ez-nez, dim=0, eps=1e-8) for ez, nez, _, _ in rows])
    center = F.normalize(directions.mean(0), dim=0, eps=1e-8)
    return (1 - (directions * center).sum(-1)).mean()


def centroid_loss(h, y_ez, mask):
    rows = _per_patient(h, y_ez, mask, normalized=True)
    if len(rows) < 2:
        return h.sum() * 0
    ez = torch.stack([r[0] for r in rows]); nez = torch.stack([r[1] for r in rows])
    return ((ez-ez.mean(0)).square().sum(-1) + (nez-nez.mean(0)).square().sum(-1)).mean()


def cross_patient_supcon(h, y_ez, mask, tau):
    rows = _per_patient(h, y_ez, mask, normalized=True)
    if len(rows) < 2:
        return h.sum() * 0
    # Each patient/class is represented by one centroid so trial-rich patients
    # cannot dominate the auxiliary objective. Same-patient positives are absent.
    vectors = F.normalize(torch.stack([v for row in rows for v in row[:2]]), dim=-1, eps=1e-8)
    n = len(rows)
    logits = vectors @ vectors.T / tau
    losses = []
    for i in range(2*n):
        positive = torch.arange(2*n, device=h.device) % 2 == i % 2
        positive[i] = False
        negative = ~positive
        negative[i] = False
        if bool(positive.any()) and bool(negative.any()):
            denom = torch.logsumexp(logits[i][positive | negative], 0)
            losses.append(-(logits[i][positive] - denom).mean())
    return torch.stack(losses).mean() if losses else h.sum() * 0


def geometry_loss(name, h, batch, config):
    y, mask = batch["labels_ez"], batch["channel_mask"]
    if name == "Z1_DIRECTION_CANONICALIZATION":
        return direction_loss(h, y, mask)
    if name == "Z2_CLASS_CONDITIONAL_ALIGNMENT":
        return centroid_loss(h, y, mask)
    if name == "Z2B_CROSSPATIENT_SUPCON":
        return cross_patient_supcon(h, y, mask, float(config["tau"]))
    raise ValueError(name)
