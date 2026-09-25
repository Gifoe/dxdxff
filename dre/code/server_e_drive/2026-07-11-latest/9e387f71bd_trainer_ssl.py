"""SSL stage helpers; target encoder is always EMA-only."""
from __future__ import annotations
import torch
from .augment import augment
from .losses import ssl_losses

def ssl_step(model, seizure, optimizer, ema_decay):
    w, side, masked = augment(seizure["windows"], seizure["side_features"], seizure["window_mask"])
    online_view = {**seizure, "windows": w, "side_features": side}
    h1, z1 = model.online.encode_seizure(online_view, masked)
    with torch.no_grad(): h2, z2 = model.target.encode_seizure(seizure)
    loss, pieces = ssl_losses(model.predictor(z1), z2, ~masked & seizure["window_mask"], model.project(h1), model.target_project(h2))
    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); model.ema(ema_decay)
    return {"ssl_total": float(loss.detach()), **{f"ssl_{k}": float(v.detach()) for k,v in pieces.items()}}
