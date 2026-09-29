"""Patient-equal supervised and train-partition-only SSL objectives."""

import torch
from torch import Tensor
from torch.nn import functional as F


def supervised_parts(logits: Tensor, labels: Tensor, record_logits: Tensor,
                     window_mask: Tensor, seed: int,
                     record_labels: Tensor | None = None):
    valid = labels[0] >= 0
    score = logits[0, valid]
    truth = labels[0, valid]
    if score.numel() == 0 and record_labels is None:
        raise RuntimeError("Patient has no official/historical labels")
    if record_labels is not None:
        observed = record_labels[0] >= 0
        classification_score = logits[0][None].expand_as(record_labels[0])[observed]
        classification_truth = record_labels[0][observed]
    else:
        classification_score, classification_truth = score, truth
    if classification_score.numel() == 0:
        raise RuntimeError("Patient has no official/historical classification labels")
    weights = torch.where(classification_truth > 0, 2.0, 1.0)
    bce = (F.binary_cross_entropy_with_logits(classification_score.float(),
                                               classification_truth.float(),
                                               reduction="none") * weights).mean()
    pos, neg = score[truth > 0], score[truth == 0]
    if pos.numel() and neg.numel():
        pairs = torch.cartesian_prod(torch.arange(len(pos), device=score.device),
                                     torch.arange(len(neg), device=score.device))
        if len(pairs) > 128:
            generator = torch.Generator(device=score.device).manual_seed(seed)
            pairs = pairs[torch.randperm(len(pairs), generator=generator,
                                          device=score.device)[:128]]
        pair = F.softplus(-(pos[pairs[:, 0]] - neg[pairs[:, 1]])).mean()
        # Stable one-patient listwise softmax: probability mass assigned to
        # positives. Does not mix ranks across patients.
        listwise = torch.logsumexp(score, 0) - torch.logsumexp(pos, 0)
    else:
        pair = bce.new_zeros(())
        listwise = bce.new_zeros(())
    present = window_mask.any(-1)[0]  # R,C
    per_record_score = record_logits[0]  # R,C
    cons_terms = []
    for channel in range(per_record_score.shape[1]):
        current = per_record_score[present[:, channel], channel]
        if len(current) >= 2:
            mean = current.mean(0, keepdim=True)
            cons_terms.append(F.smooth_l1_loss(current, mean.expand_as(current)))
    consistency = torch.stack(cons_terms).mean() if cons_terms else bce.new_zeros(())
    return {"bce": bce, "pair": pair, "list": listwise, "cons": consistency,
            "total": bce + 0.5 * pair + 0.25 * listwise + 0.05 * consistency}


def vicreg(x: Tensor, y: Tensor) -> Tensor:
    if x.ndim != 2 or x.shape != y.shape or len(x) < 2:
        return x.new_zeros(())
    invariance = F.mse_loss(x, y)
    z = torch.cat((x, y), dim=0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1 - std).mean()
    centered = z - z.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(z) - 1)
    covariance.fill_diagonal_(0)
    covariance = covariance.square().sum() / z.shape[1]
    return invariance + variance + covariance


def ssl_objective(model, patches: Tensor, frequency_mask: Tensor,
                  window_mask: Tensor, edges: Tensor, *, seed: int):
    gen = torch.Generator(device=patches.device).manual_seed(seed)
    token_valid = window_mask
    random_mask = torch.rand(token_valid.shape, generator=gen,
                             device=patches.device) < 0.4
    masked = token_valid & random_mask
    if not masked.any():
        masked = token_valid.clone()
    # The unmasked learned target is detached; only training partition is used.
    with torch.no_grad():
        target = model.tokenizer(patches, frequency_mask).detach()
    input_patch = patches.masked_fill(masked[..., None, None], 0)
    first = model(input_patch, frequency_mask, window_mask, edges, return_aux=True)
    reconstruction = model.mask_decoder(first["contextual"])
    mask_loss = F.smooth_l1_loss(reconstruction[masked].float(), target[masked].float())
    # Mild morphology-preserving amplitude and Gaussian-noise views, plus
    # sparse frequency masking. Both pass through identical shared encoder.
    amplitude = 1 + (torch.rand((1,), generator=gen, device=patches.device) - 0.5) * 0.1
    jitter = torch.randn(patches.shape, generator=gen, device=patches.device) * 0.002
    second_patch = (patches * amplitude + jitter).clamp_min(0)
    # One-window temporal jitter without circular wraparound.
    if bool(torch.rand((), generator=gen, device=patches.device) < 0.5):
        second_patch = torch.cat((second_patch[..., :1, :, :],
                                  second_patch[..., :-1, :, :]), dim=-3)
    frequency_drop = torch.rand((64,), generator=gen, device=patches.device) < 0.05
    second_patch[..., frequency_drop, :] = 0
    channel_drop = torch.rand(window_mask.shape[:-1], generator=gen,
                              device=patches.device) < 0.05
    second_patch = second_patch.masked_fill(channel_drop[..., None, None, None], 0)
    second_patch = second_patch * window_mask[..., None, None]
    second = model(second_patch, frequency_mask, window_mask, edges, return_aux=True)
    present = window_mask.any(-1)[0]
    within = vicreg(first["record_embedding"][0, present],
                    second["record_embedding"][0, present])
    by_record = first["record_embedding"][0]  # R,C,D
    cross_pairs = []
    for channel in range(by_record.shape[1]):
        available = by_record[present[:, channel], channel]
        if len(available) >= 2:
            cross_pairs.append(available[:2])
    if cross_pairs:
        pair = torch.stack(cross_pairs)
        cross = vicreg(pair[:, 0], pair[:, 1])
    else:
        cross = mask_loss.new_zeros(())
    return {"mask": mask_loss, "within": within, "cross": cross,
            "total": mask_loss + 0.5 * within + 0.5 * cross}
