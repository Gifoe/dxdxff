"""Original A0 optimizer/selection; only fixed channel BCE multipliers differ."""
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from baseline import (PRMLP, state_hash, seed_all, restore_rng, rng_state,
    groups, select_threshold, torch_write, json_write, sha)


def train_weighted(x, y, patient, tr, va, cell, seed, initial, bind, device,
                   arm, weights, max_epochs=30, stop_after=None):
    cell = Path(cell); cell.mkdir(parents=True, exist_ok=True)
    last = cell / 'LAST_PRIVATE.pt'; complete = cell / 'COMPLETE_PRIVATE.json'
    binding = {'source': bind, 'seed': seed, 'arm': arm, 'initial_hash': state_hash(initial)}
    assert np.isfinite(weights[tr]).all() and (weights[tr] > 0).all()
    model = PRMLP().to(device); model.load_state_dict(initial); seed_all(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    positive = max(int(y[tr].sum()), 1); negative = max(int((1-y[tr]).sum()), 1)
    pos = torch.tensor(negative / positive, device=device)
    history = []; best_key = None; best = None; stale = 0; start = 1
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        assert saved['binding'] == binding, 'Resume source/initial/weights/arm mismatch'
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng']); history = saved['history']; best = saved['best']
        best_key = saved['best_key']; stale = saved['stale']; start = saved['epoch'] + 1
        if complete.exists():
            meta = json.loads(complete.read_text()); assert meta['binding'] == binding
            assert sha(last) == meta['last_sha256'] and sha(cell / 'BEST_PRIVATE.pt') == meta['best_sha256']
            model.load_state_dict(best['model']); model.eval(); return model, best, history
        if saved['epoch'] >= 6 and stale >= 6: start = max_epochs + 1
    tx = torch.as_tensor(x, device=device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    tw = torch.as_tensor(weights, dtype=torch.float32, device=device)
    train_groups = [tr[ix] for ix in groups(patient[tr])]
    for epoch in range(start, max_epochs + 1):
        model.train(); losses = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_groups))
        for position in order:
            ix = train_groups[int(position)]
            opt.zero_grad(set_to_none=True)
            # Native BCE weight argument is exactly mean(weight * per-channel
            # BCE), retaining the original mean-reduction backward order.
            loss = F.binary_cross_entropy_with_logits(model(tx[ix]), ty[ix],
                weight=tw[ix], pos_weight=pos, reduction='mean')
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite weighted BCE')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): score = torch.sigmoid(model(tx[va])).cpu().numpy()
        selected = select_threshold(y[va], score, patient[va])
        key = (selected['macro_f1'], selected['ez_f1'], -epoch)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), **selected,
                        'weights_active': arm != 'A0', 'labels_unchanged': True})
        if best_key is None or key > best_key:
            best_key = key; stale = 0
            best = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'epoch': epoch, **selected}
        else: stale += 1
        torch_write(last, {'binding': binding, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'rng': rng_state(), 'epoch': epoch, 'stale': stale, 'best_key': best_key, 'best': best, 'history': history})
        print(f'EPOCH {cell.parent.name}/{arm} epoch={epoch} val={key[0]:.6f} best={best["epoch"]}', flush=True)
        if stop_after and epoch == stop_after: return None, best, history
        if epoch >= 6 and stale >= 6: break
    assert best is not None
    torch_write(cell / 'BEST_PRIVATE.pt', {'binding': binding, **best})
    json_write(complete, {'binding': binding, 'last_sha256': sha(last),
        'best_sha256': sha(cell / 'BEST_PRIVATE.pt'), 'selected_epoch': best['epoch'], 'threshold': best['threshold']})
    model.load_state_dict(best['model']); model.eval()
    return model, best, history
