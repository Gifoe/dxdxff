"""Private raw candidate bank; deterministic sparse sampling at record level."""
import json
from pathlib import Path

import numpy as np
import torch

from audit_raw import ROOT, norm
from common import sha
from sampling import select_windows


class RawBank:
    def __init__(self, patients=None):
        self.index = json.loads((ROOT/'RAW_INDEX_PRIVATE.json').read_text())
        self.bags = {}
        selected = sorted(self.index['patients']) if patients is None else sorted(set(map(str, patients)))
        for p in selected:
            row = self.index['patients'][p]
            assert sha(row['file']) == row['sha256']
            bag = torch.load(row['file'], map_location='cpu', weights_only=False)
            assert bag['binding'] == self.index['binding'] and bag['patient'] == p
            self.bags[p] = bag

    def sample(self, patient, channels, seed, epoch=None, shuffled=False, device='cpu'):
        b = self.bags[str(patient)]
        indices = {norm(c): i for i, c in enumerate(b['channels'])}
        ix = np.array([indices[norm(c)] for c in channels])
        if shuffled:
            ix = b['perm'][ix]
        wave, aux, mask = b['wave'][ix], b['aux'][ix], b['mask'][ix]
        C, S, _ = mask.shape
        picks = np.full((C, S, 12), -1, dtype=int)
        time = np.zeros((C, S, 12), np.float32)
        for s, (run, n) in enumerate(zip(b['runs'], b['counts'])):
            times = b['times'][s, :n]
            selected = select_windows(times, mask[:, s, :n].numpy(), str(patient), run, seed, epoch)
            picks[:, s] = selected
            time[:, s] = (times[np.maximum(selected, 0)] - times.min()) / max(float(np.ptp(times)), 1e-8)
        take = torch.from_numpy(np.maximum(picks, 0))
        ci = torch.arange(C)[:, None, None]
        si = torch.arange(S)[None, :, None]
        valid = mask[ci, si, take] & torch.from_numpy(picks >= 0)
        w = wave[ci, si, take] * valid.unsqueeze(-1)
        a = aux[ci, si, take] * valid.unsqueeze(-1)
        return tuple(x.to(device) for x in (w, a, valid, torch.from_numpy(time)))
