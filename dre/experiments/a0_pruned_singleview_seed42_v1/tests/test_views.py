import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import numpy as np
import torch
from folding import Compact,inflate
from views import configurations


def test_actual_manifest_views():
    torch.manual_seed(42);torch.set_num_threads(2)
    root=Path(__file__).parents[1]
    cfg=configurations(json.loads((root/'AUTHORITATIVE_FEATURE_MANIFEST.json').read_text())['columns'])
    for arm,records in cfg.items():
        net=torch.nn.Sequential(torch.nn.LayerNorm(88),torch.nn.Linear(88,96),torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(96,1)).double().eval()
        with torch.no_grad():net[0].weight.uniform_(.5,1.5);net[0].bias.uniform_(-.2,.2)
        m=Compact(net,records).eval().double();x=torch.randn(23,len(records),dtype=torch.float64)
        full=inflate(x,records)
        with torch.no_grad():assert (m(full)-net(full).squeeze(-1)).abs().max()<1e-10
        outside=[i for i in range(88) if i not in [q['index'] for q in records]];changed=full.clone();changed[:,outside]=torch.randn(23,len(outside),dtype=torch.float64)*1000
        assert torch.equal(m(full),m(changed))
        assert sum(p.numel() for p in m.parameters())==96*len(records)+193
    assert sum(q['multiplicity'] for q in cfg['DELTA_ONLY'])==14
