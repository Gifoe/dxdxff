from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..functional_graph import normalize_channel_name
from .schema import P2NEZRecord


def records_from_p2_exports(exports: Sequence[dict[str, Any]]) -> tuple[list[P2NEZRecord], pd.DataFrame]:
    records: list[P2NEZRecord] = []; rows: list[dict] = []
    for value in exports:
        final = torch.sigmoid(torch.as_tensor(value["final_nez_logit"])[0].float()).numpy()
        direct_logit = value.get("direct_nez_logit")
        direct = torch.sigmoid(torch.as_tensor(direct_logit)[0].float()).numpy() if direct_logit is not None else None
        valid = np.asarray(torch.as_tensor(value["channel_mask"])[0].bool())
        target = np.asarray(torch.as_tensor(value["clinical_target_mask"])[0].bool())
        names = [normalize_channel_name(x) for x in value["channel_names"]]
        record = P2NEZRecord(str(value["patient_key"]), str(value["center"]), names, final, direct, valid, target); records.append(record)
        for index, name in enumerate(names):
            rows.append({"patient_key": record.patient_key, "center": record.center, "channel": name, "q_nez": final[index], "direct_nez_probability_audit": direct[index] if direct is not None else np.nan, "valid": int(valid[index]), "clinical_target": int(target[index])})
    return records, pd.DataFrame(rows)


def permute_nez(record: P2NEZRecord, seed: int) -> P2NEZRecord:
    rng = np.random.default_rng(seed); indices = np.where(record.valid_channel_mask)[0]; order = rng.permutation(indices)
    final = record.final_nez_probability.copy(); final[indices] = final[order]
    direct = record.direct_nez_probability.copy() if record.direct_nez_probability is not None else None
    if direct is not None: direct[indices] = direct[order]
    return P2NEZRecord(record.patient_key, record.center, list(record.channel_names), final, direct, record.valid_channel_mask.copy(), record.clinical_target_mask.copy())


__all__ = ["permute_nez", "records_from_p2_exports"]
