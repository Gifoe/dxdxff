from __future__ import annotations

import re
from typing import Sequence


FAMILY_PRIORITY=("gamma","line_length","amplitude_variability","spectral_entropy","optional_hfo","optional_spike")
ALIASES={
 "gamma":("high_gamma","high-gamma","hgamma","30_80","30-80","bandpower_gamma","gamma"),
 "line_length":("line_length","linelength","line-length"),
 "amplitude_variability":("absolute_amplitude","amplitude","variance","std","rms","energy"),
 "spectral_entropy":("spectral_entropy","spec_entropy","entropy"),
 "optional_hfo":("fast_ripple","fast-ripple","ripple","hfo"),
 "optional_spike":("sharp_wave","sharp-wave","spike"),
}


def match_feature_families(names: Sequence[str]) -> tuple[dict[str,list[int]],list[dict[str,object]]]:
    groups={family:[] for family in FAMILY_PRIORITY}; audit=[]
    for index,name in enumerate(names):
        normalized=re.sub(r"[^a-z0-9]+","_",str(name).lower()).strip("_"); matched=None; rule=""
        for family in FAMILY_PRIORITY:
            for alias in ALIASES[family]:
                token=re.sub(r"[^a-z0-9]+","_",alias.lower()).strip("_")
                if token in normalized:
                    matched=family;rule=alias;groups[family].append(index);break
            if matched:break
        audit.append({"feature_name":str(name),"matched_family":matched or "unmatched","match_rule":rule,"used_in_primary_model":bool(matched),"missing_family":False})
    for family in FAMILY_PRIORITY:
        if not groups[family]: audit.append({"feature_name":"","matched_family":family,"match_rule":"","used_in_primary_model":False,"missing_family":True})
    if not groups["gamma"] and not groups["line_length"]: raise ValueError("NO_RELIABLE_DYNAMIC_FEATURE_FAMILY")
    return groups,audit


__all__=["ALIASES","FAMILY_PRIORITY","match_feature_families"]
