"""Read only signal header metadata for the filtered official cohort."""

import json
from collections import Counter
from pathlib import Path

import h5py
import pandas as pd


def main():
    root = Path("F:/Omni-iEEG/signal_cache")
    cohort = pd.read_csv(Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v1/audit/OMNI_COHORT_AUDIT.csv"))
    units = Counter()
    mixed_rates = []
    examples = []
    for row in cohort.itertuples(index=False):
        path = root / Path(row.edf).with_suffix(".edf.h5")
        with h5py.File(path, "r") as handle:
            meta = json.loads(handle["metadata_json"][()].decode("utf-8"))
        headers = meta["signal_headers"]
        rates = {str(h.get("sample_frequency", h.get("sample_rate"))) for h in headers}
        if len(rates) != 1:
            mixed_rates.append({"edf": row.edf, "rates": sorted(rates)})
        for h in headers:
            units[str(h.get("dimension"))] += 1
        if len(examples) < 5:
            examples.append({"edf": row.edf, "header": headers[0], "count": len(headers)})
    print(json.dumps({"units": dict(units), "mixed_rate_edfs": mixed_rates,
                      "header_examples": examples}, indent=2))


if __name__ == "__main__":
    main()
