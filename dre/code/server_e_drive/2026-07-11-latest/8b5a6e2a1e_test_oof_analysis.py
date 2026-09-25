from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from task1_aaai.oof_analysis import analyze_oof_ledgers


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger_rows = []
    for seed in (42, 52, 62):
        for fold, (subject, center) in enumerate((("hup:A", "hup"), ("lzu:B", "lzu")), start=1):
            for channel, label in (("A1", 0), ("B1", 1), ("C1", 1)):
                prq = .2 if label == 0 else .8
                bcr = .3 if label == 0 else .7
                ledger_rows.append({"seed": seed, "outer_fold": fold, "subject_id": subject, "center": center, "channel_key": channel, "label_nez": label, "p2_score_nez": prq, "v3_score_nez": bcr, "fused_score_nez": .8 * prq + .2 * bcr})
    thresholds = pd.DataFrame([{"seed": seed, "outer_fold": fold, "model": model, "threshold": .5, "threshold_source": "validation_only"} for seed in (42, 52, 62) for fold in (1, 2) for model in ("PRQ-Net", "BCR-Net", "CDEL")])
    return pd.DataFrame(ledger_rows), thresholds


def test_oof_analysis_writes_patient_level_outputs(tmp_path: Path) -> None:
    ledger, thresholds = _inputs()
    audit = analyze_oof_ledgers(ledger=ledger, thresholds=thresholds, output_dir=tmp_path / "new", bootstrap_samples=20)
    assert audit["status"] == "passed"
    assert (tmp_path / "new" / "tables" / "task1_ranking_metrics.csv").is_file()
    assert (tmp_path / "new" / "figures" / "paired_bootstrap_forest.png").is_file()
    summary = pd.read_csv(tmp_path / "new" / "tables" / "paired_bootstrap_summary.csv")
    assert set(summary.comparison) == {"CDEL-PRQ-Net", "CDEL-BCR-Net"}
    assert set(summary.n_patients) == {2}


def test_oof_analysis_rejects_non_validation_threshold(tmp_path: Path) -> None:
    ledger, thresholds = _inputs()
    thresholds.loc[0, "threshold_source"] = "test"
    with pytest.raises(ValueError, match="validation_only"):
        analyze_oof_ledgers(ledger=ledger, thresholds=thresholds, output_dir=tmp_path / "new", bootstrap_samples=2)


def test_oof_analysis_refuses_existing_output(tmp_path: Path) -> None:
    ledger, thresholds = _inputs()
    output = tmp_path / "new"; output.mkdir(); (output / "old.csv").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        analyze_oof_ledgers(ledger=ledger, thresholds=thresholds, output_dir=output, bootstrap_samples=2)
