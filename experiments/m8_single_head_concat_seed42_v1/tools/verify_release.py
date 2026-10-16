"""Standard-library anonymous release check; never loads private patient data."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PUB = HERE / "results"
METRICS = ["accuracy", "auroc", "auprc", "macro_f1", "f1", "nez_f1", "mrr", "hit_at_1"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def obj(name):
    return json.loads((PUB / name).read_text(encoding="utf-8"))


def rows(name):
    with (PUB / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert abs(float(a) - float(b)) <= 1e-12, (a, b)


def main():
    manifest = obj("RELEASE_MANIFEST.json")
    assert len(manifest["server_public_files"]) == 36
    for name, expected in manifest["server_public_files"].items():
        path = PUB / name
        assert path.resolve().is_relative_to(PUB.resolve())
        assert sha(path) == expected, name
    independent = obj("INDEPENDENT_VERIFICATION.json")
    assert independent["status"] == "PASS"
    for name, expected in independent["files"].items():
        assert sha(PUB / name) == expected, name
    training = obj("TRAINING_VERIFICATION.json")
    choice = obj("CANDIDATE_FROZEN.json")
    assert training["status"] == "PASS"
    assert training["models"] == 25 and training["epochs"] == 334
    assert training["maximum_logit_error"] == 0
    assert training["source"] == choice["source"]
    for name, expected in training["source"].items():
        assert sha(HERE / name) == expected, name
    assert choice["status"] == "SELECTED_BEFORE_NEW_OUTER"
    assert choice["selected_model"] == "S4" and choice["Outer_used"] is False
    assert choice["model_freeze_sha256"] == training["model_freeze_sha256"]
    source_audits = {
        "TRAINED_INPUT_ISOLATION_AUDIT.json": "tools/verify_actual_masks.py",
        "INDEPENDENT_COMPLEXITY_AUDIT.json": "tools/verify_complexity_and_counts.py",
    }
    for audit, source in source_audits.items():
        value = obj(audit)
        assert value["status"] == "PASS" and value["source_sha256"] == sha(HERE / source)
    recovery = obj("COMPLEXITY_RECOVERY_AUDIT.json")
    assert recovery["status"] == "PASS" and recovery["original_source_preserved"] is True
    assert recovery["new_models_or_training_changed"] is False
    assert recovery["original_source_sha256"] == sha(HERE / "complexity.py")
    assert recovery["recovery_source_sha256"] == sha(HERE / "tools/complexity_recovery.py")
    assert recovery["candidate_sha256_unchanged"] == sha(PUB / "CANDIDATE_FROZEN.json")
    status = obj("OUTER_STATUS.json")
    decision = obj("FINAL_DECISION.json")
    assert status["status"] == "ALL80_MAIN_AND_DIAGNOSTIC_SCORES_SEALED"
    assert status["selected_model"] == decision["selected_model"] == "S4"
    assert status["freeze_sha256"] == decision["freeze_sha256"]
    assert status["seal_sha256"] == decision["seal_sha256"]
    assert independent["Outer_patient_rows"] == 1280
    assert independent["development_appearances"] == 520
    assert independent["paired_interval_metrics"] == 369
    assert independent["adjusted_intervals"] == 16
    assert independent["absolute_intervals"] == 216
    assert independent["all5_reference_metrics_reproduced"] is True
    main_rows = rows("MAIN_RESULTS.csv")
    assert len(main_rows) == 10 and len(rows("DIAGNOSTIC_RESULTS.csv")) == 6
    table = {r["model"]: r for r in main_rows}
    assert set(table) == {"S0", "S1", "S2", "S3", "S4", "REF_D1", "REF_M8", "REF_M8_NO_VAR", "REF_RAW_ONLY", "REF_LATE_FUSION"}
    for r in main_rows:
        assert (r["unit"], int(r["unique_patients"]), int(r["channels"])) == ("patient_macro", 80, 7635)
        assert all(math.isfinite(float(r[m])) and 0 <= float(r[m]) <= 1 for m in METRICS)
    for r in decision["metrics"]:
        close(r["candidate"], table["S4"][r["metric"]])
        close(r["original_M8"], table["REF_M8"][r["metric"]])
        close(r["delta"], r["candidate"] - r["original_M8"])
        assert r["passes"] == (r["candidate"] >= r["original_M8"])
    failed = [r for r in decision["metrics"] if not r["passes"]]
    assert decision["failed_metrics"] == failed
    assert [r["metric"] for r in failed] == ["accuracy", "auroc", "macro_f1", "nez_f1"]
    assert decision["status"] == "FAIL_STRICT_M8_TARGET"
    assert decision["all8_point_estimate_pass"] is False
    assert decision["all4_primary_classification_pass"] is False
    assert decision["scientifically_confirmed_noninferiority"] is False
    assert decision["scientific_status"] == "EXPLORATORY_REPEATED_TEST"
    for r in rows("METRIC_COVERAGE.csv"):
        assert int(r["finite_appearances"]) == int(r["total_appearances"]) == 80
    selections = rows("MODEL_SELECTION.csv")
    assert len(selections) == 25 and sum(int(r["epochs_run"]) for r in selections) == 334
    assert len(rows("TRAINING_HISTORY.csv")) == 334
    assert len({(r["model"], r["fold"]) for r in selections}) == 25
    counts = rows("CONFUSION_COUNTS.csv")
    assert len(counts) == 160
    for r in counts:
        assert int(r["TP"]) + int(r["FN"]) == int(r["positives"])
        assert int(r["TN"]) + int(r["FP"]) == int(r["negatives"])
        assert int(r["positives"]) + int(r["negatives"]) == int(r["n"])
    # Deny identity/prediction-level fields and private/binary artifacts.
    forbidden = {"patient", "patient_id", "channel", "channel_id", "y", "label", "probability", "prediction", "embedding", "path", "checkpoint", "indices"}
    for path in PUB.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix in {".json", ".csv"} and "PRIVATE" not in path.name
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                assert not set(reader.fieldnames or []) & forbidden, path.name
    report = (HERE / "REPORT.md").read_text(encoding="utf-8")
    assert decision["failure_boundary"] in report
    assert decision["status"] in report
    print("SINGLE_HEAD_ANONYMOUS_RELEASE_PASS 36 server artifacts; 25 fits; 10 models; S4 FAIL_STRICT_M8_TARGET")


if __name__ == "__main__":
    main()
