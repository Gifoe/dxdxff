"""Engineering-only checks for private frozen-test inference recovery."""

import json
import tempfile
from pathlib import Path

import evaluate_omni_frozen as audit


def main():
    calls = []

    def fake_evaluate(*args):
        patient = args[3][0]
        calls.append(patient)
        return None, {patient: {"labels": [0, 1], "scores": [0.2, 0.8],
                               "edf": ["a", "a"], "channel": ["x", "y"]}}

    audit.evaluate_raw = fake_evaluate
    with tempfile.TemporaryDirectory() as directory:
        runtime = Path(directory)
        arguments = ("RawCNN", object(), object(), object(), ["p1", "p2"],
                     None, object(), runtime, "freeze", "checkpoint",
                     "protocol", "official")
        first = audit.cached_patient_predictions(*arguments)
        assert calls == ["p1", "p2"]
        audit.evaluate_raw = lambda *args: (_ for _ in ()).throw(
            AssertionError("A verified patient must not be inferred again"))
        assert audit.cached_patient_predictions(*arguments) == first
        path = runtime / "omni" / "FROZEN_TEST_PATIENT_CACHE_PRIVATE" / "RawCNN_000.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["checkpoint_sha256"] = "changed"
        path.write_text(json.dumps(value), encoding="utf-8")
        try:
            audit.cached_patient_predictions(*arguments)
        except RuntimeError as error:
            assert "provenance mismatch" in str(error)
        else:
            raise AssertionError("Tampered patient cache was accepted")
    print("FROZEN_OMNI_RESUME_TEST_PASS")


if __name__ == "__main__":
    main()
