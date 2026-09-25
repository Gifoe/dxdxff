from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from neuroez_c.v3_qbc_protocol import read_outer_fold_manifest, validate_v3_qbc_protocol


class V3QBCProtocolTests(unittest.TestCase):
    def _files(self, root: Path, overlap: bool = False):
        subjects = [f"p{i}" for i in range(10)]
        allowed = root / "allowed.csv"
        manifest = root / "folds.csv"
        pd.DataFrame({"subject_id": subjects}).to_csv(allowed, index=False)
        rows = [{"subject_id": subject, "outer_fold": index % 5 + 1} for index, subject in enumerate(subjects)]
        if overlap:
            rows.append({"subject_id": "p0", "outer_fold": 2})
        pd.DataFrame(rows).to_csv(manifest, index=False)
        return subjects, allowed, manifest

    def test_fixed_manifest_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            subjects, allowed, manifest = self._files(Path(directory))
            audit = validate_v3_qbc_protocol(
                patient_ids=subjects, allowed_subjects_path=allowed,
                outer_fold_manifest_path=manifest, require_n_patients=10, seed=42,
            )
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["test_union_count"], 10)

    def test_duplicate_subject_across_folds_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = self._files(Path(directory), overlap=True)
            with self.assertRaises(ValueError):
                read_outer_fold_manifest(manifest)

    def test_wrong_patient_count_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            subjects, allowed, manifest = self._files(Path(directory))
            with self.assertRaises(RuntimeError):
                validate_v3_qbc_protocol(
                    patient_ids=subjects, allowed_subjects_path=allowed,
                    outer_fold_manifest_path=manifest, require_n_patients=80, seed=42,
                )


if __name__ == "__main__":
    unittest.main()
