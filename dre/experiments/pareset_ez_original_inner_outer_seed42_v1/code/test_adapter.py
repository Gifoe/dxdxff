import unittest

import numpy as np

from build_patient_records import DESCRIPTORS, build


class AdapterTest(unittest.TestCase):
    def test_channel_alignment_padding_time_and_label_direction(self):
        names = list(DESCRIPTORS)
        def record(run_id, channels, labels, times, offset):
            n = len(channels)
            data = np.arange(len(times) * n * 9, dtype=np.float32).reshape(len(times), n, 9) + offset
            return {
                "subject_id": "hup:sample", "run_id": run_id,
                "channel_names_norm": channels, "labels": np.asarray(labels, dtype=np.float32),
                "sample": {"sample_id": run_id, "window_features": data,
                           "window_feature_names": names,
                           "window_relative_centers_sec": np.asarray(times, dtype=np.float32)},
            }
        payload = {
            "label_semantics": "EZ-positive",
            "window_feature_names": names,
            "patient_index": {"hup:sample": {
                "canonical_channels": ["A", "B", "C"],
                "labels": np.asarray([1, 0, 0], dtype=np.float32),
                "label_mask": np.ones(3, dtype=bool),
            }},
            "run_records": [
                record("r2", ["B", "C"], [1, 0], [-2, 0, 1], 100),
                record("r1", ["A", "C"], [1, 0], [-1, 1], 0),
            ],
        }
        patients, audit = build(payload, ["hup:sample"], names)
        patient = patients[0]
        self.assertEqual(patient["descriptors"].shape, (2, 3, 3, 9))
        self.assertEqual(patient["window_times"].tolist(), [[-1, 1, 0], [-2, 0, 1]])
        self.assertEqual(patient["label_nez"].tolist(), [0, 1, 1])
        self.assertEqual(patient["valid"][0, :, 0].tolist(), [True, True, False])
        self.assertFalse(patient["valid"][0, :, 1].any())
        np.testing.assert_array_equal(patient["descriptors"][0, 0, 0], np.arange(9))
        np.testing.assert_array_equal(patient["descriptors"][1, 0, 1], np.arange(9) + 100)
        self.assertEqual(audit["runs"], 2)
        self.assertEqual(audit["run_vs_patient_label_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
