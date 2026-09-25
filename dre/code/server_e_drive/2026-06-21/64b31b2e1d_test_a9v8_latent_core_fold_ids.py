import unittest

import pandas as pd

from scripts.build_latent_core_targets import observed_fold_ids, split_fold_train_targets


class A9v8LatentCoreFoldIdTests(unittest.TestCase):
    def test_fold_train_targets_use_observed_fold_ids_not_zero_based_range(self):
        targets = pd.DataFrame(
            [
                {"patient_id": "p1", "fold_id": 1, "center": "lzu", "channel_name": "a"},
                {"patient_id": "p2", "fold_id": 2, "center": "hup", "channel_name": "a"},
                {"patient_id": "p3", "fold_id": 3, "center": "pediatric", "channel_name": "a"},
                {"patient_id": "p4", "fold_id": 4, "center": "multicenter", "channel_name": "a"},
                {"patient_id": "p5", "fold_id": 5, "center": "lzu", "channel_name": "a"},
            ]
        )

        self.assertEqual(observed_fold_ids(targets), [1, 2, 3, 4, 5])
        splits = split_fold_train_targets(targets)

        self.assertEqual(sorted(splits), [1, 2, 3, 4, 5])
        self.assertNotIn(0, splits)
        self.assertFalse((splits[1]["fold_id"].astype(int) == 1).any())
        self.assertFalse((splits[5]["fold_id"].astype(int) == 5).any())


if __name__ == "__main__":
    unittest.main()
