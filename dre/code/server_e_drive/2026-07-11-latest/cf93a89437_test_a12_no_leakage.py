import unittest

from a12_vcsn.feature_registry import FeatureRegistry, FeatureRegistryError


class A12NoLeakageTests(unittest.TestCase):
    def test_registry_rejects_oracle_labels_and_center(self):
        registry = FeatureRegistry()
        registry.add("v3_score", group="v3", source="old_v3", uses_label=False)
        with self.assertRaises(FeatureRegistryError):
            registry.add("oracle_candidate_delta", group="oracle", source="ground_truth", uses_label=True)
        with self.assertRaises(FeatureRegistryError):
            registry.add("center", group="patient_context", source="metadata", uses_label=False)

