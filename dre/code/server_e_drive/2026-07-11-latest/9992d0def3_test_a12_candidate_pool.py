import unittest

from a12_vcsn.candidate_pool import CandidateConfig, build_candidate_pools
from a12_vcsn.suite import resolve_variant_feature_groups
from tests.test_a12_schema import raw_ledger


class A12CandidatePoolTests(unittest.TestCase):
    def test_candidates_use_only_v3_and_topology_fields(self):
        ledger, _ = __import__("a12_vcsn.schemas", fromlist=["build_canonical_ledger"]).build_canonical_ledger(raw_ledger())
        pools = build_candidate_pools(ledger, CandidateConfig(boundary_width=2))
        self.assertEqual(set(pools["eject_channel"]), {"A01REF", "B02"})
        self.assertEqual(set(pools["add_channel"]), {"A02REF", "B1"})
        self.assertFalse(any("clinical" in col for col in pools.columns))

    def test_preregistered_variants_select_incremental_feature_groups(self):
        available = {"v3", "anchor", "trajectory", "topology", "hnc", "patient_context"}
        self.assertEqual(set(resolve_variant_feature_groups("A12-V1", available)), {"v3", "patient_context"})
        self.assertEqual(set(resolve_variant_feature_groups("A12-V2", available)), {"v3", "anchor", "patient_context"})
        self.assertEqual(set(resolve_variant_feature_groups("A12-V3", available)), {"v3", "anchor", "trajectory", "patient_context"})
        self.assertEqual(set(resolve_variant_feature_groups("A12-V5", available)), available)
