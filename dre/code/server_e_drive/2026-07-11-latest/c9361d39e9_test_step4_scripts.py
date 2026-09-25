from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Step4ScriptTests(unittest.TestCase):
    def _read_script(self, name: str) -> str:
        return (ROOT / "scripts" / name).read_text(encoding="utf-8")

    def test_step4_training_scripts_keep_static_features_with_abs_part(self):
        for name in ("run_step4a_clinical_core_seed42.ps1", "run_step4b_clinical_burst_seed42.ps1"):
            script = self._read_script(name)
            self.assertIn('--physics_feature_parts "abs,zdelta,delta"', script)

    def test_step4_cache_scripts_use_low_frequency_spectral_settings(self):
        for name in ("build_step4a_clinical_core_cache.ps1", "build_step4b_clinical_burst_cache.ps1"):
            script = self._read_script(name)
            self.assertIn("--spectral-min-freq 1.0", script)
            self.assertIn("--spectral-max-freq 150.0", script)
            self.assertIn("--reader-bandpass-low 1.0", script)
            self.assertIn("--reader-bandpass-high 150.0", script)

    def test_step4_cache_scripts_run_all_centers_with_lzu_single_worker(self):
        for name in ("build_step4a_clinical_core_cache.ps1", "build_step4b_clinical_burst_cache.ps1"):
            script = self._read_script(name)
            self.assertIn('[string]$Centers = "lzu,hup,multicenter,pediatric,all"', script)
            self.assertIn("--low-memory-centers lzu", script)
            self.assertIn("--low-memory-num-workers 1", script)

    def test_m1_control_on_step4_cache_script_uses_original_features_with_abs_part(self):
        script = self._read_script("run_m1_control_on_step4a_cache_seed42.ps1")
        self.assertIn("neuroez_c_four_center_caches_step4a_clinical_core", script)
        self.assertIn('--physics_state_features "log_bp_high_gamma,line_length_per_sec,rms,variance"', script)
        self.assertIn('--physics_feature_parts "abs,zdelta,delta"', script)


if __name__ == "__main__":
    unittest.main()
