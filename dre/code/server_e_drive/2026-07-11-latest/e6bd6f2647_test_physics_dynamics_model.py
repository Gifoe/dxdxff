from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from neuroez_c.model import NeuroEZCModel


def _args(**overrides):
    values = {
        "model_dim": 16,
        "num_heads": 2,
        "dropout": 0.0,
        "use_channel_attention": True,
        "use_patient_relative_z": True,
        "use_physics_dynamics": True,
        "physics_model_dim": None,
        "physics_num_heads": None,
        "physics_gate_init": -3.0,
        "physics_loss_weight": 0.0,
        "physics_source_sparse_weight": 0.0,
        "physics_velocity_l2_weight": 0.0,
        "physics_detach_next_target": True,
        "use_diffusion_residual": False,
        "diffusion_graph_source": "cache_or_functional",
        "diffusion_gate_init": -4.0,
        "diffusion_beta_init": 0.10,
        "diffusion_source_sparse_weight": 0.0,
        "diffusion_residual_l2_weight": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(include_physics: bool = True):
    batch = {
        "b0_features": torch.randn(2, 1, 4, 5, 36),
        "labels": torch.ones(2, 5),
        "labels_nez": torch.ones(2, 5),
        "labels_ez": torch.zeros(2, 5),
        "channel_mask": torch.ones(2, 5, dtype=torch.bool),
        "seizure_mask": torch.ones(2, 1, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 1, 5, dtype=torch.bool),
        "window_mask": torch.ones(2, 1, 4, dtype=torch.bool),
    }
    if include_physics:
        batch["physics_features"] = torch.randn(2, 1, 4, 5, 8)
        batch["window_centers"] = torch.tensor([[[-30.0, -10.0, 10.0, 30.0]], [[-30.0, -10.0, 10.0, 30.0]]])
        batch["diffusion_adjacency"] = torch.ones(2, 1, 4, 5, 5)
    return batch


class PhysicsDynamicsModelTests(unittest.TestCase):
    def test_physics_enabled_model_forward_returns_dynamics_losses(self):
        out = NeuroEZCModel(_args())(_batch(include_physics=True))

        self.assertEqual(out["logits"].shape, (2, 5))
        self.assertEqual(out["score_nez"].shape, (2, 5))
        self.assertEqual(out["score_ez"].shape, (2, 5))
        for key in (
            "physics_next_state_loss",
            "physics_source_sparse_loss",
            "physics_velocity_l2_loss",
            "physics_gate_mean",
        ):
            self.assertIn(key, out)
            self.assertEqual(out[key].ndim, 0)
            self.assertTrue(torch.isfinite(out[key]))

    def test_physics_disabled_model_forward_does_not_require_physics_features(self):
        out = NeuroEZCModel(_args(use_physics_dynamics=False))(_batch(include_physics=False))

        self.assertEqual(out["logits"].shape, (2, 5))
        self.assertNotIn("physics_next_state_loss", out)

    def test_diffusion_enabled_model_forward_returns_diffusion_losses(self):
        out = NeuroEZCModel(_args(use_diffusion_residual=True))(_batch(include_physics=True))

        self.assertEqual(out["logits"].shape, (2, 5))
        for key in (
            "diffusion_source_sparse_loss",
            "diffusion_residual_l2_loss",
            "diffusion_gate_mean",
            "diffusion_beta",
            "diffusion_graph_density",
        ):
            self.assertIn(key, out)
            self.assertEqual(out[key].ndim, 0)
            self.assertTrue(torch.isfinite(out[key]))


if __name__ == "__main__":
    unittest.main()
