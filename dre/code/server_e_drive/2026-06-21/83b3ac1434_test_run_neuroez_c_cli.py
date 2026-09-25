from __future__ import annotations

import unittest

from run_neuroez_c import build_parser


class RunNeuroEZCCLITests(unittest.TestCase):
    def test_ez_ranking_flags_are_disabled_by_default(self):
        args = build_parser().parse_args([])

        self.assertFalse(args.use_ez_ranking_loss)
        self.assertEqual(args.ez_ranking_loss_weight, 0.0)
        self.assertEqual(args.ez_ranking_margin, 0.10)

    def test_ez_ranking_flags_can_be_enabled_explicitly(self):
        args = build_parser().parse_args(
            [
                "--use_ez_ranking_loss",
                "--ez_ranking_loss_weight",
                "0.05",
                "--ez_ranking_margin",
                "0.20",
            ]
        )

        self.assertTrue(args.use_ez_ranking_loss)
        self.assertEqual(args.ez_ranking_loss_weight, 0.05)
        self.assertEqual(args.ez_ranking_margin, 0.20)


if __name__ == "__main__":
    unittest.main()
