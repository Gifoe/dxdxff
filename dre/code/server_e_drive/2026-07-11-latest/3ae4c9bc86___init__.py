"""Strict orchestration for Task 1 confirmatory experiments.

This package intentionally delegates model training to the existing PRQ-Net and
BCR-Net entrypoints.  It owns only manifests, execution bookkeeping, analysis,
and fail-closed leakage checks.
"""

from .protocol import build_loco_manifests, build_reference_partition, validate_partition

__all__ = ["build_loco_manifests", "build_reference_partition", "validate_partition"]
