"""Synthetic invariants for the only changed feature-reference operation."""

import numpy as np

from extract_cross_channel_features import cross_views


def main():
    base = np.stack([np.full((59, 9), value, dtype=np.float32)
                     for value in (1.0, 2.0, 10.0)], axis=1)
    output = cross_views(base)
    assert output.shape == (59, 3, 36)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output[:, :, :9], base)
    np.testing.assert_allclose(output[:, 1, 9:18], 0.0, atol=1e-6)
    np.testing.assert_allclose(output[:, 2, 9:18], 8.0, atol=1e-6)
    assert float(output[0, 2, 18]) > 0
    assert float(output[0, 2, 27]) > 0
    perm = np.asarray([2, 0, 1])
    np.testing.assert_allclose(cross_views(base[:, perm])[:, np.argsort(perm)], output,
                               rtol=1e-6, atol=1e-6)
    # Unlike v1's temporal-self median, persistent channel abnormality remains.
    print({"pass": True, "persistent_channel_delta": float(output[0, 2, 9]),
           "channel_permutation_equivariant": True})


if __name__ == "__main__":
    main()
