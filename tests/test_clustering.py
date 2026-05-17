"""
test_clustering.py
==================
Unit tests for ``KMeansClusterer``.

Targets under test:
    - ``__init__()``:    argument validation and exceptions
    - ``fit()``:         clustering execution and result object
    - ``get_labels()``:  label retrieval and the not-fitted error
    - ``get_result()``:  result retrieval and the not-fitted error
    - Numerical properties of the hand-rolled norm / normalization
      (``_l2_norm_batch`` / ``_normalize_rows``)
    - Reproducibility (same seed → identical result)

How to run:
    venv/bin/python3 -m unittest tests/test_clustering.py -v
"""

from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transforms.clustering import (
    ClusterResult,
    InvalidClusterCountError,
    KMeansClusterer,
    NotFittedError,
    UnfitVectorError,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _random_vectors(n: int = 30, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Generate a random embedding matrix for tests."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _separable_vectors(seed: int = 0) -> np.ndarray:
    """Generate a matrix with 3 clearly separated clusters (used to verify k=3)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]])
    blocks = []
    for c in centers:
        noise = rng.standard_normal((10, 2)) * 0.2
        blocks.append(c + noise)
    return np.vstack(blocks).astype(np.float32)


# ---------------------------------------------------------------------------
# __init__()
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):
    """Tests for ``KMeansClusterer.__init__()``."""

    def test_default_construction(self) -> None:
        """Construction succeeds with default arguments."""
        clusterer = KMeansClusterer()
        self.assertIsInstance(clusterer, KMeansClusterer)

    def test_custom_construction(self) -> None:
        """Construction succeeds with custom arguments."""
        KMeansClusterer(n_clusters=5, seed=7, max_iter=100)

    def test_invalid_n_clusters_zero_raises(self) -> None:
        """``n_clusters=0`` raises ``InvalidClusterCountError``."""
        with self.assertRaises(InvalidClusterCountError):
            KMeansClusterer(n_clusters=0)

    def test_invalid_n_clusters_negative_raises(self) -> None:
        """Negative ``n_clusters`` raises ``InvalidClusterCountError``."""
        with self.assertRaises(InvalidClusterCountError):
            KMeansClusterer(n_clusters=-1)

    def test_n_clusters_type_error(self) -> None:
        """Non-int ``n_clusters`` raises ``TypeError``."""
        with self.assertRaises(TypeError):
            KMeansClusterer(n_clusters="3")  # type: ignore[arg-type]

    def test_seed_type_error(self) -> None:
        """Non-int ``seed`` raises ``TypeError``."""
        with self.assertRaises(TypeError):
            KMeansClusterer(seed=3.14)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit(unittest.TestCase):
    """Tests for ``KMeansClusterer.fit()``."""

    def test_returns_cluster_result(self) -> None:
        """The return value is a ``ClusterResult``."""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        result = clusterer.fit(vectors)
        self.assertIsInstance(result, ClusterResult)

    def test_labels_shape(self) -> None:
        """``labels.shape`` is (N,)."""
        vectors = _random_vectors(n=25, dim=6)
        clusterer = KMeansClusterer(n_clusters=4, seed=42)
        result = clusterer.fit(vectors)
        self.assertEqual(result.labels.shape, (25,))

    def test_labels_value_range(self) -> None:
        """``labels`` stay in ``[0, n_clusters - 1]``."""
        vectors = _random_vectors(n=30, dim=8)
        k = 5
        clusterer = KMeansClusterer(n_clusters=k, seed=42)
        result = clusterer.fit(vectors)
        self.assertGreaterEqual(int(result.labels.min()), 0)
        self.assertLess(int(result.labels.max()), k)

    def test_n_samples_field(self) -> None:
        """``n_samples`` matches the input count."""
        vectors = _random_vectors(n=15, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        result = clusterer.fit(vectors)
        self.assertEqual(result.n_samples, 15)

    def test_n_clusters_field(self) -> None:
        """``n_clusters`` matches the requested value."""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=4, seed=42)
        result = clusterer.fit(vectors)
        self.assertEqual(result.n_clusters, 4)

    def test_seed_field(self) -> None:
        """``seed`` matches the requested value."""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=7)
        result = clusterer.fit(vectors)
        self.assertEqual(result.seed, 7)

    def test_inertia_is_nonnegative(self) -> None:
        """``inertia`` is non-negative (within-cluster sum of squares ≥ 0)."""
        vectors = _random_vectors(n=30, dim=8)
        clusterer = KMeansClusterer(n_clusters=4, seed=42)
        result = clusterer.fit(vectors)
        self.assertGreaterEqual(result.inertia, 0.0)

    def test_separable_data_groups_correctly(self) -> None:
        """For clearly separable data, neighboring points share the same cluster label."""
        vectors = _separable_vectors(seed=0)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        result = clusterer.fit(vectors)
        # Each block of 10 points should share a label (at least the first two agree).
        for start in (0, 10, 20):
            self.assertEqual(int(result.labels[start]), int(result.labels[start + 1]))

    def test_invalid_input_not_ndarray(self) -> None:
        """Passing a non-ndarray raises ``UnfitVectorError``."""
        clusterer = KMeansClusterer(n_clusters=2, seed=42)
        with self.assertRaises(UnfitVectorError):
            clusterer.fit([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_invalid_input_1d(self) -> None:
        """Passing a 1-D array raises ``UnfitVectorError``."""
        clusterer = KMeansClusterer(n_clusters=2, seed=42)
        with self.assertRaises(UnfitVectorError):
            clusterer.fit(np.array([1.0, 2.0, 3.0]))

    def test_n_clusters_exceeds_n_samples(self) -> None:
        """``n_clusters > N`` raises ``InvalidClusterCountError``."""
        vectors = _random_vectors(n=3, dim=4)
        clusterer = KMeansClusterer(n_clusters=10, seed=42)
        with self.assertRaises(InvalidClusterCountError):
            clusterer.fit(vectors)


# ---------------------------------------------------------------------------
# get_labels() / get_result()
# ---------------------------------------------------------------------------

class TestGetLabels(unittest.TestCase):
    """Tests for ``KMeansClusterer.get_labels()``."""

    def test_returns_labels_after_fit(self) -> None:
        """After ``fit()``, ``get_labels()`` returns the labels."""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        clusterer.fit(vectors)
        labels = clusterer.get_labels()
        self.assertEqual(labels.shape, (20,))

    def test_raises_before_fit(self) -> None:
        """Calling before ``fit()`` raises ``NotFittedError``."""
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        with self.assertRaises(NotFittedError):
            clusterer.get_labels()


class TestGetResult(unittest.TestCase):
    """Tests for ``KMeansClusterer.get_result()``."""

    def test_returns_result_after_fit(self) -> None:
        """After ``fit()``, ``get_result()`` returns a ``ClusterResult``."""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        clusterer.fit(vectors)
        result = clusterer.get_result()
        self.assertIsInstance(result, ClusterResult)

    def test_raises_before_fit(self) -> None:
        """Calling before ``fit()`` raises ``NotFittedError``."""
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        with self.assertRaises(NotFittedError):
            clusterer.get_result()


# ---------------------------------------------------------------------------
# Hand-rolled norm / normalization (numerical properties)
# ---------------------------------------------------------------------------

class TestL2NormBatch(unittest.TestCase):
    """Tests for ``KMeansClusterer._l2_norm_batch()``."""

    def test_norms_known_values(self) -> None:
        """Norms of [[3, 4], [0, 0], [1, 0]] are [5, 0, 1]."""
        matrix = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        norms = KMeansClusterer._l2_norm_batch(matrix)
        np.testing.assert_allclose(norms, [5.0, 0.0, 1.0], atol=1e-5)

    def test_output_shape(self) -> None:
        """Output shape is (N,)."""
        matrix = _random_vectors(n=12, dim=8)
        norms = KMeansClusterer._l2_norm_batch(matrix)
        self.assertEqual(norms.shape, (12,))

    def test_norms_are_nonnegative(self) -> None:
        """Norms are non-negative."""
        matrix = _random_vectors(n=20, dim=10)
        norms = KMeansClusterer._l2_norm_batch(matrix)
        self.assertTrue((norms >= 0.0).all())


class TestNormalizeRows(unittest.TestCase):
    """Tests for ``KMeansClusterer._normalize_rows()``."""

    def test_unit_vectors_after_normalization(self) -> None:
        """Non-zero rows have norm 1.0 after normalization."""
        matrix = _random_vectors(n=10, dim=5)
        unit = KMeansClusterer._normalize_rows(matrix)
        norms = KMeansClusterer._l2_norm_batch(unit)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_zero_row_remains_zero(self) -> None:
        """Zero-vector rows remain zero after normalization (no NaN)."""
        matrix = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        unit = KMeansClusterer._normalize_rows(matrix)
        np.testing.assert_allclose(unit[1], [0.0, 0.0], atol=1e-7)
        self.assertFalse(np.any(np.isnan(unit)))

    def test_does_not_modify_input(self) -> None:
        """The input matrix is not mutated."""
        matrix = np.array([[3.0, 4.0]], dtype=np.float32)
        snapshot = matrix.copy()
        _ = KMeansClusterer._normalize_rows(matrix)
        np.testing.assert_array_equal(matrix, snapshot)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):
    """Reproducibility tests under a fixed seed."""

    def test_same_seed_same_labels(self) -> None:
        """Identical labels across runs under the same seed and data."""
        vectors = _random_vectors(n=30, dim=8, seed=0)
        labels_1 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).labels
        labels_2 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).labels
        np.testing.assert_array_equal(labels_1, labels_2)

    def test_same_seed_same_inertia(self) -> None:
        """Identical inertia across runs under the same seed."""
        vectors = _random_vectors(n=30, dim=8, seed=0)
        inertia_1 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).inertia
        inertia_2 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).inertia
        self.assertEqual(inertia_1, inertia_2)


if __name__ == "__main__":
    unittest.main()
