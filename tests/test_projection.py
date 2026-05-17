"""
test_projection.py
==================
Unit tests for ``Projector``.

Targets under test:
    - ``__init__()``:        validation of ``method`` / ``seed``
    - ``fit_transform()``:   2D projection with PCA / UMAP
    - ``attach_clusters()``: attach cluster labels (immutable operation)
    - Reproducibility (same seed → identical coordinates)

How to run:
    venv/bin/python3 -m unittest tests/test_projection.py -v

Note:
    UMAP is expensive, so tests use a small matrix (n=20, dim=8).
"""

from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transforms.projection import (
    InvalidMethodError,
    InvalidVectorError,
    Projector,
    ProjectionResult,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _random_vectors(n: int = 20, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Generate a random embedding matrix for tests."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# __init__()
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):
    """Tests for ``Projector.__init__()``."""

    def test_default_method_is_pca(self) -> None:
        """The default method is ``"pca"``."""
        proj = Projector()
        self.assertEqual(proj._method, "pca")

    def test_pca_construction(self) -> None:
        """Construction with ``method="pca"`` succeeds."""
        proj = Projector(method="pca", seed=42)
        self.assertIsInstance(proj, Projector)

    def test_umap_construction(self) -> None:
        """Construction with ``method="umap"`` succeeds."""
        proj = Projector(method="umap", seed=42)
        self.assertIsInstance(proj, Projector)

    def test_invalid_method_raises(self) -> None:
        """An unsupported method raises ``InvalidMethodError``."""
        with self.assertRaises(InvalidMethodError):
            Projector(method="tsne")

    def test_method_type_error(self) -> None:
        """A non-str ``method`` raises ``TypeError``."""
        with self.assertRaises(TypeError):
            Projector(method=123)  # type: ignore[arg-type]

    def test_seed_type_error(self) -> None:
        """A non-int ``seed`` raises ``TypeError``."""
        with self.assertRaises(TypeError):
            Projector(method="pca", seed=3.14)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fit_transform() — PCA
# ---------------------------------------------------------------------------

class TestFitTransformPCA(unittest.TestCase):
    """Tests for the PCA branch of ``Projector.fit_transform()``."""

    def setUp(self) -> None:
        self.vectors = _random_vectors(n=20, dim=8)

    def test_returns_projection_result(self) -> None:
        """The return value is a ``ProjectionResult``."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertIsInstance(result, ProjectionResult)

    def test_coords_shape(self) -> None:
        """``coords_2d.shape`` is (N, 2)."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.coords_2d.shape, (20, 2))

    def test_method_field(self) -> None:
        """The ``method`` field is ``"pca"``."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.method, "pca")

    def test_explained_variance_length(self) -> None:
        """``explained_variance`` contains 2 entries."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(len(result.explained_variance), 2)

    def test_explained_variance_in_range(self) -> None:
        """Each contribution rate is in [0.0, 1.0]."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        for v in result.explained_variance:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_explained_variance_sum_le_one(self) -> None:
        """The sum of contribution rates is at most 1.0 (top 2 components only)."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertLessEqual(sum(result.explained_variance), 1.0 + 1e-6)

    def test_explained_variance_descending(self) -> None:
        """PC1's contribution rate is at least PC2's."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertGreaterEqual(
            result.explained_variance[0],
            result.explained_variance[1],
        )

    def test_n_samples_field(self) -> None:
        """``n_samples`` matches the input count."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.n_samples, 20)

    def test_seed_field(self) -> None:
        """``seed`` matches the requested value."""
        proj = Projector(method="pca", seed=7)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.seed, 7)

    def test_cluster_labels_initially_none(self) -> None:
        """``cluster_labels`` is ``None`` immediately after ``fit_transform``."""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertIsNone(result.cluster_labels)


# ---------------------------------------------------------------------------
# fit_transform() — UMAP
# ---------------------------------------------------------------------------

class TestFitTransformUMAP(unittest.TestCase):
    """Tests for the UMAP branch of ``Projector.fit_transform()``."""

    def setUp(self) -> None:
        # UMAP's default n_neighbors is 15, so N >= 16 keeps things stable.
        self.vectors = _random_vectors(n=20, dim=8)

    def test_coords_shape(self) -> None:
        """``coords_2d.shape`` is (N, 2)."""
        proj = Projector(method="umap", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.coords_2d.shape, (20, 2))

    def test_method_field(self) -> None:
        """The ``method`` field is ``"umap"``."""
        proj = Projector(method="umap", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.method, "umap")

    def test_explained_variance_is_empty(self) -> None:
        """For UMAP, ``explained_variance`` is an empty list."""
        proj = Projector(method="umap", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.explained_variance, [])


# ---------------------------------------------------------------------------
# fit_transform() — input validation
# ---------------------------------------------------------------------------

class TestFitTransformValidation(unittest.TestCase):
    """Input validation for ``fit_transform()``."""

    def test_invalid_input_not_ndarray(self) -> None:
        """Non-ndarray input raises ``InvalidVectorError``."""
        proj = Projector(method="pca", seed=42)
        with self.assertRaises(InvalidVectorError):
            proj.fit_transform([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    def test_invalid_input_1d(self) -> None:
        """A 1-D array raises ``InvalidVectorError``."""
        proj = Projector(method="pca", seed=42)
        with self.assertRaises(InvalidVectorError):
            proj.fit_transform(np.array([1.0, 2.0, 3.0]))

    def test_too_few_samples(self) -> None:
        """Fewer than 2 samples raises ``InvalidVectorError``."""
        proj = Projector(method="pca", seed=42)
        with self.assertRaises(InvalidVectorError):
            proj.fit_transform(np.array([[1.0, 2.0, 3.0]]))


# ---------------------------------------------------------------------------
# attach_clusters()
# ---------------------------------------------------------------------------

class TestAttachClusters(unittest.TestCase):
    """Tests for ``Projector.attach_clusters()``."""

    def setUp(self) -> None:
        self.vectors = _random_vectors(n=10, dim=4)
        self.proj = Projector(method="pca", seed=42)
        self.result = self.proj.fit_transform(self.vectors)
        self.labels = np.array([0, 1, 0, 2, 1, 0, 2, 1, 0, 2])

    def test_returns_projection_result(self) -> None:
        """The return value is a ``ProjectionResult``."""
        attached = self.proj.attach_clusters(self.result, self.labels)
        self.assertIsInstance(attached, ProjectionResult)

    def test_cluster_labels_assigned(self) -> None:
        """``cluster_labels`` matches the requested label array."""
        attached = self.proj.attach_clusters(self.result, self.labels)
        np.testing.assert_array_equal(attached.cluster_labels, self.labels)

    def test_immutable_does_not_modify_input(self) -> None:
        """The original result's ``cluster_labels`` is not mutated (immutable operation)."""
        _ = self.proj.attach_clusters(self.result, self.labels)
        self.assertIsNone(self.result.cluster_labels)

    def test_returns_new_object(self) -> None:
        """The return value is a different object from the original result."""
        attached = self.proj.attach_clusters(self.result, self.labels)
        self.assertIsNot(attached, self.result)

    def test_coords_preserved(self) -> None:
        """``coords_2d`` / ``explained_variance`` / ``method`` are carried over."""
        attached = self.proj.attach_clusters(self.result, self.labels)
        np.testing.assert_array_equal(attached.coords_2d, self.result.coords_2d)
        self.assertEqual(
            attached.explained_variance, self.result.explained_variance,
        )
        self.assertEqual(attached.method, self.result.method)

    def test_labels_length_mismatch_raises(self) -> None:
        """A label array of the wrong length raises ``InvalidVectorError``."""
        wrong = np.array([0, 1, 0])  # length 3, but result.n_samples=10
        with self.assertRaises(InvalidVectorError):
            self.proj.attach_clusters(self.result, wrong)

    def test_invalid_result_type(self) -> None:
        """A non-``ProjectionResult`` ``result`` argument raises ``TypeError``."""
        with self.assertRaises(TypeError):
            self.proj.attach_clusters("not_a_result", self.labels)  # type: ignore[arg-type]

    def test_invalid_labels_type(self) -> None:
        """Non-ndarray ``cluster_labels`` raises ``TypeError``."""
        with self.assertRaises(TypeError):
            self.proj.attach_clusters(self.result, [0, 1, 0, 2, 1, 0, 2, 1, 0, 2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):
    """Reproducibility tests under a fixed seed."""

    def test_pca_same_seed_same_coords(self) -> None:
        """PCA returns identical ``coords_2d`` for the same seed and data."""
        vectors = _random_vectors(n=20, dim=8, seed=0)
        coords_1 = Projector(method="pca", seed=42).fit_transform(vectors).coords_2d
        coords_2 = Projector(method="pca", seed=42).fit_transform(vectors).coords_2d
        np.testing.assert_array_equal(coords_1, coords_2)

    def test_umap_same_seed_same_coords(self) -> None:
        """UMAP returns identical ``coords_2d`` for the same seed and data."""
        vectors = _random_vectors(n=20, dim=8, seed=0)
        coords_1 = Projector(method="umap", seed=42).fit_transform(vectors).coords_2d
        coords_2 = Projector(method="umap", seed=42).fit_transform(vectors).coords_2d
        np.testing.assert_array_equal(coords_1, coords_2)


if __name__ == "__main__":
    unittest.main()
