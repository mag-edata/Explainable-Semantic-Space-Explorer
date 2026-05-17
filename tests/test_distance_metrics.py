"""
test_distance_metrics.py
========================
Unit tests for ``DistanceMetrics``.

Targets under test:
    - ``l2_norm()``: L2 norm computation
    - ``cosine_similarity()``: Cosine similarity between two vectors
    - ``cosine_similarity_batch()``: Batched computation
    - ``explain()``: Generation of the breakdown dictionary

How to run:
    venv/bin/python3 -m unittest tests/test_distance_metrics.py -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.distance_metrics import (
    DistanceMetrics,
    VectorDimensionError,
)


class TestL2Norm(unittest.TestCase):
    """Tests for ``DistanceMetrics.l2_norm()``."""

    def test_known_vector(self) -> None:
        """The norm of [3, 4] is 5.0 (sanity check from the Pythagorean theorem)."""
        vec = np.array([3.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 5.0, places=5)

    def test_unit_vector(self) -> None:
        """The norm of the unit vector [1, 0, 0] is 1.0."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 1.0, places=5)

    def test_zero_vector(self) -> None:
        """The norm of the zero vector is 0.0."""
        vec = np.zeros(4, dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 0.0, places=5)

    def test_all_ones(self) -> None:
        """The norm of [1, 1, 1, 1] is sqrt(4) = 2.0."""
        vec = np.ones(4, dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 2.0, places=5)

    def test_returns_python_float(self) -> None:
        """The return value is a Python ``float``."""
        vec = np.array([1.0, 2.0], dtype=np.float32)
        result = DistanceMetrics.l2_norm(vec)
        self.assertIsInstance(result, float)

    def test_type_error_on_list(self) -> None:
        """Passing a list raises ``TypeError``."""
        with self.assertRaises(TypeError):
            DistanceMetrics.l2_norm([1.0, 2.0])

    def test_dimension_error_on_2d(self) -> None:
        """Passing a 2-D array raises ``VectorDimensionError``."""
        vec = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with self.assertRaises(VectorDimensionError):
            DistanceMetrics.l2_norm(vec)

    def test_high_dimensional(self) -> None:
        """The norm of a high-dimensional vector is computed accurately (300-D)."""
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(300).astype(np.float32)
        expected = float(math.sqrt(float(np.sum(vec ** 2))))
        result = DistanceMetrics.l2_norm(vec)
        self.assertAlmostEqual(result, expected, places=4)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for ``DistanceMetrics.cosine_similarity()``."""

    def test_parallel_vectors(self) -> None:
        """Cosine similarity of parallel vectors is 1.0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 6.0], dtype=np.float32)  # 2x a
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), 1.0, places=5)

    def test_antiparallel_vectors(self) -> None:
        """Cosine similarity of antiparallel vectors is -1.0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), -1.0, places=5)

    def test_orthogonal_vectors(self) -> None:
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), 0.0, places=5)

    def test_zero_vector_guard(self) -> None:
        """Similarity against the zero vector is 0.0 (zero-division guard)."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), 0.0, places=5)

    def test_range_is_minus1_to_1(self) -> None:
        """Results fall within [-1.0, 1.0] (randomized test)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(50).astype(np.float32)
            b = rng.standard_normal(50).astype(np.float32)
            sim = DistanceMetrics.cosine_similarity(a, b)
            self.assertGreaterEqual(sim, -1.0 - 1e-6)
            self.assertLessEqual(sim, 1.0 + 1e-6)

    def test_symmetry(self) -> None:
        """Verifies symmetry: ``cos(a, b) == cos(b, a)``."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(10).astype(np.float32)
        b = rng.standard_normal(10).astype(np.float32)
        self.assertAlmostEqual(
            DistanceMetrics.cosine_similarity(a, b),
            DistanceMetrics.cosine_similarity(b, a),
            places=5,
        )

    def test_dimension_mismatch(self) -> None:
        """Mismatched dimensions raise ``VectorDimensionError``."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with self.assertRaises(VectorDimensionError):
            DistanceMetrics.cosine_similarity(a, b)


class TestCosineSimilarityBatch(unittest.TestCase):
    """Tests for ``DistanceMetrics.cosine_similarity_batch()``."""

    def setUp(self) -> None:
        """Prepare test vectors."""
        self.query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.matrix = np.array([
            [1.0, 0.0, 0.0],   # identical to query → 1.0
            [0.0, 1.0, 0.0],   # orthogonal          → 0.0
            [-1.0, 0.0, 0.0],  # antiparallel        → -1.0
            [1.0, 1.0, 0.0],   # 45 degrees          → 1/sqrt(2)
        ], dtype=np.float32)

    def test_output_shape(self) -> None:
        """Output shape is (N,)."""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertEqual(result.shape, (4,))

    def test_identical_vector(self) -> None:
        """Similarity to an identical vector is 1.0."""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[0]), 1.0, places=5)

    def test_orthogonal_vector(self) -> None:
        """Similarity to an orthogonal vector is 0.0."""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_antiparallel_vector(self) -> None:
        """Similarity to an antiparallel vector is -1.0."""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[2]), -1.0, places=5)

    def test_45_degree_vector(self) -> None:
        """Similarity for a 45-degree vector is 1/sqrt(2) ≈ 0.7071."""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[3]), 1.0 / math.sqrt(2), places=5)

    def test_zero_row_guard(self) -> None:
        """No NaN appears when the matrix contains zero rows (zero-division guard)."""
        matrix_with_zero = np.array([
            [1.0, 0.0],
            [0.0, 0.0],  # zero row
        ], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        result = DistanceMetrics.cosine_similarity_batch(query, matrix_with_zero)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_consistency_with_single(self) -> None:
        """Batch result matches the single-pair result row by row."""
        rng = np.random.default_rng(1)
        q = rng.standard_normal(20).astype(np.float32)
        M = rng.standard_normal((10, 20)).astype(np.float32)

        batch_result = DistanceMetrics.cosine_similarity_batch(q, M)
        for i in range(10):
            single = DistanceMetrics.cosine_similarity(q, M[i])
            self.assertAlmostEqual(float(batch_result[i]), single, places=4)

    def test_dimension_mismatch(self) -> None:
        """Mismatched query / matrix dimensions raise ``VectorDimensionError``."""
        q = np.array([1.0, 2.0], dtype=np.float32)
        M = np.ones((5, 3), dtype=np.float32)
        with self.assertRaises(VectorDimensionError):
            DistanceMetrics.cosine_similarity_batch(q, M)


class TestExplain(unittest.TestCase):
    """Tests for ``DistanceMetrics.explain()``."""

    def setUp(self) -> None:
        """Prepare test vectors."""
        self.a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def test_required_keys(self) -> None:
        """The returned dictionary contains every required key."""
        result = DistanceMetrics.explain(self.a, self.b)
        for key in ("dot_product", "norm_a", "norm_b", "denominator", "similarity", "formula"):
            self.assertIn(key, result)

    def test_formula_is_string(self) -> None:
        """The ``formula`` field is a string."""
        result = DistanceMetrics.explain(self.a, self.b)
        self.assertIsInstance(result["formula"], str)

    def test_orthogonal_similarity_is_zero(self) -> None:
        """Similarity is 0.0 for orthogonal vectors."""
        result = DistanceMetrics.explain(self.a, self.b)
        self.assertAlmostEqual(result["similarity"], 0.0, places=5)

    def test_parallel_similarity_is_one(self) -> None:
        """Similarity is 1.0 for parallel vectors."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 6.0], dtype=np.float32)
        result = DistanceMetrics.explain(a, b)
        self.assertAlmostEqual(result["similarity"], 1.0, places=5)

    def test_dot_product_value(self) -> None:
        """``dot_product`` is computed correctly."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = DistanceMetrics.explain(a, b)
        expected_dot = float(np.dot(a, b))
        self.assertAlmostEqual(result["dot_product"], expected_dot, places=4)

    def test_norms_are_positive(self) -> None:
        """``norm_a`` and ``norm_b`` are non-negative."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(10).astype(np.float32)
        b = rng.standard_normal(10).astype(np.float32)
        result = DistanceMetrics.explain(a, b)
        self.assertGreaterEqual(result["norm_a"], 0.0)
        self.assertGreaterEqual(result["norm_b"], 0.0)


if __name__ == "__main__":
    unittest.main()
