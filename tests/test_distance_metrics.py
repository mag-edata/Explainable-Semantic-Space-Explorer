"""
test_distance_metrics.py
========================
DistanceMetrics の単体テスト。

テスト対象:
    - l2_norm(): L2 ノルムの計算
    - cosine_similarity(): 2ベクトルのコサイン類似度
    - cosine_similarity_batch(): バッチ計算
    - explain(): 計算過程辞書の生成

実行方法:
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
    """DistanceMetrics.l2_norm() のテスト。"""

    def test_known_vector(self) -> None:
        """[3, 4] のノルムは 5.0。ピタゴラスの定理の確認。"""
        vec = np.array([3.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 5.0, places=5)

    def test_unit_vector(self) -> None:
        """単位ベクトル [1, 0, 0] のノルムは 1.0。"""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 1.0, places=5)

    def test_zero_vector(self) -> None:
        """ゼロベクトルのノルムは 0.0。"""
        vec = np.zeros(4, dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 0.0, places=5)

    def test_all_ones(self) -> None:
        """[1, 1, 1, 1] のノルムは sqrt(4) = 2.0。"""
        vec = np.ones(4, dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.l2_norm(vec), 2.0, places=5)

    def test_returns_python_float(self) -> None:
        """戻り値が Python の float 型であることを確認。"""
        vec = np.array([1.0, 2.0], dtype=np.float32)
        result = DistanceMetrics.l2_norm(vec)
        self.assertIsInstance(result, float)

    def test_type_error_on_list(self) -> None:
        """list を渡した場合に TypeError が発生する。"""
        with self.assertRaises(TypeError):
            DistanceMetrics.l2_norm([1.0, 2.0])

    def test_dimension_error_on_2d(self) -> None:
        """2次元配列を渡した場合に VectorDimensionError が発生する。"""
        vec = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with self.assertRaises(VectorDimensionError):
            DistanceMetrics.l2_norm(vec)

    def test_high_dimensional(self) -> None:
        """高次元ベクトルのノルムが正確に計算される（300次元）。"""
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(300).astype(np.float32)
        expected = float(math.sqrt(float(np.sum(vec ** 2))))
        result = DistanceMetrics.l2_norm(vec)
        self.assertAlmostEqual(result, expected, places=4)


class TestCosineSimilarity(unittest.TestCase):
    """DistanceMetrics.cosine_similarity() のテスト。"""

    def test_parallel_vectors(self) -> None:
        """同一方向ベクトルのコサイン類似度は 1.0。"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 6.0], dtype=np.float32)  # a の 2 倍
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), 1.0, places=5)

    def test_antiparallel_vectors(self) -> None:
        """逆方向ベクトルのコサイン類似度は -1.0。"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), -1.0, places=5)

    def test_orthogonal_vectors(self) -> None:
        """直交ベクトルのコサイン類似度は 0.0。"""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), 0.0, places=5)

    def test_zero_vector_guard(self) -> None:
        """ゼロベクトルとの類似度は 0.0（ゼロ除算ガード）。"""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(a, b), 0.0, places=5)

    def test_range_is_minus1_to_1(self) -> None:
        """結果が [-1.0, 1.0] の範囲内に収まることを確認（乱数テスト）。"""
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(50).astype(np.float32)
            b = rng.standard_normal(50).astype(np.float32)
            sim = DistanceMetrics.cosine_similarity(a, b)
            self.assertGreaterEqual(sim, -1.0 - 1e-6)
            self.assertLessEqual(sim, 1.0 + 1e-6)

    def test_symmetry(self) -> None:
        """cos(a, b) == cos(b, a) の対称性を確認。"""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(10).astype(np.float32)
        b = rng.standard_normal(10).astype(np.float32)
        self.assertAlmostEqual(
            DistanceMetrics.cosine_similarity(a, b),
            DistanceMetrics.cosine_similarity(b, a),
            places=5,
        )

    def test_dimension_mismatch(self) -> None:
        """次元数が異なるベクトルを渡した場合に VectorDimensionError が発生する。"""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with self.assertRaises(VectorDimensionError):
            DistanceMetrics.cosine_similarity(a, b)


class TestCosineSimilarityBatch(unittest.TestCase):
    """DistanceMetrics.cosine_similarity_batch() のテスト。"""

    def setUp(self) -> None:
        """テスト用ベクトルの準備。"""
        self.query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.matrix = np.array([
            [1.0, 0.0, 0.0],   # クエリと同一 → 1.0
            [0.0, 1.0, 0.0],   # 直交          → 0.0
            [-1.0, 0.0, 0.0],  # 逆方向        → -1.0
            [1.0, 1.0, 0.0],   # 45度          → 1/sqrt(2)
        ], dtype=np.float32)

    def test_output_shape(self) -> None:
        """出力の shape が (N,) であることを確認。"""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertEqual(result.shape, (4,))

    def test_identical_vector(self) -> None:
        """クエリと同一ベクトルの類似度は 1.0。"""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[0]), 1.0, places=5)

    def test_orthogonal_vector(self) -> None:
        """直交ベクトルの類似度は 0.0。"""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_antiparallel_vector(self) -> None:
        """逆方向ベクトルの類似度は -1.0。"""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[2]), -1.0, places=5)

    def test_45_degree_vector(self) -> None:
        """45度のベクトルの類似度は 1/sqrt(2) ≈ 0.7071。"""
        result = DistanceMetrics.cosine_similarity_batch(self.query, self.matrix)
        self.assertAlmostEqual(float(result[3]), 1.0 / math.sqrt(2), places=5)

    def test_zero_row_guard(self) -> None:
        """ゼロ行を含む行列でも NaN が発生しない（ゼロ除算ガード）。"""
        matrix_with_zero = np.array([
            [1.0, 0.0],
            [0.0, 0.0],  # ゼロベクトル行
        ], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        result = DistanceMetrics.cosine_similarity_batch(query, matrix_with_zero)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_consistency_with_single(self) -> None:
        """バッチ計算の結果が単一計算の結果と一致することを確認。"""
        rng = np.random.default_rng(1)
        q = rng.standard_normal(20).astype(np.float32)
        M = rng.standard_normal((10, 20)).astype(np.float32)

        batch_result = DistanceMetrics.cosine_similarity_batch(q, M)
        for i in range(10):
            single = DistanceMetrics.cosine_similarity(q, M[i])
            self.assertAlmostEqual(float(batch_result[i]), single, places=4)

    def test_dimension_mismatch(self) -> None:
        """クエリと行列の次元数が一致しない場合に VectorDimensionError が発生する。"""
        q = np.array([1.0, 2.0], dtype=np.float32)
        M = np.ones((5, 3), dtype=np.float32)
        with self.assertRaises(VectorDimensionError):
            DistanceMetrics.cosine_similarity_batch(q, M)


class TestExplain(unittest.TestCase):
    """DistanceMetrics.explain() のテスト。"""

    def setUp(self) -> None:
        """テスト用ベクトルの準備。"""
        self.a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def test_required_keys(self) -> None:
        """返り値辞書に必須キーが含まれることを確認。"""
        result = DistanceMetrics.explain(self.a, self.b)
        for key in ("dot_product", "norm_a", "norm_b", "denominator", "similarity", "formula"):
            self.assertIn(key, result)

    def test_formula_is_string(self) -> None:
        """formula フィールドが文字列であることを確認。"""
        result = DistanceMetrics.explain(self.a, self.b)
        self.assertIsInstance(result["formula"], str)

    def test_orthogonal_similarity_is_zero(self) -> None:
        """直交ベクトルの similarity は 0.0。"""
        result = DistanceMetrics.explain(self.a, self.b)
        self.assertAlmostEqual(result["similarity"], 0.0, places=5)

    def test_parallel_similarity_is_one(self) -> None:
        """平行ベクトルの similarity は 1.0。"""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 6.0], dtype=np.float32)
        result = DistanceMetrics.explain(a, b)
        self.assertAlmostEqual(result["similarity"], 1.0, places=5)

    def test_dot_product_value(self) -> None:
        """dot_product フィールドの値が正確か確認。"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = DistanceMetrics.explain(a, b)
        expected_dot = float(np.dot(a, b))
        self.assertAlmostEqual(result["dot_product"], expected_dot, places=4)

    def test_norms_are_positive(self) -> None:
        """norm_a / norm_b フィールドが非負であることを確認。"""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(10).astype(np.float32)
        b = rng.standard_normal(10).astype(np.float32)
        result = DistanceMetrics.explain(a, b)
        self.assertGreaterEqual(result["norm_a"], 0.0)
        self.assertGreaterEqual(result["norm_b"], 0.0)


if __name__ == "__main__":
    unittest.main()
