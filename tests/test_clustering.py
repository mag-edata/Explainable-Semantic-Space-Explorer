"""
test_clustering.py
==================
KMeansClusterer の単体テスト。

テスト対象:
    - __init__():    引数バリデーション・例外
    - fit():         クラスタリング実行・結果オブジェクト
    - get_labels():  ラベル取得・未 fit エラー
    - get_result():  結果取得・未 fit エラー
    - 自前ノルム/正規化（_l2_norm_batch / _normalize_rows）の数値性質
    - 再現性（同一 seed → 同一結果）

実行方法:
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
# テスト用フィクスチャ
# ---------------------------------------------------------------------------

def _random_vectors(n: int = 30, dim: int = 8, seed: int = 0) -> np.ndarray:
    """テスト用のランダム埋め込み行列を生成。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _separable_vectors(seed: int = 0) -> np.ndarray:
    """3 つの明確に分離されたクラスタを持つ行列を生成（k=3 の検証用）。"""
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
    """KMeansClusterer.__init__() のテスト。"""

    def test_default_construction(self) -> None:
        """デフォルト引数でインスタンス化できることを確認。"""
        clusterer = KMeansClusterer()
        self.assertIsInstance(clusterer, KMeansClusterer)

    def test_custom_construction(self) -> None:
        """カスタム引数でインスタンス化できることを確認。"""
        KMeansClusterer(n_clusters=5, seed=7, max_iter=100)

    def test_invalid_n_clusters_zero_raises(self) -> None:
        """n_clusters=0 で InvalidClusterCountError が送出される。"""
        with self.assertRaises(InvalidClusterCountError):
            KMeansClusterer(n_clusters=0)

    def test_invalid_n_clusters_negative_raises(self) -> None:
        """n_clusters が負の値で InvalidClusterCountError が送出される。"""
        with self.assertRaises(InvalidClusterCountError):
            KMeansClusterer(n_clusters=-1)

    def test_n_clusters_type_error(self) -> None:
        """n_clusters が int でない場合 TypeError が送出される。"""
        with self.assertRaises(TypeError):
            KMeansClusterer(n_clusters="3")  # type: ignore[arg-type]

    def test_seed_type_error(self) -> None:
        """seed が int でない場合 TypeError が送出される。"""
        with self.assertRaises(TypeError):
            KMeansClusterer(seed=3.14)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit(unittest.TestCase):
    """KMeansClusterer.fit() のテスト。"""

    def test_returns_cluster_result(self) -> None:
        """戻り値が ClusterResult であることを確認。"""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        result = clusterer.fit(vectors)
        self.assertIsInstance(result, ClusterResult)

    def test_labels_shape(self) -> None:
        """labels の shape が (N,) であることを確認。"""
        vectors = _random_vectors(n=25, dim=6)
        clusterer = KMeansClusterer(n_clusters=4, seed=42)
        result = clusterer.fit(vectors)
        self.assertEqual(result.labels.shape, (25,))

    def test_labels_value_range(self) -> None:
        """labels が [0, n_clusters - 1] の範囲に収まることを確認。"""
        vectors = _random_vectors(n=30, dim=8)
        k = 5
        clusterer = KMeansClusterer(n_clusters=k, seed=42)
        result = clusterer.fit(vectors)
        self.assertGreaterEqual(int(result.labels.min()), 0)
        self.assertLess(int(result.labels.max()), k)

    def test_n_samples_field(self) -> None:
        """n_samples フィールドが入力件数と一致することを確認。"""
        vectors = _random_vectors(n=15, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        result = clusterer.fit(vectors)
        self.assertEqual(result.n_samples, 15)

    def test_n_clusters_field(self) -> None:
        """n_clusters フィールドが指定値と一致することを確認。"""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=4, seed=42)
        result = clusterer.fit(vectors)
        self.assertEqual(result.n_clusters, 4)

    def test_seed_field(self) -> None:
        """seed フィールドが指定値と一致することを確認。"""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=7)
        result = clusterer.fit(vectors)
        self.assertEqual(result.seed, 7)

    def test_inertia_is_nonnegative(self) -> None:
        """inertia が非負であることを確認（クラスタ内二乗和 ≥ 0）。"""
        vectors = _random_vectors(n=30, dim=8)
        clusterer = KMeansClusterer(n_clusters=4, seed=42)
        result = clusterer.fit(vectors)
        self.assertGreaterEqual(result.inertia, 0.0)

    def test_separable_data_groups_correctly(self) -> None:
        """明確に分離されたデータでは、近接する点が同一クラスタに属することを確認。"""
        vectors = _separable_vectors(seed=0)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        result = clusterer.fit(vectors)
        # 各 10 点ブロックは同じラベルになるはず（少なくとも先頭2点は一致）
        for start in (0, 10, 20):
            self.assertEqual(int(result.labels[start]), int(result.labels[start + 1]))

    def test_invalid_input_not_ndarray(self) -> None:
        """np.ndarray 以外を渡すと UnfitVectorError が送出される。"""
        clusterer = KMeansClusterer(n_clusters=2, seed=42)
        with self.assertRaises(UnfitVectorError):
            clusterer.fit([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_invalid_input_1d(self) -> None:
        """1次元配列を渡すと UnfitVectorError が送出される。"""
        clusterer = KMeansClusterer(n_clusters=2, seed=42)
        with self.assertRaises(UnfitVectorError):
            clusterer.fit(np.array([1.0, 2.0, 3.0]))

    def test_n_clusters_exceeds_n_samples(self) -> None:
        """n_clusters > N の場合に InvalidClusterCountError が送出される。"""
        vectors = _random_vectors(n=3, dim=4)
        clusterer = KMeansClusterer(n_clusters=10, seed=42)
        with self.assertRaises(InvalidClusterCountError):
            clusterer.fit(vectors)


# ---------------------------------------------------------------------------
# get_labels() / get_result()
# ---------------------------------------------------------------------------

class TestGetLabels(unittest.TestCase):
    """KMeansClusterer.get_labels() のテスト。"""

    def test_returns_labels_after_fit(self) -> None:
        """fit() 後に get_labels() でラベルが取得できることを確認。"""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        clusterer.fit(vectors)
        labels = clusterer.get_labels()
        self.assertEqual(labels.shape, (20,))

    def test_raises_before_fit(self) -> None:
        """fit() 前に呼ぶと NotFittedError が送出される。"""
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        with self.assertRaises(NotFittedError):
            clusterer.get_labels()


class TestGetResult(unittest.TestCase):
    """KMeansClusterer.get_result() のテスト。"""

    def test_returns_result_after_fit(self) -> None:
        """fit() 後に get_result() で ClusterResult が取得できることを確認。"""
        vectors = _random_vectors(n=20, dim=4)
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        clusterer.fit(vectors)
        result = clusterer.get_result()
        self.assertIsInstance(result, ClusterResult)

    def test_raises_before_fit(self) -> None:
        """fit() 前に呼ぶと NotFittedError が送出される。"""
        clusterer = KMeansClusterer(n_clusters=3, seed=42)
        with self.assertRaises(NotFittedError):
            clusterer.get_result()


# ---------------------------------------------------------------------------
# 自前ノルム・正規化（数値性質）
# ---------------------------------------------------------------------------

class TestL2NormBatch(unittest.TestCase):
    """KMeansClusterer._l2_norm_batch() のテスト。"""

    def test_norms_known_values(self) -> None:
        """[[3, 4], [0, 0], [1, 0]] のノルムは [5, 0, 1]。"""
        matrix = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        norms = KMeansClusterer._l2_norm_batch(matrix)
        np.testing.assert_allclose(norms, [5.0, 0.0, 1.0], atol=1e-5)

    def test_output_shape(self) -> None:
        """出力 shape が (N,) であることを確認。"""
        matrix = _random_vectors(n=12, dim=8)
        norms = KMeansClusterer._l2_norm_batch(matrix)
        self.assertEqual(norms.shape, (12,))

    def test_norms_are_nonnegative(self) -> None:
        """ノルムが非負であることを確認。"""
        matrix = _random_vectors(n=20, dim=10)
        norms = KMeansClusterer._l2_norm_batch(matrix)
        self.assertTrue((norms >= 0.0).all())


class TestNormalizeRows(unittest.TestCase):
    """KMeansClusterer._normalize_rows() のテスト。"""

    def test_unit_vectors_after_normalization(self) -> None:
        """正規化後、ゼロでない行のノルムが 1.0 になることを確認。"""
        matrix = _random_vectors(n=10, dim=5)
        unit = KMeansClusterer._normalize_rows(matrix)
        norms = KMeansClusterer._l2_norm_batch(unit)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_zero_row_remains_zero(self) -> None:
        """ゼロベクトル行は正規化後もゼロのまま（NaN にならない）。"""
        matrix = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        unit = KMeansClusterer._normalize_rows(matrix)
        np.testing.assert_allclose(unit[1], [0.0, 0.0], atol=1e-7)
        self.assertFalse(np.any(np.isnan(unit)))

    def test_does_not_modify_input(self) -> None:
        """入力 matrix が破壊されないことを確認。"""
        matrix = np.array([[3.0, 4.0]], dtype=np.float32)
        snapshot = matrix.copy()
        _ = KMeansClusterer._normalize_rows(matrix)
        np.testing.assert_array_equal(matrix, snapshot)


# ---------------------------------------------------------------------------
# 再現性
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):
    """seed 固定下での完全再現性のテスト。"""

    def test_same_seed_same_labels(self) -> None:
        """同一 seed・同一データで labels が完全一致することを確認。"""
        vectors = _random_vectors(n=30, dim=8, seed=0)
        labels_1 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).labels
        labels_2 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).labels
        np.testing.assert_array_equal(labels_1, labels_2)

    def test_same_seed_same_inertia(self) -> None:
        """同一 seed で inertia も完全一致することを確認。"""
        vectors = _random_vectors(n=30, dim=8, seed=0)
        inertia_1 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).inertia
        inertia_2 = KMeansClusterer(n_clusters=4, seed=42).fit(vectors).inertia
        self.assertEqual(inertia_1, inertia_2)


if __name__ == "__main__":
    unittest.main()
