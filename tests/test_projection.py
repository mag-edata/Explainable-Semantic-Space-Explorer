"""
test_projection.py
==================
Projector の単体テスト。

テスト対象:
    - __init__():       method / seed のバリデーション
    - fit_transform():  PCA / UMAP の 2D 投影
    - attach_clusters(): クラスタラベル付与（イミュータブル）
    - 再現性（同一 seed → 同一座標）

実行方法:
    venv/bin/python3 -m unittest tests/test_projection.py -v

備考:
    UMAP は計算コストがあるため、テストでは小規模行列（n=20, dim=8）に限定する。
"""

from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.projection import (
    InvalidMethodError,
    InvalidVectorError,
    Projector,
    ProjectionResult,
)


# ---------------------------------------------------------------------------
# テスト用フィクスチャ
# ---------------------------------------------------------------------------

def _random_vectors(n: int = 20, dim: int = 8, seed: int = 0) -> np.ndarray:
    """テスト用のランダム埋め込み行列を生成。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# __init__()
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):
    """Projector.__init__() のテスト。"""

    def test_default_method_is_pca(self) -> None:
        """デフォルトの手法が pca であることを確認。"""
        proj = Projector()
        self.assertEqual(proj._method, "pca")

    def test_pca_construction(self) -> None:
        """method='pca' でインスタンス化できることを確認。"""
        proj = Projector(method="pca", seed=42)
        self.assertIsInstance(proj, Projector)

    def test_umap_construction(self) -> None:
        """method='umap' でインスタンス化できることを確認。"""
        proj = Projector(method="umap", seed=42)
        self.assertIsInstance(proj, Projector)

    def test_invalid_method_raises(self) -> None:
        """サポート外の手法名を渡すと InvalidMethodError が送出される。"""
        with self.assertRaises(InvalidMethodError):
            Projector(method="tsne")

    def test_method_type_error(self) -> None:
        """method が str でない場合 TypeError が送出される。"""
        with self.assertRaises(TypeError):
            Projector(method=123)  # type: ignore[arg-type]

    def test_seed_type_error(self) -> None:
        """seed が int でない場合 TypeError が送出される。"""
        with self.assertRaises(TypeError):
            Projector(method="pca", seed=3.14)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fit_transform() — PCA
# ---------------------------------------------------------------------------

class TestFitTransformPCA(unittest.TestCase):
    """Projector.fit_transform() の PCA 経路テスト。"""

    def setUp(self) -> None:
        self.vectors = _random_vectors(n=20, dim=8)

    def test_returns_projection_result(self) -> None:
        """戻り値が ProjectionResult であることを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertIsInstance(result, ProjectionResult)

    def test_coords_shape(self) -> None:
        """coords_2d の shape が (N, 2) であることを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.coords_2d.shape, (20, 2))

    def test_method_field(self) -> None:
        """method フィールドが 'pca' であることを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.method, "pca")

    def test_explained_variance_length(self) -> None:
        """explained_variance の要素数が 2 であることを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(len(result.explained_variance), 2)

    def test_explained_variance_in_range(self) -> None:
        """各寄与率が [0.0, 1.0] の範囲に収まることを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        for v in result.explained_variance:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_explained_variance_sum_le_one(self) -> None:
        """寄与率の合計が 1.0 以下であることを確認（上位 2 主成分のみ）。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertLessEqual(sum(result.explained_variance), 1.0 + 1e-6)

    def test_explained_variance_descending(self) -> None:
        """第1主成分の寄与率が第2主成分以上であることを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertGreaterEqual(
            result.explained_variance[0],
            result.explained_variance[1],
        )

    def test_n_samples_field(self) -> None:
        """n_samples が入力件数と一致することを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.n_samples, 20)

    def test_seed_field(self) -> None:
        """seed フィールドが指定値と一致することを確認。"""
        proj = Projector(method="pca", seed=7)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.seed, 7)

    def test_cluster_labels_initially_none(self) -> None:
        """fit_transform 直後の cluster_labels は None であることを確認。"""
        proj = Projector(method="pca", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertIsNone(result.cluster_labels)


# ---------------------------------------------------------------------------
# fit_transform() — UMAP
# ---------------------------------------------------------------------------

class TestFitTransformUMAP(unittest.TestCase):
    """Projector.fit_transform() の UMAP 経路テスト。"""

    def setUp(self) -> None:
        # UMAP は n_neighbors のデフォルトが 15 なので、N >= 16 が安定
        self.vectors = _random_vectors(n=20, dim=8)

    def test_coords_shape(self) -> None:
        """coords_2d の shape が (N, 2) であることを確認。"""
        proj = Projector(method="umap", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.coords_2d.shape, (20, 2))

    def test_method_field(self) -> None:
        """method フィールドが 'umap' であることを確認。"""
        proj = Projector(method="umap", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.method, "umap")

    def test_explained_variance_is_empty(self) -> None:
        """UMAP では explained_variance が空リストであることを確認。"""
        proj = Projector(method="umap", seed=42)
        result = proj.fit_transform(self.vectors)
        self.assertEqual(result.explained_variance, [])


# ---------------------------------------------------------------------------
# fit_transform() — 入力バリデーション
# ---------------------------------------------------------------------------

class TestFitTransformValidation(unittest.TestCase):
    """fit_transform() の入力バリデーション。"""

    def test_invalid_input_not_ndarray(self) -> None:
        """np.ndarray 以外を渡すと InvalidVectorError が送出される。"""
        proj = Projector(method="pca", seed=42)
        with self.assertRaises(InvalidVectorError):
            proj.fit_transform([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    def test_invalid_input_1d(self) -> None:
        """1次元配列を渡すと InvalidVectorError が送出される。"""
        proj = Projector(method="pca", seed=42)
        with self.assertRaises(InvalidVectorError):
            proj.fit_transform(np.array([1.0, 2.0, 3.0]))

    def test_too_few_samples(self) -> None:
        """サンプル数が 2 未満の場合に InvalidVectorError が送出される。"""
        proj = Projector(method="pca", seed=42)
        with self.assertRaises(InvalidVectorError):
            proj.fit_transform(np.array([[1.0, 2.0, 3.0]]))


# ---------------------------------------------------------------------------
# attach_clusters()
# ---------------------------------------------------------------------------

class TestAttachClusters(unittest.TestCase):
    """Projector.attach_clusters() のテスト。"""

    def setUp(self) -> None:
        self.vectors = _random_vectors(n=10, dim=4)
        self.proj = Projector(method="pca", seed=42)
        self.result = self.proj.fit_transform(self.vectors)
        self.labels = np.array([0, 1, 0, 2, 1, 0, 2, 1, 0, 2])

    def test_returns_projection_result(self) -> None:
        """戻り値が ProjectionResult であることを確認。"""
        attached = self.proj.attach_clusters(self.result, self.labels)
        self.assertIsInstance(attached, ProjectionResult)

    def test_cluster_labels_assigned(self) -> None:
        """cluster_labels が指定したラベル配列と一致することを確認。"""
        attached = self.proj.attach_clusters(self.result, self.labels)
        np.testing.assert_array_equal(attached.cluster_labels, self.labels)

    def test_immutable_does_not_modify_input(self) -> None:
        """元の result の cluster_labels は変更されない（イミュータブル）ことを確認。"""
        _ = self.proj.attach_clusters(self.result, self.labels)
        self.assertIsNone(self.result.cluster_labels)

    def test_returns_new_object(self) -> None:
        """戻り値が元の result と別オブジェクトであることを確認。"""
        attached = self.proj.attach_clusters(self.result, self.labels)
        self.assertIsNot(attached, self.result)

    def test_coords_preserved(self) -> None:
        """coords_2d / explained_variance / method が引き継がれることを確認。"""
        attached = self.proj.attach_clusters(self.result, self.labels)
        np.testing.assert_array_equal(attached.coords_2d, self.result.coords_2d)
        self.assertEqual(
            attached.explained_variance, self.result.explained_variance,
        )
        self.assertEqual(attached.method, self.result.method)

    def test_labels_length_mismatch_raises(self) -> None:
        """ラベル配列の長さが n_samples と一致しないと InvalidVectorError が送出される。"""
        wrong = np.array([0, 1, 0])  # 長さ 3、result.n_samples=10
        with self.assertRaises(InvalidVectorError):
            self.proj.attach_clusters(self.result, wrong)

    def test_invalid_result_type(self) -> None:
        """result が ProjectionResult でない場合に TypeError が送出される。"""
        with self.assertRaises(TypeError):
            self.proj.attach_clusters("not_a_result", self.labels)  # type: ignore[arg-type]

    def test_invalid_labels_type(self) -> None:
        """cluster_labels が np.ndarray でない場合に TypeError が送出される。"""
        with self.assertRaises(TypeError):
            self.proj.attach_clusters(self.result, [0, 1, 0, 2, 1, 0, 2, 1, 0, 2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 再現性
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):
    """seed 固定下での完全再現性のテスト。"""

    def test_pca_same_seed_same_coords(self) -> None:
        """PCA で同一 seed・同一データの coords_2d が完全一致することを確認。"""
        vectors = _random_vectors(n=20, dim=8, seed=0)
        coords_1 = Projector(method="pca", seed=42).fit_transform(vectors).coords_2d
        coords_2 = Projector(method="pca", seed=42).fit_transform(vectors).coords_2d
        np.testing.assert_array_equal(coords_1, coords_2)

    def test_umap_same_seed_same_coords(self) -> None:
        """UMAP で同一 seed・同一データの coords_2d が完全一致することを確認。"""
        vectors = _random_vectors(n=20, dim=8, seed=0)
        coords_1 = Projector(method="umap", seed=42).fit_transform(vectors).coords_2d
        coords_2 = Projector(method="umap", seed=42).fit_transform(vectors).coords_2d
        np.testing.assert_array_equal(coords_1, coords_2)


if __name__ == "__main__":
    unittest.main()
