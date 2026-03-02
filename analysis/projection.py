"""
projection.py
=============
高次元埋め込みベクトルを 2D 座標に投影するモジュール。

対応手法:
    - PCA  (Principal Component Analysis)  : 線形次元削減
    - UMAP (Uniform Manifold Approximation): 非線形次元削減

どちらも seed を固定することで完全再現性を保証する。
UMAP は局所的な近傍構造を保持し、PCA より視覚的なクラスタ分離が明確になる傾向がある。
一方 PCA は主成分寄与率（explained_variance_ratio_）で投影の情報保持量を定量化できる。

数式（PCA）:
    Z = X · V^T      （V は上位 k 主成分の行列、ここで k=2）
    各主成分の寄与率 = λ_i / Σλ_j

制約:
    - 乱数 seed は必ず固定（再現性保証）
    - Streamlit / 外部 API への依存禁止
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

logger = logging.getLogger(__name__)

# ---------- 定数 ----------
DEFAULT_SEED: int = 42
DEFAULT_METHOD: str = "pca"
SUPPORTED_METHODS: tuple[str, ...] = ("pca", "umap")


# ---------- 例外クラス ----------


class ProjectionError(Exception):
    """projection モジュール固有の例外基底クラス。"""


class NotFittedError(ProjectionError):
    """fit_transform() を呼ぶ前に結果を参照しようとした場合の例外。"""


class InvalidMethodError(ProjectionError):
    """サポートされていない投影手法が指定された場合の例外。"""


class InvalidVectorError(ProjectionError):
    """入力ベクトルの型や形状が不正な場合の例外。"""


# ---------- データクラス ----------


@dataclass
class ProjectionResult:
    """2D 投影の結果サマリー。

    Attributes:
        coords_2d:         各単語の 2D 座標。shape (N, 2)。
                           coords_2d[i] = (x, y) が語彙インデックス i に対応する。
        explained_variance: 主成分の寄与率リスト（PCA のみ）。
                           [第1主成分の寄与率, 第2主成分の寄与率]（0.0〜1.0）。
                           UMAP の場合は空リスト []。
        method:            使用した投影手法名 ("pca" または "umap")。
        cluster_labels:    クラスタID 配列。shape (N,)。
                           attach_clusters() で付与するまでは None。
        n_samples:         投影した語彙数 N。
        seed:              使用した乱数シード（再現性確認用）。
    """

    coords_2d: np.ndarray
    explained_variance: List[float]
    method: str
    cluster_labels: np.ndarray | None
    n_samples: int
    seed: int


# ---------- メインクラス ----------


class Projector:
    """埋め込みベクトルを PCA または UMAP で 2D に投影するクラス。

    1インスタンス = 1手法 の設計。
    手法を変える場合は別インスタンスを生成する。

    乱数シードは初期化時に固定し、同一データ・同一シードで完全再現を保証する。

    Attributes:
        _method: 投影手法名 ("pca" または "umap")。
        _seed:   乱数シード。
    """

    def __init__(
        self,
        method: str = DEFAULT_METHOD,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Projector を初期化する。

        Args:
            method: 投影手法 ("pca" または "umap")。デフォルト: "pca"。
            seed:   乱数シード（デフォルト: 42）。再現性のために固定する。

        Raises:
            InvalidMethodError: method が "pca" / "umap" 以外の場合。
            TypeError:          method が str でない場合。
                                seed が int でない場合。
        """
        if not isinstance(method, str):
            raise TypeError(
                f"method は str 型である必要があります。"
                f"受け取った型: {type(method)}"
            )
        if not isinstance(seed, int):
            raise TypeError(
                f"seed は int 型である必要があります。"
                f"受け取った型: {type(seed)}"
            )
        if method not in SUPPORTED_METHODS:
            raise InvalidMethodError(
                f"method は {SUPPORTED_METHODS} のいずれかである必要があります。"
                f"指定値: '{method}'"
            )

        self._method: str = method
        self._seed: int = seed

        logger.info(
            "Projector 初期化完了: method=%s, seed=%d",
            self._method,
            self._seed,
        )

    # ---------- 公開 API ----------

    def fit_transform(self, vectors: np.ndarray) -> ProjectionResult:
        """埋め込み行列を 2D 座標に変換する。

        PCA の場合:
            Z = X · V^T  （V は上位 2 主成分）
            主成分寄与率を explained_variance に格納する。

        UMAP の場合:
            局所的な近傍グラフを構築し、低次元に射影する。
            explained_variance は [] となる（非線形手法のため定義なし）。

        Args:
            vectors: 埋め込み行列。shape (N, D)。dtype は float32 推奨。

        Returns:
            ProjectionResult: 2D 座標・寄与率・手法名等を含む結果オブジェクト。

        Raises:
            InvalidVectorError: vectors が np.ndarray でない、または ndim != 2 の場合。
        """
        self._validate_inputs(vectors)

        n_samples: int = vectors.shape[0]
        logger.debug(
            "fit_transform 開始: method=%s, n_samples=%d, dim=%d",
            self._method,
            n_samples,
            vectors.shape[1],
        )

        if self._method == "pca":
            result = self._fit_pca(vectors)
        else:
            result = self._fit_umap(vectors)

        logger.info(
            "fit_transform 完了: method=%s, n_samples=%d, coords_2d.shape=%s",
            self._method,
            n_samples,
            result.coords_2d.shape,
        )

        return result

    def attach_clusters(
        self,
        result: ProjectionResult,
        cluster_labels: np.ndarray,
    ) -> ProjectionResult:
        """既存の ProjectionResult にクラスタラベルを付与して返す。

        元の result は変更せず、cluster_labels を差し替えた新しい
        ProjectionResult を返す（イミュータブルな操作）。

        Args:
            result:         fit_transform() が返した ProjectionResult。
            cluster_labels: 各単語のクラスタID 配列。shape (N,)。
                            KMeansClusterer.get_labels() の出力を渡す。

        Returns:
            ProjectionResult: cluster_labels が付与された新しい結果オブジェクト。

        Raises:
            TypeError:         result が ProjectionResult でない場合。
                               cluster_labels が np.ndarray でない場合。
            InvalidVectorError: cluster_labels の長さが result.n_samples と一致しない場合。
        """
        if not isinstance(result, ProjectionResult):
            raise TypeError(
                f"result は ProjectionResult 型である必要があります。"
                f"受け取った型: {type(result)}"
            )
        if not isinstance(cluster_labels, np.ndarray):
            raise TypeError(
                f"cluster_labels は np.ndarray 型である必要があります。"
                f"受け取った型: {type(cluster_labels)}"
            )
        if cluster_labels.shape[0] != result.n_samples:
            raise InvalidVectorError(
                f"cluster_labels の長さ ({cluster_labels.shape[0]}) が "
                f"result.n_samples ({result.n_samples}) と一致しません。"
            )

        logger.debug(
            "attach_clusters: n_samples=%d, unique_clusters=%d",
            result.n_samples,
            int(len(set(cluster_labels.tolist()))),
        )

        return ProjectionResult(
            coords_2d=result.coords_2d,
            explained_variance=result.explained_variance,
            method=result.method,
            cluster_labels=cluster_labels.copy(),
            n_samples=result.n_samples,
            seed=result.seed,
        )

    # ---------- 手法別の投影処理（private）----------

    def _fit_pca(self, vectors: np.ndarray) -> ProjectionResult:
        """PCA で 2D 投影を実行する。

        数式:
            Z = X · V^T
            寄与率_i = λ_i / Σ_j λ_j   （λ は固有値）

        sklearn.decomposition.PCA を使用。
        random_state に seed を渡すことで再現性を保証する。

        Args:
            vectors: 埋め込み行列。shape (N, D)。

        Returns:
            ProjectionResult: 2D 座標と主成分寄与率を含む結果。
        """
        pca = PCA(n_components=2, random_state=self._seed)
        coords: np.ndarray = pca.fit_transform(vectors)

        explained: List[float] = pca.explained_variance_ratio_.tolist()

        logger.debug(
            "PCA 完了: 第1主成分 寄与率=%.4f, 第2主成分 寄与率=%.4f",
            explained[0],
            explained[1],
        )

        return ProjectionResult(
            coords_2d=coords,
            explained_variance=explained,
            method="pca",
            cluster_labels=None,
            n_samples=vectors.shape[0],
            seed=self._seed,
        )

    def _fit_umap(self, vectors: np.ndarray) -> ProjectionResult:
        """UMAP で 2D 投影を実行する。

        UMAP は局所的な近傍構造を保ちながら低次元に射影する非線形手法。
        PCA と異なり explained_variance の概念がないため、空リストを格納する。

        random_state に seed を渡すことで再現性を保証する。

        Args:
            vectors: 埋め込み行列。shape (N, D)。

        Returns:
            ProjectionResult: 2D 座標を含む結果（explained_variance は []）。
        """
        reducer = UMAP(n_components=2, random_state=self._seed)
        coords: np.ndarray = reducer.fit_transform(vectors)

        logger.debug("UMAP 完了: coords_2d.shape=%s", coords.shape)

        return ProjectionResult(
            coords_2d=coords,
            explained_variance=[],
            method="umap",
            cluster_labels=None,
            n_samples=vectors.shape[0],
            seed=self._seed,
        )

    # ---------- バリデーション ----------

    def _validate_inputs(self, vectors: np.ndarray) -> None:
        """fit_transform() の入力を検証する。

        Args:
            vectors: 検証対象の埋め込み行列。

        Raises:
            InvalidVectorError: vectors が np.ndarray でない場合。
            InvalidVectorError: vectors の ndim が 2 でない場合。
            InvalidVectorError: vectors の行数が 2 未満の場合（PCA の最低要件）。
        """
        if not isinstance(vectors, np.ndarray):
            raise InvalidVectorError(
                f"vectors は np.ndarray 型である必要があります。"
                f"受け取った型: {type(vectors)}"
            )
        if vectors.ndim != 2:
            raise InvalidVectorError(
                f"vectors は 2 次元配列（shape (N, D)）である必要があります。"
                f"受け取った次元数: {vectors.ndim}"
            )
        if vectors.shape[0] < 2:
            raise InvalidVectorError(
                f"投影には 2 サンプル以上必要です。"
                f"受け取った行数: {vectors.shape[0]}"
            )
