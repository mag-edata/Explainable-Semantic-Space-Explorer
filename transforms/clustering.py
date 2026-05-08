"""
clustering.py
=============
KMeans クラスタリングによる単語埋め込みのグループ化モジュール。

コサイン距離に基づくクラスタリングを実現するため、
ベクトルを L2 正規化したうえで sklearn の KMeans（ユークリッド距離）を適用する。

    cos(a, b) = dot(a, b) / (‖a‖ · ‖b‖)

unit vector に正規化すると ‖â‖ = ‖b̂‖ = 1 であるため、
    ‖â - b̂‖² = 2 - 2·cos(a, b)
となり、ユークリッド距離の最小化がコサイン距離の最小化と等価になる。

制約:
    - ノルム計算は自前実装（np.linalg.norm 禁止）
    - コサイン/正規化処理は自前実装（sklearn の cosine_similarity 禁止）
    - 乱数 seed は必ず固定（再現性保証）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ---------- 定数 ----------
DEFAULT_SEED: int = 42
DEFAULT_N_CLUSTERS: int = 8
DEFAULT_MAX_ITER: int = 300
EPSILON: float = 1e-10  # ゼロ除算ガード（ノルムがほぼゼロのベクトル対策）


# ---------- 例外クラス ----------


class ClusterError(Exception):
    """cluster モジュール固有の例外基底クラス。"""


class NotFittedError(ClusterError):
    """fit() を呼ぶ前に get_labels() / get_result() を呼び出した場合の例外。"""


class InvalidClusterCountError(ClusterError):
    """n_clusters の値が不正な場合の例外（0以下、または語彙数を超える）。"""


class UnfitVectorError(ClusterError):
    """入力ベクトルの型や形状が不正な場合の例外。"""


# ---------- データクラス ----------


@dataclass
class ClusterResult:
    """KMeans クラスタリングの結果サマリー。

    Attributes:
        labels:     各単語のクラスタID（0始まり）。shape (N,)。
        n_clusters: 指定したクラスタ数。
        inertia:    クラスタ内ユークリッド二乗和（正規化空間での値）。
                    値が小さいほど密なクラスタを形成している。
        seed:       使用した乱数シード（再現性確認用）。
        n_samples:  クラスタリング対象の語彙数 N。
    """

    labels: np.ndarray
    n_clusters: int
    inertia: float
    seed: int
    n_samples: int


# ---------- メインクラス ----------


class KMeansClusterer:
    """コサイン距離ベースの KMeans クラスタリングクラス。

    内部では入力ベクトルを L2 正規化したうえで
    sklearn の KMeans（ユークリッド距離）を適用する。
    これによりコサイン距離によるクラスタリングと等価な結果を得る。

    数式（コサイン ↔ ユークリッド の等価変換）:
        ‖â - b̂‖² = 2 - 2·cos(a, b)  （â, b̂ は unit vector）

    乱数シードは初期化時に固定し、再現性を保証する。

    Attributes:
        _n_clusters: クラスタ数。
        _seed:       乱数シード。
        _max_iter:   KMeans 最大反復回数。
        _result:     fit() 後に格納される ClusterResult。None = 未 fit。
    """

    def __init__(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        seed: int = DEFAULT_SEED,
        max_iter: int = DEFAULT_MAX_ITER,
    ) -> None:
        """KMeansClusterer を初期化する。

        Args:
            n_clusters: クラスタ数（デフォルト: 8）。
                        fit() 時に語彙数 N との整合性を検証する。
            seed:       乱数シード（デフォルト: 42）。
                        同一データ・同一シードで完全再現を保証する。
            max_iter:   KMeans の最大反復回数（デフォルト: 300）。

        Raises:
            TypeError:              n_clusters / seed / max_iter が int でない場合。
            InvalidClusterCountError: n_clusters が 1 未満の場合。
        """
        if not isinstance(n_clusters, int):
            raise TypeError(
                f"n_clusters は int 型である必要があります。"
                f"受け取った型: {type(n_clusters)}"
            )
        if not isinstance(seed, int):
            raise TypeError(
                f"seed は int 型である必要があります。"
                f"受け取った型: {type(seed)}"
            )
        if not isinstance(max_iter, int):
            raise TypeError(
                f"max_iter は int 型である必要があります。"
                f"受け取った型: {type(max_iter)}"
            )
        if n_clusters < 1:
            raise InvalidClusterCountError(
                f"n_clusters は 1 以上である必要があります。"
                f"指定値: {n_clusters}"
            )

        self._n_clusters: int = n_clusters
        self._seed: int = seed
        self._max_iter: int = max_iter
        self._result: ClusterResult | None = None

        logger.info(
            "KMeansClusterer 初期化完了: n_clusters=%d, seed=%d, max_iter=%d",
            self._n_clusters,
            self._seed,
            self._max_iter,
        )

    # ---------- 公開 API ----------

    def fit(self, vectors: np.ndarray) -> ClusterResult:
        """ベクトル行列に KMeans クラスタリングを適用する。

        入力ベクトルを L2 正規化してからクラスタリングを実行する。
        正規化後は Euclidean 距離最小化 ≡ コサイン距離最小化 となる。

        処理フロー:
            1. 入力バリデーション
            2. L2 ノルム一括計算: ‖v_i‖ = sqrt(Σ v_{ij}²)
            3. 行単位の L2 正規化:  v̂_i = v_i / ‖v_i‖
            4. sklearn.KMeans.fit(unit_vectors)
            5. ClusterResult 構築・返却

        Args:
            vectors: 埋め込み行列。shape (N, D)。dtype は float32 推奨。

        Returns:
            ClusterResult: クラスタラベル・inertia 等を含む結果オブジェクト。

        Raises:
            UnfitVectorError:         vectors が np.ndarray でない、または次元数が 2 でない場合。
            InvalidClusterCountError: n_clusters が語彙数 N を超える場合。
        """
        self._validate_inputs(vectors)

        n_samples: int = vectors.shape[0]
        logger.debug(
            "fit 開始: n_samples=%d, dim=%d, n_clusters=%d",
            n_samples,
            vectors.shape[1],
            self._n_clusters,
        )

        unit_vectors: np.ndarray = self._normalize_rows(vectors)

        kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._seed,
            max_iter=self._max_iter,
        )
        kmeans.fit(unit_vectors)

        self._result = ClusterResult(
            labels=kmeans.labels_.copy(),
            n_clusters=self._n_clusters,
            inertia=float(kmeans.inertia_),
            seed=self._seed,
            n_samples=n_samples,
        )

        logger.info(
            "fit 完了: n_samples=%d, n_clusters=%d, inertia=%.6f",
            n_samples,
            self._n_clusters,
            self._result.inertia,
        )

        return self._result

    def get_labels(self) -> np.ndarray:
        """クラスタラベル配列を返す。

        Returns:
            np.ndarray: 各単語のクラスタID（0始まり）。shape (N,)。

        Raises:
            NotFittedError: fit() を呼び出す前に呼んだ場合。
        """
        if self._result is None:
            raise NotFittedError(
                "get_labels() は fit() を呼び出した後にのみ使用できます。"
            )
        return self._result.labels

    def get_result(self) -> ClusterResult:
        """クラスタリング結果全体を返す。

        Returns:
            ClusterResult: labels / n_clusters / inertia / seed / n_samples を含む結果。

        Raises:
            NotFittedError: fit() を呼び出す前に呼んだ場合。
        """
        if self._result is None:
            raise NotFittedError(
                "get_result() は fit() を呼び出した後にのみ使用できます。"
            )
        return self._result

    # ---------- 自前実装: ノルム・正規化（staticmethod）----------

    @staticmethod
    def _l2_norm_batch(matrix: np.ndarray) -> np.ndarray:
        """行列の各行について L2 ノルムを一括計算する。

        数式:
            ‖v‖ = sqrt(Σ_j v_j²)   for each row v

        実装は np.linalg.norm を使わず、自前で平方和 → sqrt を計算する:
            norms = sqrt( sum(matrix * matrix, axis=1) )
        ゼロベクトルへの除算を防ぐため、結果に EPSILON を加算してクランプしない
        （正規化ステップ側で別途ガードする）。

        Args:
            matrix: 入力行列。shape (N, D)。

        Returns:
            np.ndarray: 各行の L2 ノルム。shape (N,)。
        """
        row_sq_sum: np.ndarray = (matrix * matrix).sum(axis=1)
        norms: np.ndarray = np.sqrt(row_sq_sum)
        return norms

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """行列の各行を L2 ノルムで正規化し、unit vector 行列を返す。

        数式:
            â_i = a_i / ‖a_i‖   for each row a_i

        ゼロベクトル（‖a_i‖ < EPSILON）はゼロのまま保持する（除算スキップ）。
        これにより NaN の発生を防ぐ。

        Args:
            matrix: 入力行列。shape (N, D)。

        Returns:
            np.ndarray: 正規化済み行列。shape (N, D)。各行の L2 ノルムは 1.0（ゼロ行を除く）。
        """
        norms: np.ndarray = KMeansClusterer._l2_norm_batch(matrix)

        # ゼロベクトルのマスク（除算を回避）
        valid_mask: np.ndarray = norms >= EPSILON

        # コピーして正規化（元の matrix を変更しない）
        unit_matrix: np.ndarray = matrix.copy().astype(np.float64)
        unit_matrix[valid_mask] = (
            matrix[valid_mask] / norms[valid_mask, np.newaxis]
        )

        logger.debug(
            "_normalize_rows: %d / %d 行を正規化（ゼロベクトル %d 行をスキップ）",
            int(valid_mask.sum()),
            matrix.shape[0],
            int((~valid_mask).sum()),
        )

        return unit_matrix

    # ---------- バリデーション ----------

    def _validate_inputs(self, vectors: np.ndarray) -> None:
        """fit() の入力を検証する。

        Args:
            vectors: 検証対象の埋め込み行列。

        Raises:
            UnfitVectorError:         vectors が np.ndarray でない場合。
            UnfitVectorError:         vectors の次元数が 2 でない場合。
            InvalidClusterCountError: n_clusters が語彙数 N を超える場合。
        """
        if not isinstance(vectors, np.ndarray):
            raise UnfitVectorError(
                f"vectors は np.ndarray 型である必要があります。"
                f"受け取った型: {type(vectors)}"
            )
        if vectors.ndim != 2:
            raise UnfitVectorError(
                f"vectors は 2 次元配列（shape (N, D)）である必要があります。"
                f"受け取った次元数: {vectors.ndim}"
            )

        n_samples: int = vectors.shape[0]
        if self._n_clusters > n_samples:
            raise InvalidClusterCountError(
                f"n_clusters ({self._n_clusters}) が語彙数 N ({n_samples}) を超えています。"
                f"n_clusters ≤ N である必要があります。"
            )
