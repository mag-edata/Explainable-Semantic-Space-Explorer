"""
distance_metrics.py
===================
コサイン類似度の完全自前実装。

numpy のみ使用。scipy / sklearn の cosine_similarity は一切使用しない。
外部ライブラリへの依存ゼロで、計算過程を完全にトレース可能にする。

数学的定義:
    L2 ノルム:
        ||v||_2 = sqrt(v_1^2 + v_2^2 + ... + v_D^2)
                = sqrt(v · v)

    コサイン類似度:
        cos(θ) = (a · b) / (||a||_2 × ||b||_2)
               = Σ(a_i × b_i) / (sqrt(Σa_i^2) × sqrt(Σb_i^2))

        範囲: [-1.0, 1.0]
            1.0  → 同一方向（完全に近い）
            0.0  → 直交（無関係）
           -1.0  → 逆方向（完全に遠い）
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# ゼロ除算を防ぐ最小ノルム値
_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# カスタム例外
# ---------------------------------------------------------------------------

class DistanceMetricsError(Exception):
    """DistanceMetrics 固有の例外基底クラス。"""


class VectorDimensionError(DistanceMetricsError):
    """ベクトルの次元・形状が不正な場合の例外。

    例: 2ベクトルの次元数が一致しない、1D でない配列が渡された場合。
    """


# ---------------------------------------------------------------------------
# DistanceMetrics
# ---------------------------------------------------------------------------

class DistanceMetrics:
    """コサイン類似度の完全自前実装クラス。

    全メソッドは staticmethod。インスタンス化不要で使用可能。
    numpy のみを使用し、計算過程を完全にトレースできる設計にする。

    数式の明示:
        L2 ノルム:   ||v||_2 = sqrt(v · v)
        コサイン:    cos(θ)  = (a · b) / (||a|| × ||b||)
        バッチ計算:  sim_i   = (M_i · q) / (||M_i|| × ||q||)

    使用例::

        metrics = DistanceMetrics()

        norm = DistanceMetrics.l2_norm(vec)
        sim  = DistanceMetrics.cosine_similarity(vec_a, vec_b)
        sims = DistanceMetrics.cosine_similarity_batch(query, matrix)
        info = DistanceMetrics.explain(vec_a, vec_b)
    """

    @staticmethod
    def l2_norm(vector: np.ndarray) -> float:
        """ベクトルの L2 ノルム（ユークリッドノルム）を計算する。

        数式:
            ||v||_2 = sqrt(v_1^2 + v_2^2 + ... + v_D^2)
                    = sqrt(v · v)

        実装: np.linalg.norm は使用しない。
        np.sqrt(np.dot(v, v)) で自前計算する。

        Args:
            vector: 1次元 ndarray、shape (D,)。

        Returns:
            float: L2 ノルム。ゼロベクトルの場合は 0.0。

        Raises:
            VectorDimensionError: vector が 1次元でない場合。
            TypeError:            vector が ndarray でない場合。
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError(
                f"vector は np.ndarray 型である必要があります。"
                f"受け取った型: {type(vector)}"
            )
        if vector.ndim != 1:
            raise VectorDimensionError(
                f"vector は 1次元配列である必要があります。"
                f"受け取った shape: {vector.shape}"
            )

        # np.linalg.norm を使わない自前実装
        # sqrt(v · v) = sqrt(v_1^2 + v_2^2 + ... + v_D^2)
        return float(np.sqrt(np.dot(vector, vector)))

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """2ベクトル間のコサイン類似度を計算する。

        数式:
            cos(θ) = (a · b) / (||a||_2 × ||b||_2)
                   = Σ(a_i × b_i) / (sqrt(Σa_i^2) × sqrt(Σb_i^2))

        ゼロ除算ガード:
            ||a|| < ε または ||b|| < ε の場合は 0.0 を返す。
            ゼロベクトルは意味空間上の位置を持たないため、
            類似度を定義できないと判断する。

        Args:
            vec_a: 1次元 ndarray、shape (D,)。
            vec_b: 1次元 ndarray、shape (D,)。

        Returns:
            float: コサイン類似度。範囲: [-1.0, 1.0]。

        Raises:
            VectorDimensionError: どちらかが 1次元でない、または次元数が一致しない場合。
            TypeError:            どちらかが ndarray でない場合。
        """
        _validate_vector_pair(vec_a, vec_b)

        dot: float = float(np.dot(vec_a, vec_b))
        norm_a: float = DistanceMetrics.l2_norm(vec_a)
        norm_b: float = DistanceMetrics.l2_norm(vec_b)

        # ゼロ除算ガード: ゼロベクトルは類似度 0.0 とする
        if norm_a < _EPSILON or norm_b < _EPSILON:
            logger.debug(
                "cosine_similarity: ゼロベクトル検出 (norm_a=%.2e, norm_b=%.2e) -> 0.0",
                norm_a, norm_b,
            )
            return 0.0

        return dot / (norm_a * norm_b)

    @staticmethod
    def cosine_similarity_batch(
        query: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """1クエリと行列全体のコサイン類似度を一括計算する。

        数式（行列演算）:
            dot_products = M @ q                  shape: (N,)
            row_norms    = sqrt(diag(M @ M^T))    shape: (N,)
                         = sqrt(Σ M_ij^2 for each row i)
            query_norm   = ||q||_2                scalar
            similarities = dot_products / (row_norms × query_norm)

        ゼロ除算ガード:
            denominator_i < ε の要素は 0.0 に設定する。

        計算量:
            O(N × D): 全語彙に対して1回のバッチ演算。
            np.argpartition と組み合わせると Top-K 検索は O(N + k log k)。

        Args:
            query:  クエリベクトル。1次元 ndarray、shape (D,)。
            matrix: 埋め込み行列。2次元 ndarray、shape (N, D)。

        Returns:
            np.ndarray: 各行とのコサイン類似度、shape (N,)。
                        dtype は float64。

        Raises:
            VectorDimensionError: query が 1次元でない、matrix が 2次元でない、
                                  または query の次元と matrix の列数が一致しない場合。
            TypeError:            query または matrix が ndarray でない場合。
        """
        _validate_query_matrix(query, matrix)

        # 内積: M @ q → shape (N,)
        # 行列とベクトルの積で全行との内積を一括計算
        dot_products: np.ndarray = matrix @ query

        # クエリのノルム（自前実装を使用）
        query_norm: float = DistanceMetrics.l2_norm(query)

        # 各行の L2 ノルム: sqrt(Σ M_ij^2) → shape (N,)
        # np.linalg.norm は使わない → 手動で sum of squares → sqrt
        row_norms: np.ndarray = np.sqrt((matrix * matrix).sum(axis=1))

        # 分母: ||M_i|| × ||q|| → shape (N,)
        denominators: np.ndarray = row_norms * query_norm

        # ゼロ除算ガード: ε 未満の分母を 1.0 に置換してから除算し、
        # 対応する結果を 0.0 に上書きする
        zero_mask: np.ndarray = denominators < _EPSILON
        safe_denominators: np.ndarray = np.where(zero_mask, 1.0, denominators)

        similarities: np.ndarray = dot_products / safe_denominators
        similarities[zero_mask] = 0.0

        if zero_mask.any():
            logger.debug(
                "cosine_similarity_batch: ゼロ除算ガード適用 %d 件",
                int(zero_mask.sum()),
            )

        return similarities

    @staticmethod
    def explain(vec_a: np.ndarray, vec_b: np.ndarray) -> Dict[str, Any]:
        """コサイン類似度の計算過程を辞書形式で返す。

        UI での「なぜこのスコアか」説明表示に使用する。
        計算式を文字列として含めることで、数値だけでなく
        式の構造ごと出力に埋め込める。

        数式（文字列 formula に展開される）:
            cos(θ) = (a · b) / (||a||_2 × ||b||_2)
                   = {dot_product:.6f} / ({norm_a:.6f} × {norm_b:.6f})
                   = {similarity:.6f}

        Args:
            vec_a: クエリ単語のベクトル、1次元 ndarray、shape (D,)。
            vec_b: 対象単語のベクトル、1次元 ndarray、shape (D,)。

        Returns:
            Dict[str, Any]: 計算内訳の辞書。

            - "dot_product"  (float): 内積  a · b
            - "norm_a"       (float): クエリベクトルの L2 ノルム  ||a||_2
            - "norm_b"       (float): 対象ベクトルの L2 ノルム   ||b||_2
            - "denominator"  (float): 分母  ||a||_2 × ||b||_2
            - "similarity"   (float): コサイン類似度  cos(θ)
            - "formula"      (str):   計算式を値込みで展開した文字列

        Raises:
            VectorDimensionError: 入力が不正な場合（l2_norm / cosine_similarity に準拠）。
            TypeError:            入力が ndarray でない場合。
        """
        _validate_vector_pair(vec_a, vec_b)

        dot_product: float = float(np.dot(vec_a, vec_b))
        norm_a: float = DistanceMetrics.l2_norm(vec_a)
        norm_b: float = DistanceMetrics.l2_norm(vec_b)
        denominator: float = norm_a * norm_b
        similarity: float = DistanceMetrics.cosine_similarity(vec_a, vec_b)

        formula: str = (
            f"cos(θ) = (a · b) / (||a|| × ||b||)"
            f" = {dot_product:.6f} / ({norm_a:.6f} × {norm_b:.6f})"
            f" = {dot_product:.6f} / {denominator:.6f}"
            f" = {similarity:.6f}"
        )

        return {
            "dot_product": dot_product,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "denominator": denominator,
            "similarity": similarity,
            "formula": formula,
        }


# ---------------------------------------------------------------------------
# モジュール内プライベートヘルパー（バリデーション共通化）
# ---------------------------------------------------------------------------

def _validate_vector_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> None:
    """2ベクトルの型・形状・次元数の整合性を検証する。

    Args:
        vec_a: 検証対象のベクトル A。
        vec_b: 検証対象のベクトル B。

    Raises:
        TypeError:            ndarray でない場合。
        VectorDimensionError: 1次元でない、または次元数が一致しない場合。
    """
    for name, vec in (("vec_a", vec_a), ("vec_b", vec_b)):
        if not isinstance(vec, np.ndarray):
            raise TypeError(
                f"{name} は np.ndarray 型である必要があります。"
                f"受け取った型: {type(vec)}"
            )
        if vec.ndim != 1:
            raise VectorDimensionError(
                f"{name} は 1次元配列である必要があります。"
                f"受け取った shape: {vec.shape}"
            )

    if vec_a.shape != vec_b.shape:
        raise VectorDimensionError(
            f"vec_a と vec_b の次元数が一致しません。"
            f"vec_a: {vec_a.shape}, vec_b: {vec_b.shape}"
        )


def _validate_query_matrix(query: np.ndarray, matrix: np.ndarray) -> None:
    """バッチ計算用のクエリ・行列の型・形状を検証する。

    Args:
        query:  クエリベクトル。
        matrix: 埋め込み行列。

    Raises:
        TypeError:            ndarray でない場合。
        VectorDimensionError: 形状・次元数が不正な場合。
    """
    if not isinstance(query, np.ndarray):
        raise TypeError(
            f"query は np.ndarray 型である必要があります。"
            f"受け取った型: {type(query)}"
        )
    if not isinstance(matrix, np.ndarray):
        raise TypeError(
            f"matrix は np.ndarray 型である必要があります。"
            f"受け取った型: {type(matrix)}"
        )
    if query.ndim != 1:
        raise VectorDimensionError(
            f"query は 1次元配列である必要があります。"
            f"受け取った shape: {query.shape}"
        )
    if matrix.ndim != 2:
        raise VectorDimensionError(
            f"matrix は 2次元配列である必要があります。"
            f"受け取った shape: {matrix.shape}"
        )
    if query.shape[0] != matrix.shape[1]:
        raise VectorDimensionError(
            f"query の次元数と matrix の列数が一致しません。"
            f"query: {query.shape[0]}, matrix 列数: {matrix.shape[1]}"
        )
