"""
analyzer.py
===========
距離分布の統計分析クラス。

SimilarityEngine.get_distance_distribution() の出力を受け取り、
以下の分析を提供する:

- 各 SearchResult に対する Z-score の付与
- ヒストグラムのビン集計（可視化用）
- static vs contextual の分布比較
- 近傍安定性スコア（Top-K の重複率）

DistanceMetrics・SimilarityEngine には依存しない（分析のみ）。
numpy のみを使用する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from core.similarity_engine import ComparisonResult, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# カスタム例外
# ---------------------------------------------------------------------------

class AnalyzerError(Exception):
    """Analyzer 固有の例外基底クラス。"""


class InsufficientDataError(AnalyzerError):
    """統計計算に必要なデータが不足している場合の例外。

    例: histogram_data が空、または std=0 で Z-score が計算できない場合。
    """


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class DistributionStats:
    """距離分布の統計サマリー。

    get_distance_distribution() の出力を受け取り、
    追加統計を付与した構造体。

    Attributes:
        query_word:     クエリ単語
        mean:           全語彙との平均コサイン類似度
        std:            標準偏差
        top1_similarity:Top-1 の類似度スコア
        z_score:        Top-1 スコアの Z-score = (top1 - mean) / std
        median:         中央値
        q25:            第1四分位数（25パーセンタイル）
        q75:            第3四分位数（75パーセンタイル）
        n_samples:      サンプル数（語彙サイズ - 1）
    """

    query_word: str
    mean: float
    std: float
    top1_similarity: float
    z_score: float
    median: float
    q25: float
    q75: float
    n_samples: int


@dataclass
class HistogramData:
    """ヒストグラムのビン集計結果。

    Attributes:
        bin_edges:  ビンの境界値リスト（長さ n_bins + 1）
        counts:     各ビンの頻度リスト（長さ n_bins）
        n_bins:     ビン数
        data_min:   データの最小値
        data_max:   データの最大値
    """

    bin_edges: List[float]
    counts: List[int]
    n_bins: int
    data_min: float
    data_max: float


@dataclass
class DistributionComparison:
    """static vs contextual の分布比較結果。

    Attributes:
        query_word:      クエリ単語
        static_stats:    static エンジンの分布統計
        contextual_stats:     contextual エンジンの分布統計
        mean_diff:       平均コサイン類似度の差 (static - contextual)
        std_diff:        標準偏差の差 (static - contextual)
        z_score_diff:    Z-score の差 (static - contextual)
                         正値 = static の方が Top-1 が際立っている
    """

    query_word: str
    static_stats: DistributionStats
    contextual_stats: DistributionStats
    mean_diff: float
    std_diff: float
    z_score_diff: float


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class Analyzer:
    """距離分布の統計分析クラス。

    全メソッドは staticmethod。インスタンス化不要で使用可能。
    numpy のみを使用し、すべての計算を自前実装する。

    使用例::

        dist = engine.get_distance_distribution("king")
        stats = Analyzer.enrich_distribution(dist)

        hist = Analyzer.histogram(dist["histogram_data"], n_bins=50)

        result_scores = engine.search("king", top_k=10)
        scored = Analyzer.attach_z_scores(result_scores, dist)

        static_dist = static_engine.get_distance_distribution("king")
        contextual_dist  = contextual_engine.get_distance_distribution("king")
        cmp = Analyzer.compare_distributions("king", static_dist, contextual_dist)
    """

    @staticmethod
    def enrich_distribution(distribution: dict) -> DistributionStats:
        """get_distance_distribution() の出力に追加統計を付与する。

        中央値・四分位数を追加計算し、DistributionStats として返す。
        これにより分布の非対称性や外れ値の把握が容易になる。

        Args:
            distribution: SimilarityEngine.get_distance_distribution() の返り値。
                          必須キー: query_word, mean, std, top1_similarity,
                                    z_score, histogram_data

        Returns:
            DistributionStats: 追加統計付きの分布サマリー。

        Raises:
            KeyError:             必須キーが distribution に存在しない場合。
            InsufficientDataError: histogram_data が空の場合。
        """
        required_keys = {"query_word", "mean", "std", "top1_similarity",
                         "z_score", "histogram_data"}
        missing = required_keys - distribution.keys()
        if missing:
            raise KeyError(f"distribution に必須キーがありません: {missing}")

        data: np.ndarray = np.array(distribution["histogram_data"])
        if data.size == 0:
            raise InsufficientDataError(
                "histogram_data が空です。統計計算ができません"
            )

        median: float = float(np.median(data))
        q25: float = float(np.percentile(data, 25))
        q75: float = float(np.percentile(data, 75))

        logger.debug(
            "enrich_distribution: query=%s, median=%.4f, q25=%.4f, q75=%.4f",
            distribution["query_word"], median, q25, q75,
        )

        return DistributionStats(
            query_word=distribution["query_word"],
            mean=distribution["mean"],
            std=distribution["std"],
            top1_similarity=distribution["top1_similarity"],
            z_score=distribution["z_score"],
            median=median,
            q25=q25,
            q75=q75,
            n_samples=int(data.size),
        )

    @staticmethod
    def histogram(
        data: List[float],
        n_bins: int = 50,
    ) -> HistogramData:
        """類似度データをヒストグラムのビンに集計する。

        numpy の histogram を使用し、ビン数・境界値・頻度を返す。
        結果は Streamlit の棒グラフや st.bar_chart で直接使用可能。

        Args:
            data:   類似度スコアのリスト（histogram_data の値）。
            n_bins: ヒストグラムのビン数（デフォルト 50）。

        Returns:
            HistogramData: ビン集計結果。

        Raises:
            InsufficientDataError: data が空の場合。
            ValueError:            n_bins が 1 未満の場合。
        """
        if not data:
            raise InsufficientDataError(
                "data が空です。ヒストグラムを作成できません"
            )
        if n_bins < 1:
            raise ValueError(
                f"n_bins は 1 以上である必要があります。受け取った値: {n_bins}"
            )

        arr = np.array(data)
        counts, bin_edges = np.histogram(arr, bins=n_bins)

        logger.debug(
            "histogram: n_bins=%d, data_range=[%.4f, %.4f]",
            n_bins, float(arr.min()), float(arr.max()),
        )

        return HistogramData(
            bin_edges=bin_edges.tolist(),
            counts=counts.tolist(),
            n_bins=n_bins,
            data_min=float(arr.min()),
            data_max=float(arr.max()),
        )

    @staticmethod
    def attach_z_scores(
        results: List[SearchResult],
        distribution: dict,
    ) -> List[Dict]:
        """SearchResult リストに各結果の Z-score を付与した辞書リストを返す。

        各結果の類似度スコアを、全語彙の分布上での Z-score に変換する。
        Z-score は「この結果が平均から何標準偏差離れているか」を示す。

        計算式:
            z = (similarity - mean) / std

        std=0 の場合（全ベクトルが同一など）は z_score=0.0 として返す。

        Args:
            results:      SearchResult のリスト（類似度降順）。
            distribution: SimilarityEngine.get_distance_distribution() の返り値。

        Returns:
            List[Dict]: 各要素が以下のキーを持つ辞書リスト。
                - "word"        (str):   対象単語
                - "rank"        (int):   全体順位
                - "similarity"  (float): コサイン類似度
                - "pos_tag"     (str):   品詞ラベル
                - "pos_rank"    (int):   品詞内順位
                - "z_score"     (float): 分布上の Z-score
                - "explanation" (dict):  距離計算内訳

        Raises:
            KeyError: distribution に mean / std キーがない場合。
        """
        mean: float = distribution["mean"]
        std: float = distribution["std"]

        scored: List[Dict] = []
        for r in results:
            z_score: float = (
                (r.similarity - mean) / std if std > 0.0 else 0.0
            )
            scored.append({
                "word": r.word,
                "rank": r.rank,
                "similarity": r.similarity,
                "pos_tag": r.pos_tag,
                "pos_rank": r.pos_rank,
                "z_score": z_score,
                "explanation": r.explanation,
            })

        logger.debug(
            "attach_z_scores: %d 件に Z-score を付与 (mean=%.4f, std=%.4f)",
            len(scored), mean, std,
        )
        return scored

    @staticmethod
    def compare_distributions(
        query_word: str,
        static_dist: dict,
        contextual_dist: dict,
    ) -> DistributionComparison:
        """static と contextual の距離分布を比較する。

        両モデルの mean / std / z_score の差分を算出する。
        差分の解釈:
            mean_diff > 0  → static の方が全体的に類似度が高い傾向
            z_score_diff > 0 → static の方が Top-1 が分布から際立っている

        Args:
            query_word:  クエリ単語。
            static_dist: static エンジンの get_distance_distribution() 返り値。
            contextual_dist:  contextual エンジンの get_distance_distribution() 返り値。

        Returns:
            DistributionComparison: 両分布の比較結果。

        Raises:
            InsufficientDataError: どちらかの histogram_data が空の場合。
            KeyError:              必須キーが存在しない場合。
        """
        static_stats = Analyzer.enrich_distribution(static_dist)
        contextual_stats = Analyzer.enrich_distribution(contextual_dist)

        mean_diff: float = static_stats.mean - contextual_stats.mean
        std_diff: float = static_stats.std - contextual_stats.std
        z_score_diff: float = static_stats.z_score - contextual_stats.z_score

        logger.info(
            "compare_distributions: query=%s, "
            "mean_diff=%.4f, std_diff=%.4f, z_score_diff=%.4f",
            query_word, mean_diff, std_diff, z_score_diff,
        )

        return DistributionComparison(
            query_word=query_word,
            static_stats=static_stats,
            contextual_stats=contextual_stats,
            mean_diff=mean_diff,
            std_diff=std_diff,
            z_score_diff=z_score_diff,
        )

    @staticmethod
    def neighborhood_stability(
        static_results: List[SearchResult],
        contextual_results: List[SearchResult],
    ) -> float:
        """Top-K 結果の近傍安定性スコアを計算する。

        static と contextual の Top-K 結果の重複率（Jaccard 係数）を返す。
        値が高いほど、両モデルで一致する近傍を持つ（安定した意味空間）。

        計算式:
            stability = |static ∩ contextual| / |static ∪ contextual|

        Args:
            static_results: static エンジンの Top-K 検索結果。
            contextual_results:  contextual エンジンの Top-K 検索結果。

        Returns:
            float: Jaccard 係数。範囲: [0.0, 1.0]。
                   1.0 = 完全一致（両モデルが同じ近傍を返す）
                   0.0 = 完全不一致

        Raises:
            ValueError: どちらかの results が空の場合。
        """
        if not static_results:
            raise ValueError("static_results が空です")
        if not contextual_results:
            raise ValueError("contextual_results が空です")

        static_words: set[str] = {r.word for r in static_results}
        contextual_words: set[str] = {r.word for r in contextual_results}

        intersection = len(static_words & contextual_words)
        union = len(static_words | contextual_words)

        stability: float = intersection / union if union > 0 else 0.0

        logger.debug(
            "neighborhood_stability: intersection=%d, union=%d, score=%.4f",
            intersection, union, stability,
        )
        return stability
