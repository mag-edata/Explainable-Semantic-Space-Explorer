"""
test_analyzer.py
================
Analyzer の単体テスト。

テスト対象:
    - enrich_distribution():    分布統計に中央値・四分位を付与
    - histogram():              ビン集計
    - attach_z_scores():        SearchResult に Z-score を付与
    - compare_distributions():  static vs SBERT の分布比較
    - neighborhood_stability(): Top-K の Jaccard 係数

実行方法:
    venv/bin/python3 -m unittest tests/test_analyzer.py -v
"""

from __future__ import annotations

import unittest
from typing import List

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analyzer import (
    Analyzer,
    DistributionComparison,
    DistributionStats,
    HistogramData,
    InsufficientDataError,
)
from core.similarity_engine import SearchResult


# ---------------------------------------------------------------------------
# テスト用フィクスチャ
# ---------------------------------------------------------------------------

def _make_distribution(
    query_word: str = "king",
    histogram_data: List[float] | None = None,
    top1: float | None = None,
) -> dict:
    """get_distance_distribution() の戻り値を模した dict を生成する。"""
    if histogram_data is None:
        rng = np.random.default_rng(0)
        histogram_data = rng.uniform(-0.2, 0.6, size=99).astype(float).tolist()

    arr = np.array(histogram_data)
    if arr.size == 0:
        # 空データは異常系テスト用。統計値はダミー値を入れる。
        mean, std, top1_val, z = 0.0, 0.0, 0.0, 0.0
    else:
        mean = float(arr.mean())
        std = float(arr.std())
        top1_val = float(top1) if top1 is not None else float(arr.max())
        z = (top1_val - mean) / std if std > 0 else 0.0
    return {
        "query_word": query_word,
        "mean": mean,
        "std": std,
        "top1_similarity": top1_val,
        "z_score": z,
        "histogram_data": histogram_data,
    }


def _make_search_result(
    word: str,
    rank: int,
    similarity: float,
    pos_tag: str = "NOUN",
) -> SearchResult:
    return SearchResult(
        word=word,
        index=rank - 1,
        similarity=similarity,
        rank=rank,
        pos_tag=pos_tag,
        pos_rank=rank,
        explanation={"formula": "cos = dot/(‖a‖·‖b‖)"},
    )


# ---------------------------------------------------------------------------
# enrich_distribution()
# ---------------------------------------------------------------------------

class TestEnrichDistribution(unittest.TestCase):
    """Analyzer.enrich_distribution() のテスト。"""

    def test_returns_distribution_stats(self) -> None:
        """戻り値が DistributionStats であることを確認。"""
        dist = _make_distribution()
        stats = Analyzer.enrich_distribution(dist)
        self.assertIsInstance(stats, DistributionStats)

    def test_query_word_preserved(self) -> None:
        """query_word が引き継がれることを確認。"""
        dist = _make_distribution(query_word="bank")
        stats = Analyzer.enrich_distribution(dist)
        self.assertEqual(stats.query_word, "bank")

    def test_median_matches_numpy(self) -> None:
        """median が numpy.median と一致することを確認。"""
        data = [0.1, 0.3, 0.2, 0.5, 0.4]
        dist = _make_distribution(histogram_data=data)
        stats = Analyzer.enrich_distribution(dist)
        self.assertAlmostEqual(stats.median, float(np.median(data)), places=5)

    def test_q25_q75_ordered(self) -> None:
        """q25 <= median <= q75 の関係が成立することを確認。"""
        dist = _make_distribution()
        stats = Analyzer.enrich_distribution(dist)
        self.assertLessEqual(stats.q25, stats.median)
        self.assertLessEqual(stats.median, stats.q75)

    def test_n_samples_matches_data_length(self) -> None:
        """n_samples が histogram_data の長さと一致することを確認。"""
        data = [0.1] * 7
        dist = _make_distribution(histogram_data=data)
        stats = Analyzer.enrich_distribution(dist)
        self.assertEqual(stats.n_samples, 7)

    def test_missing_key_raises(self) -> None:
        """必須キーが欠けていると KeyError が送出される。"""
        dist = _make_distribution()
        del dist["mean"]
        with self.assertRaises(KeyError):
            Analyzer.enrich_distribution(dist)

    def test_empty_histogram_raises_insufficient(self) -> None:
        """histogram_data が空の場合に InsufficientDataError が送出される。"""
        dist = _make_distribution(histogram_data=[])
        with self.assertRaises(InsufficientDataError):
            Analyzer.enrich_distribution(dist)


# ---------------------------------------------------------------------------
# histogram()
# ---------------------------------------------------------------------------

class TestHistogram(unittest.TestCase):
    """Analyzer.histogram() のテスト。"""

    def test_returns_histogram_data(self) -> None:
        """戻り値が HistogramData であることを確認。"""
        rng = np.random.default_rng(0)
        data = rng.uniform(0.0, 1.0, size=200).tolist()
        hist = Analyzer.histogram(data, n_bins=10)
        self.assertIsInstance(hist, HistogramData)

    def test_bin_edges_length(self) -> None:
        """bin_edges の長さが n_bins + 1 であることを確認。"""
        data = list(np.linspace(0.0, 1.0, 100))
        hist = Analyzer.histogram(data, n_bins=20)
        self.assertEqual(len(hist.bin_edges), 21)

    def test_counts_length(self) -> None:
        """counts の長さが n_bins と一致することを確認。"""
        data = list(np.linspace(0.0, 1.0, 100))
        hist = Analyzer.histogram(data, n_bins=20)
        self.assertEqual(len(hist.counts), 20)

    def test_counts_sum_equals_data_size(self) -> None:
        """counts の合計が data の件数と一致することを確認。"""
        rng = np.random.default_rng(7)
        data = rng.uniform(-1.0, 1.0, size=500).tolist()
        hist = Analyzer.histogram(data, n_bins=30)
        self.assertEqual(sum(hist.counts), len(data))

    def test_data_min_max(self) -> None:
        """data_min / data_max が data の最小・最大と一致することを確認。"""
        data = [0.1, -0.2, 0.5, 0.3, -0.4, 0.9]
        hist = Analyzer.histogram(data)
        self.assertAlmostEqual(hist.data_min, min(data), places=5)
        self.assertAlmostEqual(hist.data_max, max(data), places=5)

    def test_empty_data_raises(self) -> None:
        """data が空の場合に InsufficientDataError が送出される。"""
        with self.assertRaises(InsufficientDataError):
            Analyzer.histogram([])

    def test_invalid_n_bins(self) -> None:
        """n_bins が 0 以下の場合に ValueError が送出される。"""
        with self.assertRaises(ValueError):
            Analyzer.histogram([0.1, 0.2, 0.3], n_bins=0)


# ---------------------------------------------------------------------------
# attach_z_scores()
# ---------------------------------------------------------------------------

class TestAttachZScores(unittest.TestCase):
    """Analyzer.attach_z_scores() のテスト。"""

    def setUp(self) -> None:
        self.results: List[SearchResult] = [
            _make_search_result("queen",  1, 0.85),
            _make_search_result("prince", 2, 0.75),
            _make_search_result("man",    3, 0.65),
        ]
        # mean=0.0, std=1.0 になるよう設計したヒストデータ
        self.distribution = _make_distribution(
            histogram_data=[-1.0, 0.0, 1.0],
            top1=0.85,
        )

    def test_returns_list_of_dicts(self) -> None:
        """戻り値が dict のリストであることを確認。"""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        self.assertIsInstance(scored, list)
        for item in scored:
            self.assertIsInstance(item, dict)

    def test_returned_length_matches_results(self) -> None:
        """戻り値の件数が results と一致することを確認。"""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        self.assertEqual(len(scored), len(self.results))

    def test_required_keys(self) -> None:
        """各要素に必須キーが含まれることを確認。"""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        for item in scored:
            for key in ("word", "rank", "similarity", "pos_tag",
                        "pos_rank", "z_score", "explanation"):
                self.assertIn(key, item)

    def test_z_score_formula(self) -> None:
        """z_score が (similarity - mean) / std と一致することを確認。"""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        mean = self.distribution["mean"]
        std = self.distribution["std"]
        for item in scored:
            expected = (item["similarity"] - mean) / std
            self.assertAlmostEqual(item["z_score"], expected, places=5)

    def test_z_score_when_std_is_zero(self) -> None:
        """std=0 のとき z_score=0.0 が安全に返ることを確認（ゼロ除算ガード）。"""
        flat_dist = {
            "query_word": "x",
            "mean": 0.5,
            "std": 0.0,
            "top1_similarity": 0.5,
            "z_score": 0.0,
            "histogram_data": [0.5, 0.5, 0.5],
        }
        scored = Analyzer.attach_z_scores(self.results, flat_dist)
        for item in scored:
            self.assertEqual(item["z_score"], 0.0)


# ---------------------------------------------------------------------------
# compare_distributions()
# ---------------------------------------------------------------------------

class TestCompareDistributions(unittest.TestCase):
    """Analyzer.compare_distributions() のテスト。"""

    def setUp(self) -> None:
        self.static_dist = _make_distribution(
            query_word="king",
            histogram_data=[0.0, 0.1, 0.2, 0.3, 0.4],
            top1=0.85,
        )
        self.sbert_dist = _make_distribution(
            query_word="king",
            histogram_data=[0.1, 0.2, 0.3, 0.4, 0.5],
            top1=0.70,
        )

    def test_returns_distribution_comparison(self) -> None:
        """戻り値が DistributionComparison であることを確認。"""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.sbert_dist,
        )
        self.assertIsInstance(cmp_, DistributionComparison)

    def test_query_word_preserved(self) -> None:
        """query_word が引き継がれることを確認。"""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.sbert_dist,
        )
        self.assertEqual(cmp_.query_word, "king")

    def test_mean_diff_value(self) -> None:
        """mean_diff = static.mean - sbert.mean であることを確認。"""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.sbert_dist,
        )
        expected = self.static_dist["mean"] - self.sbert_dist["mean"]
        self.assertAlmostEqual(cmp_.mean_diff, expected, places=5)

    def test_std_diff_value(self) -> None:
        """std_diff = static.std - sbert.std であることを確認。"""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.sbert_dist,
        )
        expected = self.static_dist["std"] - self.sbert_dist["std"]
        self.assertAlmostEqual(cmp_.std_diff, expected, places=5)

    def test_z_score_diff_value(self) -> None:
        """z_score_diff = static.z_score - sbert.z_score であることを確認。"""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.sbert_dist,
        )
        expected = self.static_dist["z_score"] - self.sbert_dist["z_score"]
        self.assertAlmostEqual(cmp_.z_score_diff, expected, places=5)

    def test_static_and_sbert_stats_attached(self) -> None:
        """static_stats / sbert_stats が DistributionStats として格納されることを確認。"""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.sbert_dist,
        )
        self.assertIsInstance(cmp_.static_stats, DistributionStats)
        self.assertIsInstance(cmp_.sbert_stats, DistributionStats)


# ---------------------------------------------------------------------------
# neighborhood_stability()
# ---------------------------------------------------------------------------

class TestNeighborhoodStability(unittest.TestCase):
    """Analyzer.neighborhood_stability() のテスト。"""

    def test_full_overlap_is_one(self) -> None:
        """両モデルの結果が完全一致なら Jaccard = 1.0。"""
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["x", "y", "z"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["x", "y", "z"])]
        self.assertAlmostEqual(Analyzer.neighborhood_stability(a, b), 1.0, places=5)

    def test_no_overlap_is_zero(self) -> None:
        """両モデルの結果が完全に異なるなら Jaccard = 0.0。"""
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["x", "y", "z"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["p", "q", "r"])]
        self.assertAlmostEqual(Analyzer.neighborhood_stability(a, b), 0.0, places=5)

    def test_partial_overlap_value(self) -> None:
        """部分一致時に Jaccard 係数が正しく算出される。"""
        # static = {a, b, c}, sbert = {b, c, d} → 共通=2, 和=4 → 0.5
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["a", "b", "c"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["b", "c", "d"])]
        self.assertAlmostEqual(Analyzer.neighborhood_stability(a, b), 2 / 4, places=5)

    def test_range_within_zero_one(self) -> None:
        """結果が [0.0, 1.0] の範囲に収まることを確認。"""
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["a", "b", "c"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["a", "x", "y"])]
        score = Analyzer.neighborhood_stability(a, b)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_static_raises(self) -> None:
        """static_results が空の場合に ValueError が送出される。"""
        b = [_make_search_result("x", 1, 0.5)]
        with self.assertRaises(ValueError):
            Analyzer.neighborhood_stability([], b)

    def test_empty_sbert_raises(self) -> None:
        """sbert_results が空の場合に ValueError が送出される。"""
        a = [_make_search_result("x", 1, 0.5)]
        with self.assertRaises(ValueError):
            Analyzer.neighborhood_stability(a, [])


if __name__ == "__main__":
    unittest.main()
