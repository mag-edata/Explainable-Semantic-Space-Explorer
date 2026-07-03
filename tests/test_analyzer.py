"""
test_analyzer.py
================
Unit tests for ``Analyzer``.

Targets under test:
    - ``enrich_distribution()``:    Augment a distribution with median / quartiles
    - ``histogram()``:              Bin aggregation
    - ``attach_z_scores()``:        Attach Z-scores to ``SearchResult`` entries
    - ``compare_distributions()``:  Compare the static vs. contextual distributions
    - ``neighborhood_stability()``: Jaccard coefficient for the Top-K neighborhood

How to run:
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
    SimilarityVerdict,
)
from core.similarity_engine import SearchResult


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_distribution(
    query_word: str = "king",
    histogram_data: List[float] | None = None,
    top1: float | None = None,
) -> dict:
    """Build a dictionary that mimics the return value of ``get_distance_distribution()``."""
    if histogram_data is None:
        rng = np.random.default_rng(0)
        histogram_data = rng.uniform(-0.2, 0.6, size=99).astype(float).tolist()

    arr = np.array(histogram_data)
    if arr.size == 0:
        # Empty data is for the error-path tests; fill stats with dummy values.
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
    """Tests for ``Analyzer.enrich_distribution()``."""

    def test_returns_distribution_stats(self) -> None:
        """The return value is a ``DistributionStats``."""
        dist = _make_distribution()
        stats = Analyzer.enrich_distribution(dist)
        self.assertIsInstance(stats, DistributionStats)

    def test_query_word_preserved(self) -> None:
        """``query_word`` is carried over."""
        dist = _make_distribution(query_word="bank")
        stats = Analyzer.enrich_distribution(dist)
        self.assertEqual(stats.query_word, "bank")

    def test_median_matches_numpy(self) -> None:
        """``median`` matches ``numpy.median``."""
        data = [0.1, 0.3, 0.2, 0.5, 0.4]
        dist = _make_distribution(histogram_data=data)
        stats = Analyzer.enrich_distribution(dist)
        self.assertAlmostEqual(stats.median, float(np.median(data)), places=5)

    def test_q25_q75_ordered(self) -> None:
        """The ordering ``q25 <= median <= q75`` holds."""
        dist = _make_distribution()
        stats = Analyzer.enrich_distribution(dist)
        self.assertLessEqual(stats.q25, stats.median)
        self.assertLessEqual(stats.median, stats.q75)

    def test_n_samples_matches_data_length(self) -> None:
        """``n_samples`` matches the length of ``histogram_data``."""
        data = [0.1] * 7
        dist = _make_distribution(histogram_data=data)
        stats = Analyzer.enrich_distribution(dist)
        self.assertEqual(stats.n_samples, 7)

    def test_missing_key_raises(self) -> None:
        """A missing required key raises ``KeyError``."""
        dist = _make_distribution()
        del dist["mean"]
        with self.assertRaises(KeyError):
            Analyzer.enrich_distribution(dist)

    def test_empty_histogram_raises_insufficient(self) -> None:
        """Empty ``histogram_data`` raises ``InsufficientDataError``."""
        dist = _make_distribution(histogram_data=[])
        with self.assertRaises(InsufficientDataError):
            Analyzer.enrich_distribution(dist)


# ---------------------------------------------------------------------------
# histogram()
# ---------------------------------------------------------------------------

class TestHistogram(unittest.TestCase):
    """Tests for ``Analyzer.histogram()``."""

    def test_returns_histogram_data(self) -> None:
        """The return value is a ``HistogramData``."""
        rng = np.random.default_rng(0)
        data = rng.uniform(0.0, 1.0, size=200).tolist()
        hist = Analyzer.histogram(data, n_bins=10)
        self.assertIsInstance(hist, HistogramData)

    def test_bin_edges_length(self) -> None:
        """``bin_edges`` length is ``n_bins + 1``."""
        data = list(np.linspace(0.0, 1.0, 100))
        hist = Analyzer.histogram(data, n_bins=20)
        self.assertEqual(len(hist.bin_edges), 21)

    def test_counts_length(self) -> None:
        """``counts`` length matches ``n_bins``."""
        data = list(np.linspace(0.0, 1.0, 100))
        hist = Analyzer.histogram(data, n_bins=20)
        self.assertEqual(len(hist.counts), 20)

    def test_counts_sum_equals_data_size(self) -> None:
        """The sum of ``counts`` matches the size of ``data``."""
        rng = np.random.default_rng(7)
        data = rng.uniform(-1.0, 1.0, size=500).tolist()
        hist = Analyzer.histogram(data, n_bins=30)
        self.assertEqual(sum(hist.counts), len(data))

    def test_data_min_max(self) -> None:
        """``data_min`` / ``data_max`` match the min / max of ``data``."""
        data = [0.1, -0.2, 0.5, 0.3, -0.4, 0.9]
        hist = Analyzer.histogram(data)
        self.assertAlmostEqual(hist.data_min, min(data), places=5)
        self.assertAlmostEqual(hist.data_max, max(data), places=5)

    def test_empty_data_raises(self) -> None:
        """Empty ``data`` raises ``InsufficientDataError``."""
        with self.assertRaises(InsufficientDataError):
            Analyzer.histogram([])

    def test_invalid_n_bins(self) -> None:
        """Non-positive ``n_bins`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            Analyzer.histogram([0.1, 0.2, 0.3], n_bins=0)


# ---------------------------------------------------------------------------
# attach_z_scores()
# ---------------------------------------------------------------------------

class TestAttachZScores(unittest.TestCase):
    """Tests for ``Analyzer.attach_z_scores()``."""

    def setUp(self) -> None:
        self.results: List[SearchResult] = [
            _make_search_result("queen",  1, 0.85),
            _make_search_result("prince", 2, 0.75),
            _make_search_result("man",    3, 0.65),
        ]
        # Histogram designed so mean=0.0 and std=1.0
        self.distribution = _make_distribution(
            histogram_data=[-1.0, 0.0, 1.0],
            top1=0.85,
        )

    def test_returns_list_of_dicts(self) -> None:
        """The return value is a list of dictionaries."""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        self.assertIsInstance(scored, list)
        for item in scored:
            self.assertIsInstance(item, dict)

    def test_returned_length_matches_results(self) -> None:
        """The number of returned items matches ``results``."""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        self.assertEqual(len(scored), len(self.results))

    def test_required_keys(self) -> None:
        """Each item contains every required key."""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        for item in scored:
            for key in ("word", "rank", "similarity", "pos_tag",
                        "pos_rank", "z_score", "explanation"):
                self.assertIn(key, item)

    def test_z_score_formula(self) -> None:
        """``z_score`` equals ``(similarity - mean) / std``."""
        scored = Analyzer.attach_z_scores(self.results, self.distribution)
        mean = self.distribution["mean"]
        std = self.distribution["std"]
        for item in scored:
            expected = (item["similarity"] - mean) / std
            self.assertAlmostEqual(item["z_score"], expected, places=5)

    def test_z_score_when_std_is_zero(self) -> None:
        """When ``std == 0`` the Z-score is safely returned as 0.0 (zero-division guard)."""
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
    """Tests for ``Analyzer.compare_distributions()``."""

    def setUp(self) -> None:
        self.static_dist = _make_distribution(
            query_word="king",
            histogram_data=[0.0, 0.1, 0.2, 0.3, 0.4],
            top1=0.85,
        )
        self.contextual_dist = _make_distribution(
            query_word="king",
            histogram_data=[0.1, 0.2, 0.3, 0.4, 0.5],
            top1=0.70,
        )

    def test_returns_distribution_comparison(self) -> None:
        """The return value is a ``DistributionComparison``."""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.contextual_dist,
        )
        self.assertIsInstance(cmp_, DistributionComparison)

    def test_query_word_preserved(self) -> None:
        """``query_word`` is carried over."""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.contextual_dist,
        )
        self.assertEqual(cmp_.query_word, "king")

    def test_mean_diff_value(self) -> None:
        """``mean_diff`` equals ``static.mean - contextual.mean``."""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.contextual_dist,
        )
        expected = self.static_dist["mean"] - self.contextual_dist["mean"]
        self.assertAlmostEqual(cmp_.mean_diff, expected, places=5)

    def test_std_diff_value(self) -> None:
        """``std_diff`` equals ``static.std - contextual.std``."""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.contextual_dist,
        )
        expected = self.static_dist["std"] - self.contextual_dist["std"]
        self.assertAlmostEqual(cmp_.std_diff, expected, places=5)

    def test_z_score_diff_value(self) -> None:
        """``z_score_diff`` equals ``static.z_score - contextual.z_score``."""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.contextual_dist,
        )
        expected = self.static_dist["z_score"] - self.contextual_dist["z_score"]
        self.assertAlmostEqual(cmp_.z_score_diff, expected, places=5)

    def test_static_and_contextual_stats_attached(self) -> None:
        """``static_stats`` and ``contextual_stats`` are stored as ``DistributionStats``."""
        cmp_ = Analyzer.compare_distributions(
            "king", self.static_dist, self.contextual_dist,
        )
        self.assertIsInstance(cmp_.static_stats, DistributionStats)
        self.assertIsInstance(cmp_.contextual_stats, DistributionStats)


# ---------------------------------------------------------------------------
# neighborhood_stability()
# ---------------------------------------------------------------------------

class TestNeighborhoodStability(unittest.TestCase):
    """Tests for ``Analyzer.neighborhood_stability()``."""

    def test_full_overlap_is_one(self) -> None:
        """When both models agree completely, the Jaccard coefficient is 1.0."""
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["x", "y", "z"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["x", "y", "z"])]
        self.assertAlmostEqual(Analyzer.neighborhood_stability(a, b), 1.0, places=5)

    def test_no_overlap_is_zero(self) -> None:
        """When both models disagree completely, the Jaccard coefficient is 0.0."""
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["x", "y", "z"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["p", "q", "r"])]
        self.assertAlmostEqual(Analyzer.neighborhood_stability(a, b), 0.0, places=5)

    def test_partial_overlap_value(self) -> None:
        """Jaccard coefficient is correct under partial overlap."""
        # static = {a, b, c}, contextual = {b, c, d} → intersection=2, union=4 → 0.5
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["a", "b", "c"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["b", "c", "d"])]
        self.assertAlmostEqual(Analyzer.neighborhood_stability(a, b), 2 / 4, places=5)

    def test_range_within_zero_one(self) -> None:
        """The result stays in [0.0, 1.0]."""
        a = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["a", "b", "c"])]
        b = [_make_search_result(w, i + 1, 0.5) for i, w in enumerate(["a", "x", "y"])]
        score = Analyzer.neighborhood_stability(a, b)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_static_raises(self) -> None:
        """Empty ``static_results`` raises ``ValueError``."""
        b = [_make_search_result("x", 1, 0.5)]
        with self.assertRaises(ValueError):
            Analyzer.neighborhood_stability([], b)

    def test_empty_contextual_raises(self) -> None:
        """Empty ``contextual_results`` raises ``ValueError``."""
        a = [_make_search_result("x", 1, 0.5)]
        with self.assertRaises(ValueError):
            Analyzer.neighborhood_stability(a, [])


# ---------------------------------------------------------------------------
# interpret_similarity()
# ---------------------------------------------------------------------------

class TestInterpretSimilarity(unittest.TestCase):
    """Tests for ``Analyzer.interpret_similarity()`` (FR-25 verdicts)."""

    def setUp(self) -> None:
        # 1000 evenly spaced similarities in [0.000, 0.999].
        self.data = [i / 1000 for i in range(1000)]
        self.dist = _make_distribution(histogram_data=self.data)

    def test_returns_similarity_verdict(self) -> None:
        """The return value is a ``SimilarityVerdict``."""
        verdict = Analyzer.interpret_similarity(0.5, self.dist)
        self.assertIsInstance(verdict, SimilarityVerdict)

    def test_top_outlier_is_unusually_close(self) -> None:
        """A value only ~0.1% of the vocabulary reaches is 'unusually close'."""
        verdict = Analyzer.interpret_similarity(0.999, self.dist)
        self.assertLessEqual(verdict.top_fraction, 0.001)
        self.assertEqual(verdict.label, "unusually close")

    def test_very_close_tier(self) -> None:
        """~0.5% of the vocabulary at least this similar → 'very close'."""
        verdict = Analyzer.interpret_similarity(0.995, self.dist)
        self.assertEqual(verdict.label, "very close")

    def test_mid_value_is_around_average(self) -> None:
        """The median value lands in the 'around the vocabulary average' tier."""
        verdict = Analyzer.interpret_similarity(0.5, self.dist)
        self.assertAlmostEqual(verdict.top_fraction, 0.5, places=3)
        self.assertEqual(verdict.label, "around the vocabulary average")

    def test_z_score_formula(self) -> None:
        """``z_score`` equals ``(similarity - mean) / std``."""
        verdict = Analyzer.interpret_similarity(0.8, self.dist)
        expected = (0.8 - self.dist["mean"]) / self.dist["std"]
        self.assertAlmostEqual(verdict.z_score, expected, places=5)

    def test_text_mentions_count_and_label(self) -> None:
        """The display text includes the vocabulary size and the tier label."""
        verdict = Analyzer.interpret_similarity(0.999, self.dist)
        self.assertIn("1,000 words", verdict.text)
        self.assertIn(verdict.label, verdict.text)

    def test_missing_key_raises(self) -> None:
        """A missing required key raises ``KeyError``."""
        dist = _make_distribution(histogram_data=self.data)
        del dist["std"]
        with self.assertRaises(KeyError):
            Analyzer.interpret_similarity(0.5, dist)

    def test_empty_histogram_raises(self) -> None:
        """Empty ``histogram_data`` raises ``InsufficientDataError``."""
        dist = _make_distribution(histogram_data=[])
        with self.assertRaises(InsufficientDataError):
            Analyzer.interpret_similarity(0.5, dist)

    def test_zero_std_gives_zero_z(self) -> None:
        """When ``std == 0`` the Z-score is safely 0.0 (zero-division guard)."""
        dist = {"mean": 0.5, "std": 0.0, "histogram_data": [0.5, 0.5, 0.5]}
        verdict = Analyzer.interpret_similarity(0.5, dist)
        self.assertEqual(verdict.z_score, 0.0)


if __name__ == "__main__":
    unittest.main()
