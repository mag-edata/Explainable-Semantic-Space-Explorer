"""
test_similarity_engine.py
=========================
Unit tests for ``SimilarityEngine``.

Targets under test:
    - ``search()``: Top-K similar-word search
    - ``compare()``: Comparison between two engines
    - ``get_distance_distribution()``: Distance distribution computation
    - ``word_to_index()``: Inverse lookup of vocabulary indices

How to run:
    venv/bin/python3 -m unittest tests/test_similarity_engine.py -v
"""

from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.distance_metrics import DistanceMetrics
from core.similarity_engine import (
    ComparisonResult,
    SearchResult,
    SimilarityEngine,
    SimilarityEngineError,
    InvalidTopKError,
    UnknownWordError,
)


# ---------------------------------------------------------------------------
# Test fixtures (small mock data)
# ---------------------------------------------------------------------------

def _make_engine(
    n: int = 6,
    dim: int = 4,
    seed: int = 42,
) -> tuple[SimilarityEngine, dict[str, int], np.ndarray]:
    """Build a small ``SimilarityEngine`` for tests.

    Args:
        n:    Vocabulary size.
        dim:  Vector dimensionality.
        seed: Random seed.

    Returns:
        Tuple of ``(engine, vocab, vectors)``.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vocab = {f"word{i}": i for i in range(n)}
    pos_tags = np.array(["NOUN", "VERB", "NOUN", "ADJ", "VERB", "NOUN"][:n])
    metrics = DistanceMetrics()

    engine = SimilarityEngine(
        vectors=vectors,
        vocab=vocab,
        pos_tags=pos_tags,
        metrics=metrics,
    )
    return engine, vocab, vectors


class TestSearch(unittest.TestCase):
    """Tests for ``SimilarityEngine.search()``."""

    def setUp(self) -> None:
        self.engine, self.vocab, self.vectors = _make_engine(n=6, dim=4)

    def test_returns_list_of_search_results(self) -> None:
        """The return value is a list of ``SearchResult``."""
        results = self.engine.search("word0", top_k=3)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, SearchResult) for r in results))

    def test_top_k_count(self) -> None:
        """The number of returned items is at most ``top_k`` (query itself excluded)."""
        results = self.engine.search("word0", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_excludes_query_itself(self) -> None:
        """Results do not contain the query word itself."""
        results = self.engine.search("word0", top_k=5)
        words = [r.word for r in results]
        self.assertNotIn("word0", words)

    def test_sorted_by_similarity_descending(self) -> None:
        """Results are sorted by similarity in descending order."""
        results = self.engine.search("word0", top_k=4)
        sims = [r.similarity for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_rank_starts_at_one(self) -> None:
        """Rank values start at 1."""
        results = self.engine.search("word0", top_k=3)
        self.assertEqual(results[0].rank, 1)

    def test_result_has_explanation(self) -> None:
        """``SearchResult`` carries the ``explanation`` field."""
        results = self.engine.search("word0", top_k=1)
        self.assertIn("formula", results[0].explanation)
        self.assertIn("dot_product", results[0].explanation)

    def test_pos_filter(self) -> None:
        """When ``pos_filter`` is supplied, only that POS is returned."""
        results = self.engine.search("word0", top_k=5, pos_filter="NOUN")
        for r in results:
            self.assertEqual(r.pos_tag, "NOUN")

    def test_unknown_word_error(self) -> None:
        """An out-of-vocabulary word raises ``UnknownWordError``."""
        with self.assertRaises(UnknownWordError):
            self.engine.search("nonexistent_word", top_k=3)

    def test_invalid_top_k_zero(self) -> None:
        """``top_k=0`` raises ``InvalidTopKError``."""
        with self.assertRaises(InvalidTopKError):
            self.engine.search("word0", top_k=0)

    def test_invalid_top_k_negative(self) -> None:
        """Negative ``top_k`` raises ``InvalidTopKError``."""
        with self.assertRaises(InvalidTopKError):
            self.engine.search("word0", top_k=-1)


class TestCompare(unittest.TestCase):
    """Tests for ``SimilarityEngine.compare()``."""

    def setUp(self) -> None:
        self.engine_a, _, _ = _make_engine(n=6, dim=4, seed=0)
        self.engine_b, _, _ = _make_engine(n=6, dim=4, seed=1)

    def test_returns_comparison_result(self) -> None:
        """The return value is a ``ComparisonResult``."""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=3)
        self.assertIsInstance(result, ComparisonResult)

    def test_query_word_field(self) -> None:
        """The ``query_word`` field is correct."""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=3)
        self.assertEqual(result.query_word, "word0")

    def test_static_and_contextual_results_lengths(self) -> None:
        """Both ``static_results`` and ``contextual_results`` are at most ``top_k`` entries long."""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=3)
        self.assertLessEqual(len(result.static_results), 3)
        self.assertLessEqual(len(result.contextual_results), 3)

    def test_common_words_subset(self) -> None:
        """``common_words`` equals the intersection of words from both engines."""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=4)
        static_words = {r.word for r in result.static_results}
        contextual_words  = {r.word for r in result.contextual_results}
        expected_common = static_words & contextual_words
        self.assertEqual(set(result.common_words), expected_common)

    def test_rank_diff_keys_are_common_words(self) -> None:
        """``rank_diff`` keys match the set of common words."""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=4)
        self.assertEqual(set(result.rank_diff.keys()), set(result.common_words))

    def test_unknown_word_error(self) -> None:
        """An out-of-vocabulary word raises ``UnknownWordError``."""
        with self.assertRaises(UnknownWordError):
            self.engine_a.compare("no_such_word", other=self.engine_b, top_k=3)


class TestGetDistanceDistribution(unittest.TestCase):
    """Tests for ``SimilarityEngine.get_distance_distribution()``."""

    def setUp(self) -> None:
        self.engine, _, _ = _make_engine(n=10, dim=8)

    def test_returns_dict(self) -> None:
        """The return value is a dictionary."""
        result = self.engine.get_distance_distribution("word0")
        self.assertIsInstance(result, dict)

    def test_required_keys(self) -> None:
        """Every required key is present."""
        result = self.engine.get_distance_distribution("word0")
        for key in ("query_word", "mean", "std", "top1_similarity", "z_score", "histogram_data"):
            self.assertIn(key, result, msg=f"キー '{key}' が存在しません")

    def test_query_word_field(self) -> None:
        """``query_word`` matches the input query."""
        result = self.engine.get_distance_distribution("word0")
        self.assertEqual(result["query_word"], "word0")

    def test_top1_ge_mean(self) -> None:
        """``top1_similarity`` is greater than or equal to the mean similarity."""
        result = self.engine.get_distance_distribution("word0")
        self.assertGreaterEqual(result["top1_similarity"], result["mean"] - 1e-6)

    def test_histogram_data_length(self) -> None:
        """``histogram_data`` length is N - 1 (query itself excluded)."""
        n = 10
        result = self.engine.get_distance_distribution("word0")
        self.assertEqual(len(result["histogram_data"]), n - 1)

    def test_std_is_nonnegative(self) -> None:
        """Standard deviation is non-negative."""
        result = self.engine.get_distance_distribution("word0")
        self.assertGreaterEqual(result["std"], 0.0)

    def test_unknown_word_error(self) -> None:
        """An out-of-vocabulary word raises ``UnknownWordError``."""
        with self.assertRaises(UnknownWordError):
            self.engine.get_distance_distribution("no_such_word")


class TestWordToIndex(unittest.TestCase):
    """Tests for ``SimilarityEngine.word_to_index()``."""

    def setUp(self) -> None:
        self.engine, self.vocab, _ = _make_engine(n=6, dim=4)

    def test_correct_index(self) -> None:
        """Each word maps to its expected index."""
        for word, expected_idx in self.vocab.items():
            self.assertEqual(self.engine.word_to_index(word), expected_idx)

    def test_unknown_word_error(self) -> None:
        """An out-of-vocabulary word raises ``UnknownWordError``."""
        with self.assertRaises(UnknownWordError):
            self.engine.word_to_index("unknown_word_xyz")


if __name__ == "__main__":
    unittest.main()
