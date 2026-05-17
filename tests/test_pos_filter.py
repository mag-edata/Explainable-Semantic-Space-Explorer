"""
test_pos_filter.py
==================
Unit tests for ``POSFilter``.

Targets under test:
    - ``filter()``:             Filter by a specified POS
    - ``group_by_pos()``:       Group results by POS
    - ``pos_distribution()``:   Aggregate per-POS occurrence counts
    - ``heterogeneity_rate()``: Heterogeneity rate
    - ``assign_pos_ranks()``:   Assign within-POS rank
    - ``top_pos()``:            List of POS tags by frequency

How to run:
    venv/bin/python3 -m unittest tests/test_pos_filter.py -v
"""

from __future__ import annotations

import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pos_filter import POSFilter, UnknownPOSTagError
from core.similarity_engine import SearchResult


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_result(
    word: str,
    rank: int,
    pos_tag: str,
    similarity: float = 0.5,
) -> SearchResult:
    """Helper that builds a ``SearchResult`` easily for tests.

    ``pos_rank`` is initialized to 0 since it assumes ``assign_pos_ranks()``.
    ``explanation`` is an empty dict (``POSFilter`` does not inspect it).
    """
    return SearchResult(
        word=word,
        index=rank - 1,
        similarity=similarity,
        rank=rank,
        pos_tag=pos_tag,
        pos_rank=0,
        explanation={},
    )


def _sample_results() -> list[SearchResult]:
    """6 SearchResults in descending similarity order (NOUN×3 / VERB×2 / ADJ×1)."""
    return [
        _make_result("apple",  1, "NOUN", similarity=0.95),
        _make_result("run",    2, "VERB", similarity=0.90),
        _make_result("banana", 3, "NOUN", similarity=0.85),
        _make_result("walk",   4, "VERB", similarity=0.80),
        _make_result("cherry", 5, "NOUN", similarity=0.75),
        _make_result("blue",   6, "ADJ",  similarity=0.70),
    ]


# ---------------------------------------------------------------------------
# filter()
# ---------------------------------------------------------------------------

class TestFilter(unittest.TestCase):
    """Tests for ``POSFilter.filter()``."""

    def setUp(self) -> None:
        self.results = _sample_results()

    def test_filter_returns_only_specified_pos(self) -> None:
        """Only the requested POS is returned."""
        filtered = POSFilter.filter(self.results, "NOUN")
        self.assertEqual(len(filtered), 3)
        for r in filtered:
            self.assertEqual(r.pos_tag, "NOUN")

    def test_filter_preserves_original_order(self) -> None:
        """Filtered order matches the original ``results`` order."""
        filtered = POSFilter.filter(self.results, "NOUN")
        words = [r.word for r in filtered]
        self.assertEqual(words, ["apple", "banana", "cherry"])

    def test_filter_preserves_original_rank(self) -> None:
        """Original ``rank`` (overall position) is preserved after filtering."""
        filtered = POSFilter.filter(self.results, "VERB")
        ranks = [r.rank for r in filtered]
        self.assertEqual(ranks, [2, 4])  # original ranks preserved

    def test_filter_empty_pos_tag_raises_value_error(self) -> None:
        """An empty ``pos_tag`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            POSFilter.filter(self.results, "")

    def test_filter_unknown_pos_raises(self) -> None:
        """A POS that does not appear in ``results`` raises ``UnknownPOSTagError``."""
        with self.assertRaises(UnknownPOSTagError):
            POSFilter.filter(self.results, "ADV")


# ---------------------------------------------------------------------------
# group_by_pos()
# ---------------------------------------------------------------------------

class TestGroupByPos(unittest.TestCase):
    """Tests for ``POSFilter.group_by_pos()``."""

    def setUp(self) -> None:
        self.results = _sample_results()

    def test_returns_dict(self) -> None:
        """The return value is a dictionary."""
        groups = POSFilter.group_by_pos(self.results)
        self.assertIsInstance(groups, dict)

    def test_keys_are_unique_pos_tags(self) -> None:
        """Keys match the set of POS tags appearing in ``results``."""
        groups = POSFilter.group_by_pos(self.results)
        self.assertEqual(set(groups.keys()), {"NOUN", "VERB", "ADJ"})

    def test_each_group_size(self) -> None:
        """Each group has the correct count."""
        groups = POSFilter.group_by_pos(self.results)
        self.assertEqual(len(groups["NOUN"]), 3)
        self.assertEqual(len(groups["VERB"]), 2)
        self.assertEqual(len(groups["ADJ"]),  1)

    def test_group_preserves_order(self) -> None:
        """Order within each group is preserved from the original ``results``."""
        groups = POSFilter.group_by_pos(self.results)
        noun_words = [r.word for r in groups["NOUN"]]
        self.assertEqual(noun_words, ["apple", "banana", "cherry"])

    def test_empty_results(self) -> None:
        """Empty ``results`` returns an empty dict."""
        groups = POSFilter.group_by_pos([])
        self.assertEqual(groups, {})


# ---------------------------------------------------------------------------
# pos_distribution()
# ---------------------------------------------------------------------------

class TestPosDistribution(unittest.TestCase):
    """Tests for ``POSFilter.pos_distribution()``."""

    def setUp(self) -> None:
        self.results = _sample_results()

    def test_counts_are_correct(self) -> None:
        """Per-POS counts are aggregated correctly."""
        dist = POSFilter.pos_distribution(self.results)
        self.assertEqual(dist["NOUN"], 3)
        self.assertEqual(dist["VERB"], 2)
        self.assertEqual(dist["ADJ"],  1)

    def test_sorted_descending(self) -> None:
        """Counts are sorted in descending order."""
        dist = POSFilter.pos_distribution(self.results)
        counts = list(dist.values())
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_sum_equals_total(self) -> None:
        """Sum of counts equals the length of ``results``."""
        dist = POSFilter.pos_distribution(self.results)
        self.assertEqual(sum(dist.values()), len(self.results))

    def test_empty_results(self) -> None:
        """Empty ``results`` returns an empty dict."""
        self.assertEqual(POSFilter.pos_distribution([]), {})


# ---------------------------------------------------------------------------
# heterogeneity_rate()
# ---------------------------------------------------------------------------

class TestHeterogeneityRate(unittest.TestCase):
    """Tests for ``POSFilter.heterogeneity_rate()``."""

    def setUp(self) -> None:
        self.results = _sample_results()  # NOUN×3, VERB×2, ADJ×1 = 6 items

    def test_rate_with_query_pos_noun(self) -> None:
        """For ``query_pos=NOUN`` the heterogeneity rate is 3/6 = 0.5."""
        rate = POSFilter.heterogeneity_rate(self.results, "NOUN")
        self.assertAlmostEqual(rate, 0.5, places=5)

    def test_rate_with_query_pos_verb(self) -> None:
        """For ``query_pos=VERB`` the heterogeneity rate is 4/6 ≈ 0.6667."""
        rate = POSFilter.heterogeneity_rate(self.results, "VERB")
        self.assertAlmostEqual(rate, 4 / 6, places=5)

    def test_rate_all_same_pos_is_zero(self) -> None:
        """If every item matches ``query_pos``, the rate is 0.0."""
        homog = [_make_result(f"w{i}", i + 1, "NOUN") for i in range(5)]
        rate = POSFilter.heterogeneity_rate(homog, "NOUN")
        self.assertAlmostEqual(rate, 0.0, places=5)

    def test_rate_all_different_pos_is_one(self) -> None:
        """If every item differs from ``query_pos``, the rate is 1.0."""
        rate = POSFilter.heterogeneity_rate(self.results, "ADV")
        self.assertAlmostEqual(rate, 1.0, places=5)

    def test_range(self) -> None:
        """Heterogeneity rate stays in [0.0, 1.0]."""
        rate = POSFilter.heterogeneity_rate(self.results, "NOUN")
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_empty_results_raises(self) -> None:
        """Empty ``results`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            POSFilter.heterogeneity_rate([], "NOUN")

    def test_empty_query_pos_raises(self) -> None:
        """Empty ``query_pos`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            POSFilter.heterogeneity_rate(self.results, "")


# ---------------------------------------------------------------------------
# assign_pos_ranks()
# ---------------------------------------------------------------------------

class TestAssignPosRanks(unittest.TestCase):
    """Tests for ``POSFilter.assign_pos_ranks()``."""

    def test_ranks_within_pos_start_at_one(self) -> None:
        """``pos_rank`` for the first member of each POS group starts at 1."""
        results = _sample_results()
        ranked = POSFilter.assign_pos_ranks(results)
        first_per_pos: dict[str, int] = {}
        for r in ranked:
            first_per_pos.setdefault(r.pos_tag, r.pos_rank)
        for pos_tag, first_rank in first_per_pos.items():
            self.assertEqual(first_rank, 1, msg=f"first rank for pos_tag={pos_tag} is not 1")

    def test_ranks_increase_within_each_pos(self) -> None:
        """Within each POS, ``pos_rank`` increases as 1, 2, 3, …"""
        results = _sample_results()
        ranked = POSFilter.assign_pos_ranks(results)
        per_pos: dict[str, list[int]] = {}
        for r in ranked:
            per_pos.setdefault(r.pos_tag, []).append(r.pos_rank)
        for pos_tag, ranks in per_pos.items():
            self.assertEqual(
                ranks, list(range(1, len(ranks) + 1)),
                msg=f"ranks for pos_tag={pos_tag} are not contiguous",
            )

    def test_preserves_original_order(self) -> None:
        """The original ``results`` order is preserved."""
        results = _sample_results()
        original_order = [r.word for r in results]
        ranked = POSFilter.assign_pos_ranks(results)
        self.assertEqual([r.word for r in ranked], original_order)

    def test_returns_same_length(self) -> None:
        """Output length equals input length."""
        results = _sample_results()
        ranked = POSFilter.assign_pos_ranks(results)
        self.assertEqual(len(ranked), len(results))


# ---------------------------------------------------------------------------
# top_pos()
# ---------------------------------------------------------------------------

class TestTopPos(unittest.TestCase):
    """Tests for ``POSFilter.top_pos()``."""

    def setUp(self) -> None:
        self.results = _sample_results()  # NOUN×3, VERB×2, ADJ×1

    def test_default_n_returns_top3(self) -> None:
        """Default ``n=3`` returns the top 3 POS tags."""
        top = POSFilter.top_pos(self.results)
        self.assertEqual(top, ["NOUN", "VERB", "ADJ"])

    def test_n_smaller_than_total(self) -> None:
        """``n=1`` returns only the most frequent POS."""
        top = POSFilter.top_pos(self.results, n=1)
        self.assertEqual(top, ["NOUN"])

    def test_n_greater_than_total(self) -> None:
        """When ``n`` exceeds the number of distinct POS tags, all available ones are returned."""
        top = POSFilter.top_pos(self.results, n=10)
        self.assertEqual(set(top), {"NOUN", "VERB", "ADJ"})
        self.assertLessEqual(len(top), 10)

    def test_invalid_n_raises(self) -> None:
        """``n=0`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            POSFilter.top_pos(self.results, n=0)


if __name__ == "__main__":
    unittest.main()
