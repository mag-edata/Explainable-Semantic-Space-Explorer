"""
test_pos_filter.py
==================
POSFilter の単体テスト。

テスト対象:
    - filter():            指定品詞のフィルタリング
    - group_by_pos():      品詞ごとのグループ化
    - pos_distribution():  品詞ごとの出現数集計
    - heterogeneity_rate(): 異品詞率
    - assign_pos_ranks():  同品詞内順位の付与
    - top_pos():           出現上位の品詞リスト

実行方法:
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
# テスト用フィクスチャ
# ---------------------------------------------------------------------------

def _make_result(
    word: str,
    rank: int,
    pos_tag: str,
    similarity: float = 0.5,
) -> SearchResult:
    """SearchResult をテスト用に簡単に生成するヘルパ。

    pos_rank は assign_pos_ranks() 前提のため 0 で初期化する。
    explanation も空辞書とする（POSFilter は見ない）。
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
    """類似度降順の SearchResult 6件（NOUN×3 / VERB×2 / ADJ×1）。"""
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
    """POSFilter.filter() のテスト。"""

    def setUp(self) -> None:
        self.results = _sample_results()

    def test_filter_returns_only_specified_pos(self) -> None:
        """指定品詞のみが返ることを確認。"""
        filtered = POSFilter.filter(self.results, "NOUN")
        self.assertEqual(len(filtered), 3)
        for r in filtered:
            self.assertEqual(r.pos_tag, "NOUN")

    def test_filter_preserves_original_order(self) -> None:
        """フィルタ後の順序が元の results の順序を保つことを確認。"""
        filtered = POSFilter.filter(self.results, "NOUN")
        words = [r.word for r in filtered]
        self.assertEqual(words, ["apple", "banana", "cherry"])

    def test_filter_preserves_original_rank(self) -> None:
        """フィルタ後も元の rank（全体順位）が保たれることを確認。"""
        filtered = POSFilter.filter(self.results, "VERB")
        ranks = [r.rank for r in filtered]
        self.assertEqual(ranks, [2, 4])  # 元の順位を保持

    def test_filter_empty_pos_tag_raises_value_error(self) -> None:
        """pos_tag が空文字の場合に ValueError が送出される。"""
        with self.assertRaises(ValueError):
            POSFilter.filter(self.results, "")

    def test_filter_unknown_pos_raises(self) -> None:
        """results に存在しない品詞を指定した場合に UnknownPOSTagError が送出される。"""
        with self.assertRaises(UnknownPOSTagError):
            POSFilter.filter(self.results, "ADV")


# ---------------------------------------------------------------------------
# group_by_pos()
# ---------------------------------------------------------------------------

class TestGroupByPos(unittest.TestCase):
    """POSFilter.group_by_pos() のテスト。"""

    def setUp(self) -> None:
        self.results = _sample_results()

    def test_returns_dict(self) -> None:
        """戻り値が dict 型であることを確認。"""
        groups = POSFilter.group_by_pos(self.results)
        self.assertIsInstance(groups, dict)

    def test_keys_are_unique_pos_tags(self) -> None:
        """キーが results に登場する品詞集合と一致することを確認。"""
        groups = POSFilter.group_by_pos(self.results)
        self.assertEqual(set(groups.keys()), {"NOUN", "VERB", "ADJ"})

    def test_each_group_size(self) -> None:
        """各グループの件数が正しいことを確認。"""
        groups = POSFilter.group_by_pos(self.results)
        self.assertEqual(len(groups["NOUN"]), 3)
        self.assertEqual(len(groups["VERB"]), 2)
        self.assertEqual(len(groups["ADJ"]),  1)

    def test_group_preserves_order(self) -> None:
        """各グループ内の順序が元の results の順序を保つことを確認。"""
        groups = POSFilter.group_by_pos(self.results)
        noun_words = [r.word for r in groups["NOUN"]]
        self.assertEqual(noun_words, ["apple", "banana", "cherry"])

    def test_empty_results(self) -> None:
        """空 results の場合に空 dict が返ることを確認。"""
        groups = POSFilter.group_by_pos([])
        self.assertEqual(groups, {})


# ---------------------------------------------------------------------------
# pos_distribution()
# ---------------------------------------------------------------------------

class TestPosDistribution(unittest.TestCase):
    """POSFilter.pos_distribution() のテスト。"""

    def setUp(self) -> None:
        self.results = _sample_results()

    def test_counts_are_correct(self) -> None:
        """各品詞の出現数が正しく集計されることを確認。"""
        dist = POSFilter.pos_distribution(self.results)
        self.assertEqual(dist["NOUN"], 3)
        self.assertEqual(dist["VERB"], 2)
        self.assertEqual(dist["ADJ"],  1)

    def test_sorted_descending(self) -> None:
        """出現数の多い順にソートされていることを確認。"""
        dist = POSFilter.pos_distribution(self.results)
        counts = list(dist.values())
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_sum_equals_total(self) -> None:
        """全件数の合計が results の長さと一致することを確認。"""
        dist = POSFilter.pos_distribution(self.results)
        self.assertEqual(sum(dist.values()), len(self.results))

    def test_empty_results(self) -> None:
        """空 results の場合に空 dict が返ることを確認。"""
        self.assertEqual(POSFilter.pos_distribution([]), {})


# ---------------------------------------------------------------------------
# heterogeneity_rate()
# ---------------------------------------------------------------------------

class TestHeterogeneityRate(unittest.TestCase):
    """POSFilter.heterogeneity_rate() のテスト。"""

    def setUp(self) -> None:
        self.results = _sample_results()  # NOUN×3, VERB×2, ADJ×1 = 6件

    def test_rate_with_query_pos_noun(self) -> None:
        """query_pos=NOUN の異品詞率は 3/6 = 0.5。"""
        rate = POSFilter.heterogeneity_rate(self.results, "NOUN")
        self.assertAlmostEqual(rate, 0.5, places=5)

    def test_rate_with_query_pos_verb(self) -> None:
        """query_pos=VERB の異品詞率は 4/6 ≈ 0.6667。"""
        rate = POSFilter.heterogeneity_rate(self.results, "VERB")
        self.assertAlmostEqual(rate, 4 / 6, places=5)

    def test_rate_all_same_pos_is_zero(self) -> None:
        """全件が query_pos と同じなら異品詞率は 0.0。"""
        homog = [_make_result(f"w{i}", i + 1, "NOUN") for i in range(5)]
        rate = POSFilter.heterogeneity_rate(homog, "NOUN")
        self.assertAlmostEqual(rate, 0.0, places=5)

    def test_rate_all_different_pos_is_one(self) -> None:
        """全件が query_pos と異なるなら異品詞率は 1.0。"""
        rate = POSFilter.heterogeneity_rate(self.results, "ADV")
        self.assertAlmostEqual(rate, 1.0, places=5)

    def test_range(self) -> None:
        """異品詞率は [0.0, 1.0] の範囲に収まる。"""
        rate = POSFilter.heterogeneity_rate(self.results, "NOUN")
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_empty_results_raises(self) -> None:
        """results が空の場合に ValueError が送出される。"""
        with self.assertRaises(ValueError):
            POSFilter.heterogeneity_rate([], "NOUN")

    def test_empty_query_pos_raises(self) -> None:
        """query_pos が空文字の場合に ValueError が送出される。"""
        with self.assertRaises(ValueError):
            POSFilter.heterogeneity_rate(self.results, "")


# ---------------------------------------------------------------------------
# assign_pos_ranks()
# ---------------------------------------------------------------------------

class TestAssignPosRanks(unittest.TestCase):
    """POSFilter.assign_pos_ranks() のテスト。"""

    def test_ranks_within_pos_start_at_one(self) -> None:
        """各品詞グループの先頭 pos_rank が 1 から始まることを確認。"""
        results = _sample_results()
        ranked = POSFilter.assign_pos_ranks(results)
        first_per_pos: dict[str, int] = {}
        for r in ranked:
            first_per_pos.setdefault(r.pos_tag, r.pos_rank)
        for pos_tag, first_rank in first_per_pos.items():
            self.assertEqual(first_rank, 1, msg=f"pos_tag={pos_tag} の先頭順位が 1 でない")

    def test_ranks_increase_within_each_pos(self) -> None:
        """同品詞内で pos_rank が 1, 2, 3 ... と連続することを確認。"""
        results = _sample_results()
        ranked = POSFilter.assign_pos_ranks(results)
        per_pos: dict[str, list[int]] = {}
        for r in ranked:
            per_pos.setdefault(r.pos_tag, []).append(r.pos_rank)
        for pos_tag, ranks in per_pos.items():
            self.assertEqual(
                ranks, list(range(1, len(ranks) + 1)),
                msg=f"pos_tag={pos_tag} の順位が連続していない",
            )

    def test_preserves_original_order(self) -> None:
        """元の results の順序が変わらないことを確認。"""
        results = _sample_results()
        original_order = [r.word for r in results]
        ranked = POSFilter.assign_pos_ranks(results)
        self.assertEqual([r.word for r in ranked], original_order)

    def test_returns_same_length(self) -> None:
        """戻り値の件数が入力と一致することを確認。"""
        results = _sample_results()
        ranked = POSFilter.assign_pos_ranks(results)
        self.assertEqual(len(ranked), len(results))


# ---------------------------------------------------------------------------
# top_pos()
# ---------------------------------------------------------------------------

class TestTopPos(unittest.TestCase):
    """POSFilter.top_pos() のテスト。"""

    def setUp(self) -> None:
        self.results = _sample_results()  # NOUN×3, VERB×2, ADJ×1

    def test_default_n_returns_top3(self) -> None:
        """デフォルト n=3 で上位3品詞が返ることを確認。"""
        top = POSFilter.top_pos(self.results)
        self.assertEqual(top, ["NOUN", "VERB", "ADJ"])

    def test_n_smaller_than_total(self) -> None:
        """n=1 のとき最頻品詞だけが返ることを確認。"""
        top = POSFilter.top_pos(self.results, n=1)
        self.assertEqual(top, ["NOUN"])

    def test_n_greater_than_total(self) -> None:
        """n が品詞種数を超えても利用可能な全品詞が返ることを確認。"""
        top = POSFilter.top_pos(self.results, n=10)
        self.assertEqual(set(top), {"NOUN", "VERB", "ADJ"})
        self.assertLessEqual(len(top), 10)

    def test_invalid_n_raises(self) -> None:
        """n=0 の場合に ValueError が送出される。"""
        with self.assertRaises(ValueError):
            POSFilter.top_pos(self.results, n=0)


if __name__ == "__main__":
    unittest.main()
