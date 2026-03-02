"""
test_similarity_engine.py
=========================
SimilarityEngine の単体テスト。

テスト対象:
    - search(): Top-K 類似語検索
    - compare(): 2エンジン間の比較
    - get_distance_distribution(): 距離分布の計算
    - word_to_index(): 語彙インデックスの逆引き

実行方法:
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
# テスト用フィクスチャ（小規模モックデータ）
# ---------------------------------------------------------------------------

def _make_engine(
    n: int = 6,
    dim: int = 4,
    seed: int = 42,
) -> tuple[SimilarityEngine, dict[str, int], np.ndarray]:
    """テスト用の小規模 SimilarityEngine を生成する。

    Args:
        n:    語彙数
        dim:  ベクトル次元数
        seed: 乱数シード

    Returns:
        (engine, vocab, vectors) のタプル
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
    """SimilarityEngine.search() のテスト。"""

    def setUp(self) -> None:
        self.engine, self.vocab, self.vectors = _make_engine(n=6, dim=4)

    def test_returns_list_of_search_results(self) -> None:
        """戻り値が SearchResult のリストであることを確認。"""
        results = self.engine.search("word0", top_k=3)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, SearchResult) for r in results))

    def test_top_k_count(self) -> None:
        """返す件数が top_k と一致することを確認（クエリ自身を除く）。"""
        results = self.engine.search("word0", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_excludes_query_itself(self) -> None:
        """検索結果にクエリ単語自身が含まれないことを確認。"""
        results = self.engine.search("word0", top_k=5)
        words = [r.word for r in results]
        self.assertNotIn("word0", words)

    def test_sorted_by_similarity_descending(self) -> None:
        """類似度の降順で並んでいることを確認。"""
        results = self.engine.search("word0", top_k=4)
        sims = [r.similarity for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_rank_starts_at_one(self) -> None:
        """順位が 1 から始まることを確認。"""
        results = self.engine.search("word0", top_k=3)
        self.assertEqual(results[0].rank, 1)

    def test_result_has_explanation(self) -> None:
        """SearchResult が explanation フィールドを持つことを確認。"""
        results = self.engine.search("word0", top_k=1)
        self.assertIn("formula", results[0].explanation)
        self.assertIn("dot_product", results[0].explanation)

    def test_pos_filter(self) -> None:
        """pos_filter 指定時、指定品詞のみ返ることを確認。"""
        results = self.engine.search("word0", top_k=5, pos_filter="NOUN")
        for r in results:
            self.assertEqual(r.pos_tag, "NOUN")

    def test_unknown_word_error(self) -> None:
        """語彙外の単語を渡した場合に UnknownWordError が発生する。"""
        with self.assertRaises(UnknownWordError):
            self.engine.search("nonexistent_word", top_k=3)

    def test_invalid_top_k_zero(self) -> None:
        """top_k=0 を渡した場合に InvalidTopKError が発生する。"""
        with self.assertRaises(InvalidTopKError):
            self.engine.search("word0", top_k=0)

    def test_invalid_top_k_negative(self) -> None:
        """top_k が負の値の場合に InvalidTopKError が発生する。"""
        with self.assertRaises(InvalidTopKError):
            self.engine.search("word0", top_k=-1)


class TestCompare(unittest.TestCase):
    """SimilarityEngine.compare() のテスト。"""

    def setUp(self) -> None:
        self.engine_a, _, _ = _make_engine(n=6, dim=4, seed=0)
        self.engine_b, _, _ = _make_engine(n=6, dim=4, seed=1)

    def test_returns_comparison_result(self) -> None:
        """戻り値が ComparisonResult であることを確認。"""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=3)
        self.assertIsInstance(result, ComparisonResult)

    def test_query_word_field(self) -> None:
        """query_word フィールドが正しいことを確認。"""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=3)
        self.assertEqual(result.query_word, "word0")

    def test_static_and_sbert_results_lengths(self) -> None:
        """static_results と sbert_results の件数が top_k 以内であることを確認。"""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=3)
        self.assertLessEqual(len(result.static_results), 3)
        self.assertLessEqual(len(result.sbert_results), 3)

    def test_common_words_subset(self) -> None:
        """common_words が static_results と sbert_results の語彙の共通部分であることを確認。"""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=4)
        static_words = {r.word for r in result.static_results}
        sbert_words  = {r.word for r in result.sbert_results}
        expected_common = static_words & sbert_words
        self.assertEqual(set(result.common_words), expected_common)

    def test_rank_diff_keys_are_common_words(self) -> None:
        """rank_diff のキーが共通語であることを確認。"""
        result = self.engine_a.compare("word0", other=self.engine_b, top_k=4)
        self.assertEqual(set(result.rank_diff.keys()), set(result.common_words))

    def test_unknown_word_error(self) -> None:
        """語彙外の単語を渡した場合に UnknownWordError が発生する。"""
        with self.assertRaises(UnknownWordError):
            self.engine_a.compare("no_such_word", other=self.engine_b, top_k=3)


class TestGetDistanceDistribution(unittest.TestCase):
    """SimilarityEngine.get_distance_distribution() のテスト。"""

    def setUp(self) -> None:
        self.engine, _, _ = _make_engine(n=10, dim=8)

    def test_returns_dict(self) -> None:
        """戻り値が dict であることを確認。"""
        result = self.engine.get_distance_distribution("word0")
        self.assertIsInstance(result, dict)

    def test_required_keys(self) -> None:
        """必須キーが全て含まれることを確認。"""
        result = self.engine.get_distance_distribution("word0")
        for key in ("query_word", "mean", "std", "top1_similarity", "z_score", "histogram_data"):
            self.assertIn(key, result, msg=f"キー '{key}' が存在しません")

    def test_query_word_field(self) -> None:
        """query_word フィールドがクエリと一致することを確認。"""
        result = self.engine.get_distance_distribution("word0")
        self.assertEqual(result["query_word"], "word0")

    def test_top1_ge_mean(self) -> None:
        """Top-1 類似度 >= 平均類似度 であることを確認。"""
        result = self.engine.get_distance_distribution("word0")
        self.assertGreaterEqual(result["top1_similarity"], result["mean"] - 1e-6)

    def test_histogram_data_length(self) -> None:
        """histogram_data の長さが語彙数 - 1 であることを確認（クエリ自身を除く）。"""
        n = 10
        result = self.engine.get_distance_distribution("word0")
        self.assertEqual(len(result["histogram_data"]), n - 1)

    def test_std_is_nonnegative(self) -> None:
        """標準偏差が非負であることを確認。"""
        result = self.engine.get_distance_distribution("word0")
        self.assertGreaterEqual(result["std"], 0.0)

    def test_unknown_word_error(self) -> None:
        """語彙外の単語を渡した場合に UnknownWordError が発生する。"""
        with self.assertRaises(UnknownWordError):
            self.engine.get_distance_distribution("no_such_word")


class TestWordToIndex(unittest.TestCase):
    """SimilarityEngine.word_to_index() のテスト。"""

    def setUp(self) -> None:
        self.engine, self.vocab, _ = _make_engine(n=6, dim=4)

    def test_correct_index(self) -> None:
        """単語に対応するインデックスが正しく返ることを確認。"""
        for word, expected_idx in self.vocab.items():
            self.assertEqual(self.engine.word_to_index(word), expected_idx)

    def test_unknown_word_error(self) -> None:
        """語彙外の単語を渡した場合に UnknownWordError が発生する。"""
        with self.assertRaises(UnknownWordError):
            self.engine.word_to_index("unknown_word_xyz")


if __name__ == "__main__":
    unittest.main()
