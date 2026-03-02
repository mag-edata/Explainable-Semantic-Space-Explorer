"""
similarity_engine.py
====================
単語埋め込み空間における類似度検索エンジン。

単一の埋め込み行列を担当するシングルモデル設計。
static (Word2Vec) と SBERT の比較は、それぞれのエンジンを
インスタンス化して compare() に渡すことで実現する。

外部ライブラリによる距離計算は一切使用しない。
すべての距離計算は DistanceMetrics に委譲する。

使用例::

    loader = EmbeddingLoader(Path("assets"))
    loader.load_all()
    metrics = DistanceMetrics()

    static_engine = SimilarityEngine(
        vectors=loader.static_vectors,
        vocab=loader.vocab,
        pos_tags=loader.pos,
        metrics=metrics,
    )
    sbert_engine = SimilarityEngine(
        vectors=loader.sbert_vectors,
        vocab=loader.vocab,
        pos_tags=loader.pos,
        metrics=metrics,
    )

    results     = static_engine.search("king", top_k=10)
    comparison  = static_engine.compare("king", other=sbert_engine, top_k=10)
    dist        = static_engine.get_distance_distribution("king")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from core.distance_metrics import DistanceMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# カスタム例外
# ---------------------------------------------------------------------------

class SimilarityEngineError(Exception):
    """SimilarityEngine 固有の例外基底クラス。"""


class UnknownWordError(SimilarityEngineError):
    """語彙に存在しない単語が指定された場合の例外。"""


class InvalidTopKError(SimilarityEngineError):
    """top_k の値が不正な場合の例外。"""


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """1件の類似語検索結果。

    Attributes:
        word:        対象単語
        index:       語彙インデックス（0 始まり）
        similarity:  クエリとのコサイン類似度スコア [-1.0, 1.0]
        rank:        全体順位（1 始まり、類似度降順）
        pos_tag:     品詞ラベル（例: "NOUN", "VERB"）
        pos_rank:    同品詞内での順位（1 始まり）
        explanation: 距離計算の内訳辞書
            - dot_product (float): 内積
            - norm_a      (float): クエリベクトルの L2 ノルム
            - norm_b      (float): 対象ベクトルの L2 ノルム
            - similarity  (float): コサイン類似度
            - formula     (str):   計算式の文字列表現
    """

    word: str
    index: int
    similarity: float
    rank: int
    pos_tag: str
    pos_rank: int
    explanation: dict


@dataclass
class ComparisonResult:
    """2つの SimilarityEngine の Top-K 類似語比較結果。

    compare(query, other) を呼んだとき、
    self のエンジン結果が static_results、
    other のエンジン結果が sbert_results に格納される。

    Attributes:
        query_word:      クエリ単語
        static_results:  self エンジンによる Top-K 検索結果
        sbert_results:   other エンジンによる Top-K 検索結果
        common_words:    両エンジンに共通する単語リスト（アルファベット順）
        static_only:     self エンジンのみに出現する単語リスト（アルファベット順）
        sbert_only:      other エンジンのみに出現する単語リスト（アルファベット順）
        rank_diff:       共通語ごとの順位差 {word: static順位 - sbert順位}
                         正値 = static で低順位（下位）、負値 = SBERT で低順位
        similarity_diff: 共通語ごとの類似度差 {word: static類似度 - sbert類似度}
    """

    query_word: str
    static_results: List[SearchResult]
    sbert_results: List[SearchResult]
    common_words: List[str]
    static_only: List[str]
    sbert_only: List[str]
    rank_diff: Dict[str, int]
    similarity_diff: Dict[str, float]


# ---------------------------------------------------------------------------
# SimilarityEngine
# ---------------------------------------------------------------------------

class SimilarityEngine:
    """単語埋め込み空間における類似度検索エンジン（単一モデル）。

    1インスタンスが1つの埋め込み行列（static または SBERT）を担当する。
    static vs SBERT の比較は compare(other=sbert_engine) で実行する。

    EmbeddingLoader には依存しない。
    vectors / vocab / pos_tags / metrics を直接注入する（依存注入）。

    外部ライブラリによる距離計算は一切使用しない。
    すべての距離計算は DistanceMetrics に委譲する。

    Attributes:
        _vectors:       埋め込み行列 shape (N, D)
        _vocab:         単語 → インデックス辞書 {"word": index}
        _index_to_word: インデックス → 単語リスト（vocab の逆引き）
        _pos_tags:      品詞ラベル配列 shape (N,)
        _metrics:       コサイン類似度計算クラス
        _n_vocab:       語彙サイズ N
    """

    def __init__(
        self,
        vectors: np.ndarray,
        vocab: Dict[str, int],
        pos_tags: np.ndarray,
        metrics: DistanceMetrics,
    ) -> None:
        """SimilarityEngine を初期化する。

        index_to_word（逆引きリスト）は vocab から自動構築する。

        Args:
            vectors:   埋め込み行列 shape (N, D)。
            vocab:     単語 → インデックス辞書 {"word": index}。
            pos_tags:  品詞ラベル配列 shape (N,)。
            metrics:   DistanceMetrics インスタンス。

        Raises:
            TypeError:  引数の型が不正な場合。
            ValueError: vectors と vocab のサイズが一致しない場合。
        """
        if not isinstance(vectors, np.ndarray):
            raise TypeError(
                f"vectors は np.ndarray 型である必要があります。"
                f"受け取った型: {type(vectors)}"
            )
        if not isinstance(vocab, dict):
            raise TypeError(
                f"vocab は dict 型である必要があります。"
                f"受け取った型: {type(vocab)}"
            )
        if not isinstance(pos_tags, np.ndarray):
            raise TypeError(
                f"pos_tags は np.ndarray 型である必要があります。"
                f"受け取った型: {type(pos_tags)}"
            )
        if not isinstance(metrics, DistanceMetrics):
            raise TypeError(
                f"metrics は DistanceMetrics 型である必要があります。"
                f"受け取った型: {type(metrics)}"
            )

        n_vocab: int = len(vocab)
        if vectors.shape[0] != n_vocab:
            raise ValueError(
                f"vectors の行数 ({vectors.shape[0]}) と "
                f"vocab のサイズ ({n_vocab}) が一致しません"
            )

        self._vectors: np.ndarray = vectors
        self._vocab: Dict[str, int] = vocab
        self._pos_tags: np.ndarray = pos_tags
        self._metrics: DistanceMetrics = metrics
        self._n_vocab: int = n_vocab

        # vocab から逆引きリストを構築: O(N)
        self._index_to_word: List[str] = [""] * n_vocab
        for word, idx in vocab.items():
            self._index_to_word[idx] = word

        logger.info(
            "SimilarityEngine 初期化完了: n_vocab=%d, dim=%d",
            self._n_vocab,
            self._vectors.shape[1],
        )

    # -----------------------------------------------------------------------
    # 公開メソッド
    # -----------------------------------------------------------------------

    def search(
        self,
        query_word: str,
        top_k: int = 10,
        pos_filter: str | None = None,
    ) -> List[SearchResult]:
        """クエリ単語の Top-K 類似語を検索する。

        Args:
            query_word: 検索クエリ単語（語彙内に存在する必要がある）。
            top_k:      返す類似語の最大件数（デフォルト 10）。
            pos_filter: 指定した品詞のみに絞り込む（例: "NOUN"）。
                        None の場合は全品詞を返す。

        Returns:
            list[SearchResult]: 類似度降順の検索結果リスト。

        Raises:
            UnknownWordError: query_word が語彙に存在しない場合。
            InvalidTopKError: top_k が 1 未満の場合。
        """
        self._validate_top_k(top_k)
        query_idx: int = self.word_to_index(query_word)
        query_vec: np.ndarray = self._vectors[query_idx]

        logger.debug(
            "search 開始: query=%s (idx=%d), top_k=%d, pos_filter=%s",
            query_word, query_idx, top_k, pos_filter,
        )

        results: List[SearchResult] = self._build_results(query_vec, query_idx, top_k)

        if pos_filter is not None:
            results = [r for r in results if r.pos_tag == pos_filter]
            logger.debug(
                "品詞フィルタ適用: pos_filter=%s -> %d 件", pos_filter, len(results)
            )

        return self._assign_pos_ranks(results)

    def compare(
        self,
        query_word: str,
        other: "SimilarityEngine",
        top_k: int = 10,
    ) -> ComparisonResult:
        """self エンジンと other エンジンの Top-K 類似語を比較分析する。

        self.search() の結果が static_results に、
        other.search() の結果が sbert_results に格納される。
        呼び出し例: static_engine.compare("king", other=sbert_engine)

        Args:
            query_word: 検索クエリ単語。
            other:      比較対象の SimilarityEngine インスタンス。
            top_k:      各エンジンで返す類似語の最大件数。

        Returns:
            ComparisonResult: 両エンジンの比較結果。

        Raises:
            UnknownWordError: query_word が self の語彙に存在しない場合。
            InvalidTopKError: top_k が 1 未満の場合。
            TypeError:        other が SimilarityEngine 型でない場合。
        """
        if not isinstance(other, SimilarityEngine):
            raise TypeError(
                f"other は SimilarityEngine 型である必要があります。"
                f"受け取った型: {type(other)}"
            )
        self._validate_top_k(top_k)

        # 両エンジンの公開メソッドのみ使用（実装詳細に依存しない）
        static_results: List[SearchResult] = self.search(query_word, top_k=top_k)
        sbert_results: List[SearchResult] = other.search(query_word, top_k=top_k)

        # 集合演算で共通語・固有語を算出
        static_word_set: set[str] = {r.word for r in static_results}
        sbert_word_set: set[str] = {r.word for r in sbert_results}

        common_words: List[str] = sorted(static_word_set & sbert_word_set)
        static_only: List[str] = sorted(static_word_set - sbert_word_set)
        sbert_only: List[str] = sorted(sbert_word_set - static_word_set)

        # 共通語ごとの順位差・類似度差を計算
        static_rank_map: Dict[str, int] = {r.word: r.rank for r in static_results}
        sbert_rank_map: Dict[str, int] = {r.word: r.rank for r in sbert_results}
        static_sim_map: Dict[str, float] = {r.word: r.similarity for r in static_results}
        sbert_sim_map: Dict[str, float] = {r.word: r.similarity for r in sbert_results}

        rank_diff: Dict[str, int] = {
            word: static_rank_map[word] - sbert_rank_map[word]
            for word in common_words
        }
        similarity_diff: Dict[str, float] = {
            word: static_sim_map[word] - sbert_sim_map[word]
            for word in common_words
        }

        logger.info(
            "compare 完了: query=%s, 共通=%d語, static固有=%d語, sbert固有=%d語",
            query_word, len(common_words), len(static_only), len(sbert_only),
        )

        return ComparisonResult(
            query_word=query_word,
            static_results=static_results,
            sbert_results=sbert_results,
            common_words=common_words,
            static_only=static_only,
            sbert_only=sbert_only,
            rank_diff=rank_diff,
            similarity_diff=similarity_diff,
        )

    def get_distance_distribution(
        self,
        query_word: str,
    ) -> dict:
        """クエリ単語と全語彙間のコサイン類似度の分布統計を返す。

        全語彙（N-1件、自己参照除外）との類似度を一括計算し、
        分布の平均・標準偏差・Z-score を算出する。

        Z-score = (top1_similarity - mean) / std
        「Top-1 の類似語が平均から何標準偏差離れているか」を示す。
        高いほど、上位の類似語が孤立した意味的近隣を持つ。

        Args:
            query_word: 検索クエリ単語。

        Returns:
            dict: 以下のキーを持つ辞書。

            - "query_word"      (str):        クエリ単語
            - "mean"            (float):      全語彙との平均コサイン類似度
            - "std"             (float):      標準偏差
            - "top1_similarity" (float):      Top-1 の類似度スコア
            - "z_score"         (float):      Top-1 スコアの Z-score
            - "histogram_data"  (list[float]): 全 N-1 件の類似度スコア（可視化用）

        Raises:
            UnknownWordError: query_word が語彙に存在しない場合。
        """
        query_idx: int = self.word_to_index(query_word)
        query_vec: np.ndarray = self._vectors[query_idx]

        # 全語彙との類似度を一括計算（DistanceMetrics に委譲）
        all_similarities: np.ndarray = self._metrics.cosine_similarity_batch(
            query_vec, self._vectors
        )

        # 自己参照（類似度 1.0）を除外したマスクを作成
        mask: np.ndarray = np.ones(self._n_vocab, dtype=bool)
        mask[query_idx] = False
        sims_without_self: np.ndarray = all_similarities[mask]

        mean_sim: float = float(np.mean(sims_without_self))
        std_sim: float = float(np.std(sims_without_self))
        top1_sim: float = float(np.max(sims_without_self))

        # std がゼロの場合（全ベクトルが同一など異常系）はゼロ除算を回避
        z_score: float = (
            (top1_sim - mean_sim) / std_sim if std_sim > 0.0 else 0.0
        )

        logger.debug(
            "距離分布: query=%s, mean=%.4f, std=%.4f, top1=%.4f, z_score=%.4f",
            query_word, mean_sim, std_sim, top1_sim, z_score,
        )

        return {
            "query_word": query_word,
            "mean": mean_sim,
            "std": std_sim,
            "top1_similarity": top1_sim,
            "z_score": z_score,
            "histogram_data": sims_without_self.tolist(),
        }

    def word_to_index(self, word: str) -> int:
        """単語を語彙インデックスに変換する。

        Args:
            word: 変換対象の単語。

        Returns:
            int: 語彙インデックス（0 始まり）。

        Raises:
            UnknownWordError: word が語彙に存在しない場合。
        """
        index: int | None = self._vocab.get(word)
        if index is None:
            raise UnknownWordError(
                f"'{word}' は語彙に存在しません。"
                f"（語彙サイズ: {self._n_vocab}）"
            )
        return index

    # -----------------------------------------------------------------------
    # Private メソッド
    # -----------------------------------------------------------------------

    def _build_results(
        self,
        query_vec: np.ndarray,
        query_idx: int,
        top_k: int,
    ) -> List[SearchResult]:
        """クエリベクトルから SearchResult のリストを構築する。

        _search_single() で Top-K インデックスと類似度を取得し、
        各結果に対して _build_explanation() で説明辞書を付与する。
        pos_rank は 0 で初期化し、_assign_pos_ranks() で後から設定する。

        Args:
            query_vec: クエリ単語の埋め込みベクトル shape (D,)。
            query_idx: クエリ単語のインデックス（自己参照除外に使用）。
            top_k:     返す件数。

        Returns:
            list[SearchResult]: 類似度降順のリスト（pos_rank=0 で初期化済み）。
        """
        indices, similarities = self._search_single(query_vec, query_idx, top_k)

        results: List[SearchResult] = []
        for rank, (idx, sim) in enumerate(zip(indices, similarities), start=1):
            word: str = self._index_to_word[int(idx)]
            pos_tag: str = str(self._pos_tags[int(idx)])
            target_vec: np.ndarray = self._vectors[int(idx)]
            explanation: dict = self._build_explanation(query_vec, target_vec)

            results.append(SearchResult(
                word=word,
                index=int(idx),
                similarity=float(sim),
                rank=rank,
                pos_tag=pos_tag,
                pos_rank=0,   # _assign_pos_ranks() で後から設定
                explanation=explanation,
            ))

        return results

    def _search_single(
        self,
        query_vec: np.ndarray,
        query_idx: int,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """クエリベクトルと全語彙行列のコサイン類似度を計算し Top-K を返す。

        計算は DistanceMetrics.cosine_similarity_batch() に完全委譲する。
        自己参照（クエリ単語自身）は -inf に置き換えて除外する。
        top_k が語彙サイズを超える場合は語彙サイズに切り詰める。

        効率化:
            np.argpartition で O(N) に Top-K を絞り込んだ後、
            その k 件のみを O(k log k) でソートする。
            全語彙ソートの O(N log N) より高速。

        Args:
            query_vec: クエリベクトル shape (D,)。
            query_idx: クエリ単語のインデックス（除外対象）。
            top_k:     返す件数。

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - indices      shape (effective_k,): 類似度降順のインデックス
                - similarities shape (effective_k,): 対応するコサイン類似度スコア
        """
        # 類似度を一括計算（外部ライブラリ禁止: DistanceMetrics に委譲）
        # cosine_similarity_batch は新規 ndarray を返すため self._vectors は変更されない
        all_sims: np.ndarray = self._metrics.cosine_similarity_batch(
            query_vec, self._vectors
        )

        # 自己参照を -inf に設定して Top-K から除外
        all_sims[query_idx] = -np.inf

        # top_k を有効語彙数（自己除外後: N-1）に制限
        effective_k: int = min(top_k, self._n_vocab - 1)

        # argpartition で Top-K インデックスを O(N) で取得（順序は不定）
        top_k_unordered: np.ndarray = np.argpartition(all_sims, -effective_k)[-effective_k:]

        # Top-K 内をスコア降順でソート: O(k log k)
        sorted_order: np.ndarray = np.argsort(all_sims[top_k_unordered])[::-1]
        top_k_indices: np.ndarray = top_k_unordered[sorted_order]
        top_k_sims: np.ndarray = all_sims[top_k_indices]

        return top_k_indices, top_k_sims

    def _assign_pos_ranks(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """SearchResult リストに同品詞内順位（pos_rank）を付与する。

        results は類似度降順でソート済みであることを前提とする。
        同品詞グループ内での登場順（類似度順）がそのまま pos_rank になる。
        元の results の順序（全体順位）は変えない。

        Args:
            results: SearchResult のリスト（類似度降順）。

        Returns:
            list[SearchResult]: pos_rank が設定されたリスト（元の順序を保持）。
        """
        pos_counter: Dict[str, int] = {}

        for result in results:
            pos: str = result.pos_tag
            pos_counter[pos] = pos_counter.get(pos, 0) + 1
            result.pos_rank = pos_counter[pos]

        return results

    def _build_explanation(
        self,
        query_vec: np.ndarray,
        target_vec: np.ndarray,
    ) -> dict:
        """2ベクトル間のコサイン類似度計算過程を説明辞書として構築する。

        DistanceMetrics.explain() に委譲する。
        返される辞書は UI での「なぜこのスコアか」の表示に使用する。

        Args:
            query_vec:  クエリ単語のベクトル shape (D,)。
            target_vec: 対象単語のベクトル shape (D,)。

        Returns:
            dict: DistanceMetrics.explain() が返す辞書。
                - "dot_product" (float): 内積
                - "norm_a"      (float): クエリベクトルの L2 ノルム
                - "norm_b"      (float): 対象ベクトルの L2 ノルム
                - "similarity"  (float): コサイン類似度
                - "formula"     (str):   計算式の文字列表現
        """
        return self._metrics.explain(query_vec, target_vec)

    def _validate_top_k(self, top_k: int) -> None:
        """top_k の値を検証する。

        Args:
            top_k: 検証対象の値。

        Raises:
            InvalidTopKError: top_k が 1 未満の場合。
        """
        if top_k < 1:
            raise InvalidTopKError(
                f"top_k は 1 以上である必要があります。受け取った値: {top_k}"
            )
