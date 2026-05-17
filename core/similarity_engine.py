"""
similarity_engine.py
====================
Similarity-search engine over a word embedding space.

A single-model design: each instance owns one embedding matrix. The
static (Word2Vec) vs. contextual (SBERT) comparison is realized by
instantiating two engines and passing one to ``compare()``.

No external library is ever used for distance computation. All distance
calculations are delegated to ``DistanceMetrics``.

Example usage::

    loader = EmbeddingLoader(Path("data"))
    loader.load_all()
    metrics = DistanceMetrics()

    static_engine = SimilarityEngine(
        vectors=loader.static_vectors,
        vocab=loader.vocab,
        pos_tags=loader.pos,
        metrics=metrics,
    )
    contextual_engine = SimilarityEngine(
        vectors=loader.contextual_vectors,
        vocab=loader.vocab,
        pos_tags=loader.pos,
        metrics=metrics,
    )

    results     = static_engine.search("king", top_k=10)
    comparison  = static_engine.compare("king", other=contextual_engine, top_k=10)
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
# Custom exceptions
# ---------------------------------------------------------------------------

class SimilarityEngineError(Exception):
    """Base exception class specific to SimilarityEngine."""


class UnknownWordError(SimilarityEngineError):
    """Raised when a requested word is not found in the vocabulary."""


class InvalidTopKError(SimilarityEngineError):
    """Raised when ``top_k`` is invalid."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single similar-word search result.

    Attributes:
        word:        The target word
        index:       Vocabulary index (0-based)
        similarity:  Cosine similarity with the query, in [-1.0, 1.0]
        rank:        Overall rank (1-based, descending by similarity)
        pos_tag:     POS label (for example, "NOUN", "VERB")
        pos_rank:    Rank within the same POS group (1-based)
        explanation: Breakdown dictionary of the distance computation
            - dot_product (float): Dot product
            - norm_a      (float): L2 norm of the query vector
            - norm_b      (float): L2 norm of the target vector
            - similarity  (float): Cosine similarity
            - formula     (str):   String representation of the formula
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
    """Top-K similar-word comparison between two SimilarityEngine instances.

    When ``compare(query, other)`` is invoked, results from ``self`` are
    stored in ``static_results`` and results from ``other`` are stored in
    ``contextual_results``.

    Attributes:
        query_word:          The query word
        static_results:      Top-K results from the ``self`` engine
        contextual_results:  Top-K results from the ``other`` engine
        common_words:        Words present in both engines (alphabetical order)
        static_only:         Words unique to the ``self`` engine (alphabetical order)
        contextual_only:     Words unique to the ``other`` engine (alphabetical order)
        rank_diff:           Per-shared-word rank difference
                             ``{word: static_rank - contextual_rank}``.
                             Positive → ranked lower in static; negative → lower in contextual.
        similarity_diff:     Per-shared-word similarity difference
                             ``{word: static_similarity - contextual_similarity}``
    """

    query_word: str
    static_results: List[SearchResult]
    contextual_results: List[SearchResult]
    common_words: List[str]
    static_only: List[str]
    contextual_only: List[str]
    rank_diff: Dict[str, int]
    similarity_diff: Dict[str, float]


# ---------------------------------------------------------------------------
# SimilarityEngine
# ---------------------------------------------------------------------------

class SimilarityEngine:
    """Similarity-search engine over a word embedding space (single model).

    Each instance is responsible for one embedding matrix (static or
    contextual). Static vs. contextual comparison is performed via
    ``compare(other=contextual_engine)``.

    Does not depend on ``EmbeddingLoader``. ``vectors`` / ``vocab`` /
    ``pos_tags`` / ``metrics`` are injected directly (dependency injection).

    No external library is ever used for distance computation. All
    distance calculations are delegated to ``DistanceMetrics``.

    Attributes:
        _vectors:       Embedding matrix, shape (N, D)
        _vocab:         Word → index dictionary {"word": index}
        _index_to_word: Index → word list (the inverse mapping of vocab)
        _pos_tags:      POS label array, shape (N,)
        _metrics:       Cosine-similarity computation class
        _n_vocab:       Vocabulary size N
    """

    def __init__(
        self,
        vectors: np.ndarray,
        vocab: Dict[str, int],
        pos_tags: np.ndarray,
        metrics: DistanceMetrics,
    ) -> None:
        """Initialize the SimilarityEngine.

        ``index_to_word`` (the inverse mapping) is built automatically from
        ``vocab``.

        Args:
            vectors:   Embedding matrix, shape (N, D).
            vocab:     Word → index dictionary {"word": index}.
            pos_tags:  POS label array, shape (N,).
            metrics:   ``DistanceMetrics`` instance.

        Raises:
            TypeError:  If any argument has the wrong type.
            ValueError: If the size of ``vectors`` does not match ``vocab``.
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

        # Build the inverse index → word list from vocab: O(N)
        self._index_to_word: List[str] = [""] * n_vocab
        for word, idx in vocab.items():
            self._index_to_word[idx] = word

        logger.info(
            "SimilarityEngine 初期化完了: n_vocab=%d, dim=%d",
            self._n_vocab,
            self._vectors.shape[1],
        )

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def search(
        self,
        query_word: str,
        top_k: int = 10,
        pos_filter: str | None = None,
    ) -> List[SearchResult]:
        """Search for the Top-K most similar words to a query word.

        Args:
            query_word: The query word (must exist in the vocabulary).
            top_k:      Maximum number of similar words to return (default 10).
            pos_filter: Restrict results to the given POS (for example, ``"NOUN"``).
                        ``None`` returns all POS tags.

        Returns:
            list[SearchResult]: Results sorted by similarity (descending).

        Raises:
            UnknownWordError: If ``query_word`` is not in the vocabulary.
            InvalidTopKError: If ``top_k`` is less than 1.
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
        """Compare the Top-K similar words between ``self`` and ``other``.

        Results from ``self.search()`` are stored in ``static_results`` and
        results from ``other.search()`` in ``contextual_results``.
        Example call: ``static_engine.compare("king", other=contextual_engine)``.

        Args:
            query_word: The query word.
            other:      The ``SimilarityEngine`` instance to compare against.
            top_k:      Maximum number of similar words per engine.

        Returns:
            ComparisonResult: Comparison result for both engines.

        Raises:
            UnknownWordError: If ``query_word`` is not in ``self`` 's vocabulary.
            InvalidTopKError: If ``top_k`` is less than 1.
            TypeError:        If ``other`` is not a ``SimilarityEngine``.
        """
        if not isinstance(other, SimilarityEngine):
            raise TypeError(
                f"other は SimilarityEngine 型である必要があります。"
                f"受け取った型: {type(other)}"
            )
        self._validate_top_k(top_k)

        # Use only the public methods of both engines (no dependence on internals).
        static_results: List[SearchResult] = self.search(query_word, top_k=top_k)
        contextual_results: List[SearchResult] = other.search(query_word, top_k=top_k)

        # Compute shared and unique words via set operations
        static_word_set: set[str] = {r.word for r in static_results}
        contextual_word_set: set[str] = {r.word for r in contextual_results}

        common_words: List[str] = sorted(static_word_set & contextual_word_set)
        static_only: List[str] = sorted(static_word_set - contextual_word_set)
        contextual_only: List[str] = sorted(contextual_word_set - static_word_set)

        # Compute rank / similarity differences for each shared word
        static_rank_map: Dict[str, int] = {r.word: r.rank for r in static_results}
        contextual_rank_map: Dict[str, int] = {r.word: r.rank for r in contextual_results}
        static_sim_map: Dict[str, float] = {r.word: r.similarity for r in static_results}
        contextual_sim_map: Dict[str, float] = {r.word: r.similarity for r in contextual_results}

        rank_diff: Dict[str, int] = {
            word: static_rank_map[word] - contextual_rank_map[word]
            for word in common_words
        }
        similarity_diff: Dict[str, float] = {
            word: static_sim_map[word] - contextual_sim_map[word]
            for word in common_words
        }

        logger.info(
            "compare 完了: query=%s, 共通=%d語, static固有=%d語, 文脈固有=%d語",
            query_word, len(common_words), len(static_only), len(contextual_only),
        )

        return ComparisonResult(
            query_word=query_word,
            static_results=static_results,
            contextual_results=contextual_results,
            common_words=common_words,
            static_only=static_only,
            contextual_only=contextual_only,
            rank_diff=rank_diff,
            similarity_diff=similarity_diff,
        )

    def get_distance_distribution(
        self,
        query_word: str,
    ) -> dict:
        """Return distribution statistics of cosine similarities between the query and the whole vocabulary.

        Computes similarities against the full vocabulary (N-1 entries,
        excluding the self-reference) in a single batch, and derives the
        distribution's mean, standard deviation, and Z-score.

        Z-score = (top1_similarity - mean) / std
        Indicates "how many standard deviations away from the mean the
        Top-1 similar word is." Larger values indicate that the top
        neighbors are more isolated within semantic space.

        Args:
            query_word: The query word.

        Returns:
            dict: A dictionary with the keys below.

            - "query_word"      (str):         The query word
            - "mean"            (float):       Mean cosine similarity over the full vocabulary
            - "std"             (float):       Standard deviation
            - "top1_similarity" (float):       Top-1 similarity score
            - "z_score"         (float):       Z-score of the Top-1 score
            - "histogram_data"  (list[float]): All N-1 similarity scores (for visualization)

        Raises:
            UnknownWordError: If ``query_word`` is not in the vocabulary.
        """
        query_idx: int = self.word_to_index(query_word)
        query_vec: np.ndarray = self._vectors[query_idx]

        # Compute similarities against the full vocabulary in a single batch
        # (delegated to DistanceMetrics).
        all_similarities: np.ndarray = self._metrics.cosine_similarity_batch(
            query_vec, self._vectors
        )

        # Build a mask that excludes the self-reference (similarity 1.0).
        mask: np.ndarray = np.ones(self._n_vocab, dtype=bool)
        mask[query_idx] = False
        sims_without_self: np.ndarray = all_similarities[mask]

        mean_sim: float = float(np.mean(sims_without_self))
        std_sim: float = float(np.std(sims_without_self))
        top1_sim: float = float(np.max(sims_without_self))

        # Avoid division by zero when std is zero (for example, when every vector is identical).
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
        """Convert a word to its vocabulary index.

        Args:
            word: The word to convert.

        Returns:
            int: Vocabulary index (0-based).

        Raises:
            UnknownWordError: If ``word`` is not in the vocabulary.
        """
        index: int | None = self._vocab.get(word)
        if index is None:
            raise UnknownWordError(
                f"'{word}' は語彙に存在しません。"
                f"（語彙サイズ: {self._n_vocab}）"
            )
        return index

    # -----------------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------------

    def _build_results(
        self,
        query_vec: np.ndarray,
        query_idx: int,
        top_k: int,
    ) -> List[SearchResult]:
        """Build a list of ``SearchResult`` objects from a query vector.

        Obtains Top-K indices and similarities via ``_search_single()``, then
        attaches the explanation dictionary built by ``_build_explanation()``
        to each result. ``pos_rank`` is initialized to 0 and set later by
        ``_assign_pos_ranks()``.

        Args:
            query_vec: Embedding vector of the query word, shape (D,).
            query_idx: Index of the query word (used to exclude self-reference).
            top_k:     Number of results to return.

        Returns:
            list[SearchResult]: List sorted by similarity (descending), with
            ``pos_rank`` left at 0.
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
                pos_rank=0,   # Filled in later by _assign_pos_ranks()
                explanation=explanation,
            ))

        return results

    def _search_single(
        self,
        query_vec: np.ndarray,
        query_idx: int,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute cosine similarities between the query and the whole matrix, returning the Top-K.

        Computation is delegated entirely to
        ``DistanceMetrics.cosine_similarity_batch()``.
        The self-reference (the query word itself) is replaced with ``-inf``
        to exclude it.
        If ``top_k`` exceeds the vocabulary size, it is clipped down.

        Optimization:
            ``np.argpartition`` narrows down the Top-K in O(N), and only
            those k entries are sorted in O(k log k). This is faster than
            sorting the entire vocabulary in O(N log N).

        Args:
            query_vec: The query vector, shape (D,).
            query_idx: Index of the query word (excluded from results).
            top_k:     Number of results to return.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - indices      shape (effective_k,): Indices sorted by descending similarity
                - similarities shape (effective_k,): Corresponding cosine similarities
        """
        # Compute similarities in a single batch (no external libraries
        # allowed; delegated to DistanceMetrics).
        # cosine_similarity_batch returns a fresh ndarray, so self._vectors is not modified.
        all_sims: np.ndarray = self._metrics.cosine_similarity_batch(
            query_vec, self._vectors
        )

        # Set the self-reference to -inf so it is excluded from Top-K.
        all_sims[query_idx] = -np.inf

        # Clip top_k to the effective vocabulary size (N-1, after self-exclusion).
        effective_k: int = min(top_k, self._n_vocab - 1)

        # Use argpartition to fetch Top-K indices in O(N) (order is unspecified).
        top_k_unordered: np.ndarray = np.argpartition(all_sims, -effective_k)[-effective_k:]

        # Sort within the Top-K by descending score: O(k log k).
        sorted_order: np.ndarray = np.argsort(all_sims[top_k_unordered])[::-1]
        top_k_indices: np.ndarray = top_k_unordered[sorted_order]
        top_k_sims: np.ndarray = all_sims[top_k_indices]

        return top_k_indices, top_k_sims

    def _assign_pos_ranks(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Assign within-POS rank (``pos_rank``) to each ``SearchResult``.

        Assumes ``results`` is already sorted by descending similarity.
        Appearance order within each POS group (i.e. similarity order) becomes
        ``pos_rank``. The original overall order of ``results`` is preserved.

        Args:
            results: List of ``SearchResult`` (descending by similarity).

        Returns:
            list[SearchResult]: The same list with ``pos_rank`` filled in.
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
        """Build the cosine-similarity explanation dictionary for two vectors.

        Delegates to ``DistanceMetrics.explain()``. The returned dictionary
        is used to render the "why this score" panel in the UI.

        Args:
            query_vec:  Query word vector, shape (D,).
            target_vec: Target word vector, shape (D,).

        Returns:
            dict: The dictionary returned by ``DistanceMetrics.explain()``.
                - "dot_product" (float): Dot product
                - "norm_a"      (float): L2 norm of the query vector
                - "norm_b"      (float): L2 norm of the target vector
                - "similarity"  (float): Cosine similarity
                - "formula"     (str):   String representation of the formula
        """
        return self._metrics.explain(query_vec, target_vec)

    def _validate_top_k(self, top_k: int) -> None:
        """Validate the ``top_k`` value.

        Args:
            top_k: Value to validate.

        Raises:
            InvalidTopKError: If ``top_k`` is less than 1.
        """
        if top_k < 1:
            raise InvalidTopKError(
                f"top_k は 1 以上である必要があります。受け取った値: {top_k}"
            )
