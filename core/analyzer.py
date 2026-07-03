"""
analyzer.py
===========
Statistical analysis of distance distributions.

Consumes the output of
``SimilarityEngine.get_distance_distribution()`` and provides:

- Attaching a Z-score to each ``SearchResult``
- Aggregating histogram bins (for visualization)
- Comparing the static vs. contextual distributions
- A neighborhood-stability score (Top-K overlap)

Does not depend on ``DistanceMetrics`` or ``SimilarityEngine``
(analysis only).
Uses numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from core.similarity_engine import ComparisonResult, SearchResult

logger = logging.getLogger(__name__)

# Verdict tiers for SimilarityVerdict, keyed by the fraction of the vocabulary
# that is at least as similar (``top_fraction``). Evaluated in ascending order;
# the first threshold the value falls under wins. Below all thresholds falls
# back to ``_VERDICT_FALLBACK``.
_VERDICT_TIERS: List[tuple[float, str]] = [
    (0.001, "unusually close"),
    (0.01, "very close"),
    (0.05, "notably close"),
    (0.25, "moderately close"),
    (0.50, "around the vocabulary average"),
]
_VERDICT_FALLBACK: str = "below the vocabulary average"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AnalyzerError(Exception):
    """Base exception class specific to Analyzer."""


class InsufficientDataError(AnalyzerError):
    """Raised when there is not enough data to compute statistics.

    Examples: ``histogram_data`` is empty, or ``std=0`` so the Z-score is undefined.
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DistributionStats:
    """Statistical summary of a distance distribution.

    Receives the output of ``get_distance_distribution()`` and augments
    it with additional statistics.

    Attributes:
        query_word:      The query word
        mean:            Mean cosine similarity over the full vocabulary
        std:             Standard deviation
        top1_similarity: Top-1 similarity score
        z_score:         Z-score of the Top-1 score = (top1 - mean) / std
        median:          Median
        q25:             First quartile (25th percentile)
        q75:             Third quartile (75th percentile)
        n_samples:       Sample count (vocabulary size - 1)
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
    """Result of histogram bin aggregation.

    Attributes:
        bin_edges: List of bin boundaries (length ``n_bins + 1``)
        counts:    List of per-bin frequencies (length ``n_bins``)
        n_bins:    Number of bins
        data_min:  Minimum value in the data
        data_max:  Maximum value in the data
    """

    bin_edges: List[float]
    counts: List[int]
    n_bins: int
    data_min: float
    data_max: float


@dataclass
class SimilarityVerdict:
    """Plain-language interpretation of a similarity score (roadmap FR-25).

    Turns a raw cosine similarity into an answer to "is this actually high?"
    by locating it within the whole-vocabulary distribution.

    Attributes:
        similarity:    The similarity score being interpreted.
        z_score:       Standard deviations above the vocabulary mean.
        top_fraction:  Fraction of the vocabulary at least this similar,
                       in [0.0, 1.0] (smaller = more of an outlier).
        label:         Short tier label (e.g. "very close").
        text:          Full one-line verdict ready for display.
    """

    similarity: float
    z_score: float
    top_fraction: float
    label: str
    text: str


@dataclass
class DistributionComparison:
    """Result of comparing the static vs. contextual distributions.

    Attributes:
        query_word:       The query word
        static_stats:     Distribution statistics from the static engine
        contextual_stats: Distribution statistics from the contextual engine
        mean_diff:        Difference in mean cosine similarity (static - contextual)
        std_diff:         Difference in standard deviation (static - contextual)
        z_score_diff:     Difference in Z-score (static - contextual)
                          Positive → static has a more "outstanding" Top-1
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
    """Statistical analysis of distance distributions.

    All methods are staticmethods, so no instantiation is required.
    Uses numpy only, with every calculation implemented from scratch.

    Example usage::

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
        """Augment the output of ``get_distance_distribution()`` with extra statistics.

        Computes the median and quartiles in addition to the existing fields
        and returns a ``DistributionStats`` instance. This makes it easier
        to see the distribution's asymmetry and outliers.

        Args:
            distribution: Return value of
                ``SimilarityEngine.get_distance_distribution()``.
                Required keys: ``query_word``, ``mean``, ``std``,
                ``top1_similarity``, ``z_score``, ``histogram_data``.

        Returns:
            DistributionStats: Distribution summary with extra statistics.

        Raises:
            KeyError:              If a required key is missing.
            InsufficientDataError: If ``histogram_data`` is empty.
        """
        required_keys = {"query_word", "mean", "std", "top1_similarity",
                         "z_score", "histogram_data"}
        missing = required_keys - distribution.keys()
        if missing:
            raise KeyError(f"distribution is missing required keys: {missing}")

        data: np.ndarray = np.array(distribution["histogram_data"])
        if data.size == 0:
            raise InsufficientDataError(
                "histogram_data is empty; cannot compute statistics"
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
    def interpret_similarity(
        similarity: float,
        distribution: dict,
    ) -> SimilarityVerdict:
        """Interpret a similarity score in plain language (roadmap FR-25).

        Answers "is this similarity actually high?" by placing the score
        within the whole-vocabulary distribution: it reports how large a
        fraction of the vocabulary is at least this similar (``top_fraction``)
        and how many standard deviations above the mean it sits (``z_score``),
        then assigns a short tier label.

        Args:
            similarity:   The cosine similarity to interpret.
            distribution: Return value of
                ``SimilarityEngine.get_distance_distribution()``.
                Required keys: ``mean``, ``std``, ``histogram_data``.

        Returns:
            SimilarityVerdict: The interpretation, including a ready-to-display
            ``text`` such as ``"Top 0.3% of 40,032 words (Z=+4.1) — very close"``.

        Raises:
            KeyError:              If a required key is missing.
            InsufficientDataError: If ``histogram_data`` is empty.
        """
        required_keys = {"mean", "std", "histogram_data"}
        missing = required_keys - distribution.keys()
        if missing:
            raise KeyError(f"distribution is missing required keys: {missing}")

        data: np.ndarray = np.asarray(distribution["histogram_data"], dtype=float)
        if data.size == 0:
            raise InsufficientDataError(
                "histogram_data is empty; cannot interpret a similarity"
            )

        mean: float = distribution["mean"]
        std: float = distribution["std"]
        z_score: float = (similarity - mean) / std if std > 0.0 else 0.0

        # Fraction of the vocabulary at least this similar (ties included).
        top_fraction: float = float(np.count_nonzero(data >= similarity)) / data.size

        label: str = _VERDICT_FALLBACK
        for threshold, tier_label in _VERDICT_TIERS:
            if top_fraction <= threshold:
                label = tier_label
                break

        top_pct: str = "<0.1%" if top_fraction < 0.001 else f"{top_fraction * 100:.1f}%"
        text: str = (
            f"Top {top_pct} of {data.size:,} words (Z={z_score:+.1f}) — {label}"
        )

        logger.debug(
            "interpret_similarity: sim=%.4f, top_fraction=%.5f, z=%.2f, label=%s",
            similarity, top_fraction, z_score, label,
        )

        return SimilarityVerdict(
            similarity=float(similarity),
            z_score=z_score,
            top_fraction=top_fraction,
            label=label,
            text=text,
        )

    @staticmethod
    def histogram(
        data: List[float],
        n_bins: int = 50,
    ) -> HistogramData:
        """Aggregate similarity data into histogram bins.

        Uses ``numpy.histogram`` to return bin counts, edges, and totals.
        The result can be fed directly to a Streamlit bar chart or
        ``st.bar_chart``.

        Args:
            data:   List of similarity scores (the ``histogram_data`` value).
            n_bins: Number of histogram bins (default 50).

        Returns:
            HistogramData: Bin aggregation result.

        Raises:
            InsufficientDataError: If ``data`` is empty.
            ValueError:            If ``n_bins`` is less than 1.
        """
        if not data:
            raise InsufficientDataError(
                "data is empty; cannot build a histogram"
            )
        if n_bins < 1:
            raise ValueError(
                f"n_bins must be at least 1. Received: {n_bins}"
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
        """Return a list of dicts that augments each ``SearchResult`` with its Z-score.

        Converts each result's similarity score into a Z-score against the
        full-vocabulary distribution. The Z-score indicates "how many
        standard deviations away from the mean this result is."

        Formula:
            z = (similarity - mean) / std

        When ``std == 0`` (for example, every vector is identical), the
        Z-score is returned as 0.0.

        Args:
            results:      List of ``SearchResult`` (descending by similarity).
            distribution: Return value of
                ``SimilarityEngine.get_distance_distribution()``.

        Returns:
            List[Dict]: List of dictionaries with the keys below.
                - "word"        (str):   Target word
                - "rank"        (int):   Overall rank
                - "similarity"  (float): Cosine similarity
                - "pos_tag"     (str):   POS label
                - "pos_rank"    (int):   Within-POS rank
                - "z_score"     (float): Z-score against the distribution
                - "explanation" (dict):  Distance computation breakdown

        Raises:
            KeyError: If ``distribution`` is missing ``mean`` or ``std``.
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
            "attach_z_scores: attached Z-score to %d entries (mean=%.4f, std=%.4f)",
            len(scored), mean, std,
        )
        return scored

    @staticmethod
    def compare_distributions(
        query_word: str,
        static_dist: dict,
        contextual_dist: dict,
    ) -> DistributionComparison:
        """Compare the static and contextual distance distributions.

        Computes the differences between the two models' ``mean`` / ``std`` /
        ``z_score``.
        Interpretation of the diffs:
            ``mean_diff > 0``    → static tends to have overall higher similarities
            ``z_score_diff > 0`` → static's Top-1 stands out more from the distribution

        Args:
            query_word:      The query word.
            static_dist:     Return value of ``get_distance_distribution()`` from the static engine.
            contextual_dist: Return value of ``get_distance_distribution()`` from the contextual engine.

        Returns:
            DistributionComparison: Comparison result for both distributions.

        Raises:
            InsufficientDataError: If either ``histogram_data`` is empty.
            KeyError:              If a required key is missing.
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
        """Compute the neighborhood-stability score for Top-K results.

        Returns the overlap rate (Jaccard coefficient) between the
        static and contextual Top-K results. A higher value indicates
        that both models share the same neighbors (a stable semantic
        neighborhood).

        Formula:
            stability = |static ∩ contextual| / |static ∪ contextual|

        Args:
            static_results:     Top-K results from the static engine.
            contextual_results: Top-K results from the contextual engine.

        Returns:
            float: Jaccard coefficient, in [0.0, 1.0].
                   1.0 = perfect agreement (both models return the same neighbors)
                   0.0 = no overlap

        Raises:
            ValueError: If either ``results`` argument is empty.
        """
        if not static_results:
            raise ValueError("static_results is empty")
        if not contextual_results:
            raise ValueError("contextual_results is empty")

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
