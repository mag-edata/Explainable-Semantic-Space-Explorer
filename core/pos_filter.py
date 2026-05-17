"""
pos_filter.py
=============
Part-of-Speech (POS) filtering and per-POS statistics.

Extends the basic functionality of ``SimilarityEngine._assign_pos_ranks``
and provides the following features:

- Filtering by POS
- Grouping by POS
- Aggregating the POS distribution
- Heterogeneity rate (proportion of words with a different POS from the query)
- POS-internal ranking (as a standalone utility)

Does not depend on ``SimilarityEngine``.
Depends on the ``SearchResult`` dataclass.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from core.similarity_engine import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class POSFilterError(Exception):
    """Base exception class specific to POSFilter."""


class UnknownPOSTagError(POSFilterError):
    """Raised when the requested POS tag is not present in the results."""


# ---------------------------------------------------------------------------
# POSFilter
# ---------------------------------------------------------------------------

class POSFilter:
    """Utility class for POS filtering and per-POS statistics.

    All methods are staticmethods; instantiation is not required.
    Accepts a list of ``SearchResult`` objects and returns POS-related
    analyses or filtered results.

    Example usage::

        results = engine.search("king", top_k=20)

        noun_results = POSFilter.filter(results, "NOUN")
        groups       = POSFilter.group_by_pos(results)
        dist         = POSFilter.pos_distribution(results)
        rate         = POSFilter.heterogeneity_rate(results, query_pos="NOUN")
    """

    @staticmethod
    def filter(
        results: List[SearchResult],
        pos_tag: str,
    ) -> List[SearchResult]:
        """Return only the ``SearchResult`` entries with the specified POS.

        The overall ``rank`` is preserved as-is after filtering.
        ``pos_rank`` can be re-numbered with ``assign_pos_ranks()``.

        Args:
            results: List of ``SearchResult``.
            pos_tag: POS label to filter on (for example, ``"NOUN"``, ``"VERB"``).

        Returns:
            List[SearchResult]: List restricted to the given POS, with order preserved.

        Raises:
            ValueError:         If ``pos_tag`` is an empty string.
            UnknownPOSTagError: If the requested POS is not in ``results``.
        """
        if not pos_tag:
            raise ValueError("pos_tag は空文字にできません")

        filtered = [r for r in results if r.pos_tag == pos_tag]

        if not filtered:
            available = sorted({r.pos_tag for r in results})
            raise UnknownPOSTagError(
                f"品詞 '{pos_tag}' の結果が見つかりません。"
                f"利用可能な品詞: {available}"
            )

        logger.debug(
            "filter: pos_tag=%s, %d / %d 件",
            pos_tag, len(filtered), len(results),
        )
        return filtered

    @staticmethod
    def group_by_pos(
        results: List[SearchResult],
    ) -> Dict[str, List[SearchResult]]:
        """Group ``SearchResult`` entries by POS.

        Order within each group preserves the original order of ``results``
        (descending similarity).

        Args:
            results: List of ``SearchResult``.

        Returns:
            Dict[str, List[SearchResult]]:
                Keys are POS labels, values are the ``SearchResult`` list for that POS.
                Example: ``{"NOUN": [...], "VERB": [...]}``
        """
        groups: Dict[str, List[SearchResult]] = defaultdict(list)

        for result in results:
            groups[result.pos_tag].append(result)

        logger.debug(
            "group_by_pos: %d 品詞グループ: %s",
            len(groups), list(groups.keys()),
        )
        return dict(groups)

    @staticmethod
    def pos_distribution(
        results: List[SearchResult],
    ) -> Dict[str, int]:
        """Count occurrences per POS.

        Indicates how many entries of each POS are included in the Top-K
        results. Useful for analyzing "which POS dominates the semantic
        neighborhood."

        Args:
            results: List of ``SearchResult``.

        Returns:
            Dict[str, int]: Keys are POS labels, values are counts.
                            Sorted by count, descending.
        """
        counter: Dict[str, int] = {}
        for result in results:
            counter[result.pos_tag] = counter.get(result.pos_tag, 0) + 1

        # Sort by count, descending
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def heterogeneity_rate(
        results: List[SearchResult],
        query_pos: str,
    ) -> float:
        """Return the proportion of results whose POS differs from the query's POS.

        A higher value means that semantically close words span multiple
        POS categories, indicating low POS-dependence of the model.

        Formula:
            heterogeneity = (count of results whose POS ≠ query_pos) / (total count)

        Args:
            results:   List of ``SearchResult``.
            query_pos: POS label of the query word.

        Returns:
            float: Heterogeneity rate, in [0.0, 1.0].
                   0.0 = every result has the same POS (fully POS-dependent)
                   1.0 = every result has a different POS

        Raises:
            ValueError: If ``results`` is empty, or ``query_pos`` is an empty string.
        """
        if not results:
            raise ValueError("results が空です")
        if not query_pos:
            raise ValueError("query_pos は空文字にできません")

        different_count = sum(1 for r in results if r.pos_tag != query_pos)
        rate = different_count / len(results)

        logger.debug(
            "heterogeneity_rate: query_pos=%s, rate=%.3f (%d/%d)",
            query_pos, rate, different_count, len(results),
        )
        return rate

    @staticmethod
    def assign_pos_ranks(
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Assign within-POS rank (``pos_rank``) to each ``SearchResult``.

        Assumes ``results`` is already sorted by descending similarity.
        Appearance order within each POS group becomes ``pos_rank``.
        The original order of ``results`` is preserved.

        Provides the same logic as ``SimilarityEngine._assign_pos_ranks``
        as a standalone utility.

        Args:
            results: List of ``SearchResult`` (descending by similarity).

        Returns:
            List[SearchResult]: The same list with ``pos_rank`` filled in.
        """
        pos_counter: Dict[str, int] = {}

        for result in results:
            pos = result.pos_tag
            pos_counter[pos] = pos_counter.get(pos, 0) + 1
            result.pos_rank = pos_counter[pos]

        return results

    @staticmethod
    def top_pos(
        results: List[SearchResult],
        n: int = 3,
    ) -> List[str]:
        """Return the top ``n`` POS labels by frequency.

        Args:
            results: List of ``SearchResult``.
            n:       Maximum number of POS labels to return (default 3).

        Returns:
            List[str]: POS labels sorted by frequency (up to ``n`` entries).

        Raises:
            ValueError: If ``n`` is less than 1.
        """
        if n < 1:
            raise ValueError(f"n は 1 以上である必要があります。受け取った値: {n}")

        dist = POSFilter.pos_distribution(results)
        return list(dist.keys())[:n]
