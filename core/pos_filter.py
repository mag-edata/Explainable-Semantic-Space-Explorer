"""
pos_filter.py
=============
品詞（Part-of-Speech）フィルタリングと品詞内統計。

SimilarityEngine._assign_pos_ranks の基本機能を拡張し、
以下の機能を提供する:

- 品詞によるフィルタリング
- 品詞ごとのグループ化
- 品詞分布の集計
- 異品詞率（クエリ単語と異なる品詞の割合）
- 品詞内での順位付け（独立ユーティリティとして）

SimilarityEngine には依存しない。
SearchResult dataclass に依存する。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from core.similarity_engine import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# カスタム例外
# ---------------------------------------------------------------------------

class POSFilterError(Exception):
    """POSFilter 固有の例外基底クラス。"""


class UnknownPOSTagError(POSFilterError):
    """指定した品詞タグが結果に存在しない場合の例外。"""


# ---------------------------------------------------------------------------
# POSFilter
# ---------------------------------------------------------------------------

class POSFilter:
    """品詞フィルタリングと品詞内統計のユーティリティクラス。

    全メソッドは staticmethod。インスタンス化不要で使用可能。
    SearchResult のリストを入力として受け取り、
    品詞に関する分析・フィルタリング結果を返す。

    使用例::

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
        """指定した品詞の SearchResult のみを返す。

        全体順位（rank）はフィルタ後も元の値を保持する。
        品詞内順位（pos_rank）は assign_pos_ranks() で再付番できる。

        Args:
            results: SearchResult のリスト。
            pos_tag: フィルタリングする品詞ラベル（例: "NOUN", "VERB"）。

        Returns:
            List[SearchResult]: 指定品詞のみのリスト（元の順序を保持）。

        Raises:
            ValueError:       pos_tag が空文字の場合。
            UnknownPOSTagError: 指定した品詞が results に存在しない場合。
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
        """SearchResult を品詞ごとにグループ化する。

        各グループ内の順序は元の results の順序（類似度降順）を保持する。

        Args:
            results: SearchResult のリスト。

        Returns:
            Dict[str, List[SearchResult]]:
                キーが品詞ラベル、値がその品詞の SearchResult リスト。
                例: {"NOUN": [...], "VERB": [...]}
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
        """品詞ごとの出現数を集計する。

        Top-K 結果の中に各品詞が何件含まれるかを示す。
        「どの品詞の単語が意味的に近いか」の分析に使用する。

        Args:
            results: SearchResult のリスト。

        Returns:
            Dict[str, int]: キーが品詞ラベル、値が出現数。
                            出現数の多い順にソート済み。
        """
        counter: Dict[str, int] = {}
        for result in results:
            counter[result.pos_tag] = counter.get(result.pos_tag, 0) + 1

        # 出現数の多い順にソート
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def heterogeneity_rate(
        results: List[SearchResult],
        query_pos: str,
    ) -> float:
        """クエリ単語の品詞と異なる品詞の割合（異品詞率）を返す。

        異品詞率が高いほど、意味的に近い単語が品詞をまたいで分布している。
        これはモデルの品詞依存性の低さを示す。

        計算式:
            異品詞率 = (query_pos と異なる品詞の件数) / (全件数)

        Args:
            results:   SearchResult のリスト。
            query_pos: クエリ単語の品詞ラベル。

        Returns:
            float: 異品詞率。範囲: [0.0, 1.0]。
                   0.0 = 全結果が同じ品詞（完全に品詞依存）
                   1.0 = 全結果が異なる品詞

        Raises:
            ValueError: results が空の場合、または query_pos が空文字の場合。
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
        """SearchResult リストに同品詞内順位（pos_rank）を付与する。

        results は類似度降順でソート済みであることを前提とする。
        同品詞グループ内の登場順がそのまま pos_rank になる。
        元の results の順序は変えない。

        SimilarityEngine._assign_pos_ranks と同じロジックを
        独立ユーティリティとして提供する。

        Args:
            results: SearchResult のリスト（類似度降順）。

        Returns:
            List[SearchResult]: pos_rank が設定されたリスト（元の順序を保持）。
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
        """出現数の多い品詞を上位 n 件返す。

        Args:
            results: SearchResult のリスト。
            n:       返す品詞の最大件数（デフォルト 3）。

        Returns:
            List[str]: 出現数の多い順の品詞ラベルリスト（最大 n 件）。

        Raises:
            ValueError: n が 1 未満の場合。
        """
        if n < 1:
            raise ValueError(f"n は 1 以上である必要があります。受け取った値: {n}")

        dist = POSFilter.pos_distribution(results)
        return list(dist.keys())[:n]
