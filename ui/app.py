"""
ui/app.py
=========
Explainable Semantic Space Explorer — Streamlit UI

単語埋め込み空間を可視化・分析し、「なぜその単語が近いのか」を説明するUI。

設計方針:
    - このファイルはロジックを持たない。core/ と transforms/ を呼ぶだけ。
    - 計算・変換・判定はすべて core/transforms 層に委譲する。
    - @st.cache_resource でエンジンを1回だけ初期化する（大規模ベクトルの再読み込み防止）。

タブ構成:
    Tab 1: 類似語検索           — Top-K 類似語・品詞分布・コサイン計算式の表示
    Tab 2: 静的 vs 文脈 比較   — 2モデルの比較・順位差・近傍安定性
    Tab 3: 距離分布分析         — ヒストグラム・Z-score・分布統計
    Tab 4: 投影・クラスタ       — PCA/UMAP 2D散布図・クラスタ色分け
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from transforms.clustering import KMeansClusterer
from transforms.projection import Projector
from core.analyzer import Analyzer
from core.distance_metrics import DistanceMetrics
from core.embedding_loader import EmbeddingLoader, EmbeddingLoaderError
from core.pos_filter import POSFilter
from core.similarity_engine import (
    SearchResult,
    SimilarityEngine,
    UnknownWordError,
)

logger = logging.getLogger(__name__)

# ---------- ページ設定 ----------

st.set_page_config(
    page_title="Explainable Semantic Space Explorer",
    page_icon="🔍",
    layout="wide",
)


# ---------- キャッシュ付きリソース初期化 ----------


@st.cache_resource(show_spinner="埋め込みベクトルを読み込み中...")
def load_all_engines() -> tuple[EmbeddingLoader, SimilarityEngine, SimilarityEngine]:
    """EmbeddingLoader と 2 つの SimilarityEngine を初期化する。

    @st.cache_resource により、アプリ起動後に1回だけ実行される。
    83,823 語彙 × 300/384 次元のベクトルをメモリに保持するため、
    再初期化を避けることで応答速度を確保する。

    Returns:
        tuple: (loader, static_engine, contextual_engine)

    Raises:
        EmbeddingLoaderError: 資産ファイルが見つからない、または整合エラーの場合。
    """
    data_root = Path("data")
    loader = EmbeddingLoader(data_root)
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

    logger.info("全エンジン初期化完了")
    return loader, static_engine, contextual_engine


# ---------- ヘルパー: SearchResult → DataFrame ----------


def results_to_df(scored: List[dict]) -> pd.DataFrame:
    """attach_z_scores() の出力を st.dataframe 用 DataFrame に変換する。

    Args:
        scored: Analyzer.attach_z_scores() の返り値リスト。

    Returns:
        pd.DataFrame: 表示用 DataFrame。
    """
    rows = []
    for item in scored:
        rows.append(
            {
                "順位": item["rank"],
                "単語": item["word"],
                "類似度": round(item["similarity"], 4),
                "品詞": item["pos_tag"],
                "品詞内順位": item["pos_rank"],
                "Z-score": round(item["z_score"], 3),
            }
        )
    return pd.DataFrame(rows)


# ---------- メイン ----------


def main() -> None:
    """アプリのエントリポイント。サイドバーと4タブを構築する。"""

    st.title("Explainable Semantic Space Explorer")
    st.caption(
        "静的埋め込み（Word2Vec）と文脈埋め込み（SBERT）の差異を数値で説明するツール"
    )

    # --- エンジン読み込み ---
    try:
        loader, static_engine, contextual_engine = load_all_engines()
    except EmbeddingLoaderError as e:
        st.error(f"資産ファイルの読み込みに失敗しました: {e}")
        st.stop()

    # 品詞ラベルを動的取得（POSFilter.group_by_pos は SearchResult が必要なため直接取得）
    unique_pos: list[str] = sorted(set(loader.pos.tolist()))

    # ---------- サイドバー ----------
    with st.sidebar:
        st.header("検索設定")

        query_word: str = st.text_input(
            "クエリ単語",
            value="king",
            help="語彙内に存在する英単語を入力してください",
        )

        top_k: int = st.slider(
            "Top-K（表示件数）",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

        pos_options = ["ALL"] + unique_pos
        pos_selected: str = st.selectbox("品詞フィルター", pos_options, index=0)
        pos_filter: str | None = None if pos_selected == "ALL" else pos_selected

        st.divider()
        st.header("投影・クラスタ設定")

        projection_method: str = st.radio(
            "投影手法",
            options=["pca", "umap"],
            format_func=lambda x: x.upper(),
            horizontal=True,
        )

        n_clusters: int = st.slider(
            "クラスタ数",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
        )

    # --- クエリが空の場合は待機 ---
    if not query_word.strip():
        st.info("サイドバーからクエリ単語を入力してください。")
        return

    query_word = query_word.strip().lower()

    # --- 語彙チェック ---
    if query_word not in loader.vocab:
        st.error(
            f"「{query_word}」は語彙に存在しません。別の単語を試してください。"
        )
        return

    # ---------- 4タブ ----------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["類似語検索", "静的 vs 文脈 比較", "距離分布分析", "投影・クラスタ"]
    )

    # =========================================================
    # Tab 1: 類似語検索
    # =========================================================
    with tab1:
        st.subheader(f"「{query_word}」の類似語 Top-{top_k}")

        # core: 類似度検索 + 分布取得 + Z-score 付与
        try:
            static_results: list[SearchResult] = static_engine.search(
                query_word, top_k=top_k, pos_filter=pos_filter
            )
        except UnknownWordError as e:
            st.error(str(e))
            return

        static_dist = static_engine.get_distance_distribution(query_word)
        scored = Analyzer.attach_z_scores(static_results, static_dist)

        # 結果テーブル
        df = results_to_df(scored)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # コサイン計算式（先頭結果）
        if scored:
            with st.expander("計算式の内訳（Top-1）を見る"):
                exp = scored[0]["explanation"]
                st.code(exp["formula"], language=None)
                col1, col2, col3 = st.columns(3)
                col1.metric("内積 dot(a,b)", f"{exp['dot_product']:.4f}")
                col2.metric("‖query‖", f"{exp['norm_a']:.4f}")
                col3.metric("‖target‖", f"{exp['norm_b']:.4f}")

        st.divider()

        # core: 品詞分布
        pos_dist: dict[str, int] = POSFilter.pos_distribution(static_results)
        if pos_dist:
            st.subheader("品詞分布")
            pos_df = pd.DataFrame(
                [{"品詞": k, "件数": v} for k, v in pos_dist.items()]
            )
            bar = (
                alt.Chart(pos_df)
                .mark_bar()
                .encode(
                    x=alt.X("品詞:N", sort="-y"),
                    y=alt.Y("件数:Q"),
                    tooltip=["品詞", "件数"],
                )
                .properties(height=200)
            )
            st.altair_chart(bar, use_container_width=True)

        # core: 異品詞率
        query_pos: str = loader.pos[loader.vocab[query_word]]
        hetero: float = POSFilter.heterogeneity_rate(static_results, query_pos)
        st.metric(
            "異品詞率（クエリと異なる品詞の割合）",
            f"{hetero:.1%}",
            help="値が高いほど品詞を超えた類似語が多い",
        )

    # =========================================================
    # Tab 2: 静的 vs 文脈 比較
    # =========================================================
    with tab2:
        st.subheader(f"「{query_word}」— 静的 (Word2Vec) vs 文脈 (SBERT) 比較")

        # core: 比較結果取得
        try:
            comparison = static_engine.compare(
                query_word, other=contextual_engine, top_k=top_k
            )
        except UnknownWordError as e:
            st.error(str(e))
            return

        # core: 近傍安定性スコア
        stability: float = Analyzer.neighborhood_stability(
            comparison.static_results, comparison.contextual_results
        )

        # 集計メトリクス
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("共通語数", len(comparison.common_words))
        m2.metric("静的 固有語数", len(comparison.static_only))
        m3.metric("文脈 固有語数", len(comparison.contextual_only))
        m4.metric(
            "近傍安定性（Jaccard）",
            f"{stability:.3f}",
            help="1.0 = 完全一致、0.0 = 完全不一致",
        )

        st.divider()

        # 2列: 静的 Top-K / 文脈 Top-K
        col_s, col_b = st.columns(2)

        with col_s:
            st.write("**静的 (Word2Vec) Top-K**")
            static_scored = Analyzer.attach_z_scores(
                comparison.static_results,
                static_engine.get_distance_distribution(query_word),
            )
            st.dataframe(results_to_df(static_scored), hide_index=True)

        with col_b:
            st.write("**文脈 (SBERT) Top-K**")
            contextual_scored = Analyzer.attach_z_scores(
                comparison.contextual_results,
                contextual_engine.get_distance_distribution(query_word),
            )
            st.dataframe(results_to_df(contextual_scored), hide_index=True)

        st.divider()

        # 順位差テーブル（両モデルに共通する単語）
        if comparison.rank_diff:
            st.subheader("順位差（共通語）")
            rd_rows = [
                {"単語": w, "順位差 (静的 − 文脈)": d}
                for w, d in sorted(
                    comparison.rank_diff.items(), key=lambda x: abs(x[1]), reverse=True
                )
            ]
            st.dataframe(pd.DataFrame(rd_rows), hide_index=True, use_container_width=True)

        # 固有語バッジ
        if comparison.static_only:
            st.write("**静的 固有語:**", "  ".join(f"`{w}`" for w in comparison.static_only))
        if comparison.contextual_only:
            st.write("**文脈 固有語:**", "  ".join(f"`{w}`" for w in comparison.contextual_only))

    # =========================================================
    # Tab 3: 距離分布分析
    # =========================================================
    with tab3:
        st.subheader(f"「{query_word}」の距離分布分析")

        # core: 分布取得 + 統計拡張 + 比較
        static_dist = static_engine.get_distance_distribution(query_word)
        contextual_dist = contextual_engine.get_distance_distribution(query_word)

        static_stats = Analyzer.enrich_distribution(static_dist)
        contextual_stats = Analyzer.enrich_distribution(contextual_dist)
        dist_cmp = Analyzer.compare_distributions(query_word, static_dist, contextual_dist)

        # モデル別メトリクス
        col_s, col_b = st.columns(2)
        with col_s:
            st.write("**静的 (Word2Vec)**")
            st.metric("平均類似度", f"{static_stats.mean:.4f}")
            st.metric("標準偏差", f"{static_stats.std:.4f}")
            st.metric("Top-1 類似度", f"{static_stats.top1_similarity:.4f}")
            st.metric("Z-score", f"{static_stats.z_score:.3f}")
            st.metric("中央値", f"{static_stats.median:.4f}")

        with col_b:
            st.write("**文脈 (SBERT)**")
            st.metric("平均類似度", f"{contextual_stats.mean:.4f}")
            st.metric("標準偏差", f"{contextual_stats.std:.4f}")
            st.metric("Top-1 類似度", f"{contextual_stats.top1_similarity:.4f}")
            st.metric("Z-score", f"{contextual_stats.z_score:.3f}")
            st.metric("中央値", f"{contextual_stats.median:.4f}")

        st.divider()

        # ヒストグラム（静的）
        st.subheader("類似度分布ヒストグラム（静的）")

        hist_data = Analyzer.histogram(static_dist["histogram_data"], n_bins=50)
        bin_centers = [
            (hist_data.bin_edges[i] + hist_data.bin_edges[i + 1]) / 2
            for i in range(len(hist_data.counts))
        ]
        hist_df = pd.DataFrame(
            {
                "類似度": bin_centers,
                "頻度": hist_data.counts,
            }
        )

        # Top-1 の縦線（rule）
        rule_df = pd.DataFrame({"x": [static_dist["top1_similarity"]]})
        base = alt.Chart(hist_df).mark_bar(opacity=0.7).encode(
            x=alt.X("類似度:Q", bin=False, title="コサイン類似度"),
            y=alt.Y("頻度:Q"),
            tooltip=["類似度", "頻度"],
        )
        rule = (
            alt.Chart(rule_df)
            .mark_rule(color="red", strokeDash=[4, 4], strokeWidth=2)
            .encode(x="x:Q")
        )
        st.altair_chart((base + rule).properties(height=250), use_container_width=True)
        st.caption(f"赤い点線 = Top-1 類似度 ({static_dist['top1_similarity']:.4f})")

        # 分布比較メトリクス
        st.divider()
        st.subheader("静的 vs 文脈 分布比較")
        c1, c2, c3 = st.columns(3)
        c1.metric("平均差（静的 − 文脈）", f"{dist_cmp.mean_diff:.4f}")
        c2.metric("標準偏差差", f"{dist_cmp.std_diff:.4f}")
        c3.metric("Z-score 差", f"{dist_cmp.z_score_diff:.3f}")

    # =========================================================
    # Tab 4: 投影・クラスタ
    # =========================================================
    with tab4:
        st.subheader(f"「{query_word}」周辺の 2D 投影（{projection_method.upper()}）")

        # クエリ周辺 Top-K インデックスを取得（core: search）
        try:
            proj_results: list[SearchResult] = static_engine.search(
                query_word, top_k=top_k
            )
        except UnknownWordError as e:
            st.error(str(e))
            return

        query_idx: int = loader.vocab[query_word]
        neighbor_indices: list[int] = [r.index for r in proj_results]
        # クエリ自身が neighbor に含まれていない場合のみ追加
        if query_idx not in neighbor_indices:
            target_indices = neighbor_indices + [query_idx]
        else:
            target_indices = neighbor_indices

        target_words: list[str] = [
            list(loader.vocab.keys())[list(loader.vocab.values()).index(i)]
            for i in target_indices
        ]
        # 高速化: 逆引き辞書を作成
        index_to_word: dict[int, str] = {v: k for k, v in loader.vocab.items()}
        target_words = [index_to_word[i] for i in target_indices]

        target_vectors: np.ndarray = loader.static_vectors[target_indices]

        # analysis: クラスタリング（クエリ周辺のみ）
        n_clust_actual = min(n_clusters, len(target_indices))
        clusterer = KMeansClusterer(n_clusters=n_clust_actual, seed=42)
        cluster_result = clusterer.fit(target_vectors)

        # analysis: 投影（PCA または UMAP）
        projector = Projector(method=projection_method, seed=42)
        proj_result = projector.fit_transform(target_vectors)
        proj_result = projector.attach_clusters(proj_result, cluster_result.labels)

        # 類似度マップ（クエリとの類似度）
        sim_map: dict[str, float] = {r.word: r.similarity for r in proj_results}
        sim_map[query_word] = 1.0  # クエリ自身

        # 散布図データ
        coords = proj_result.coords_2d
        scatter_df = pd.DataFrame(
            {
                "x": coords[:, 0].tolist(),
                "y": coords[:, 1].tolist(),
                "単語": target_words,
                "クラスタ": [str(c) for c in proj_result.cluster_labels.tolist()],
                "類似度": [round(sim_map.get(w, 0.0), 4) for w in target_words],
                "クエリ": ["✕ クエリ" if w == query_word else "近傍語" for w in target_words],
            }
        )

        # Altair 散布図
        axis_label = "PC" if projection_method == "pca" else "UMAP"
        scatter = (
            alt.Chart(scatter_df)
            .mark_point(size=120, filled=True)
            .encode(
                x=alt.X("x:Q", title=f"{axis_label}1"),
                y=alt.Y("y:Q", title=f"{axis_label}2"),
                color=alt.Color("クラスタ:N", title="クラスタ"),
                shape=alt.Shape(
                    "クエリ:N",
                    scale=alt.Scale(
                        domain=["✕ クエリ", "近傍語"],
                        range=["cross", "circle"],
                    ),
                ),
                tooltip=["単語", "類似度", "クラスタ", "クエリ"],
            )
            .properties(height=400)
            .interactive()
        )

        # 単語ラベル
        text = (
            alt.Chart(scatter_df)
            .mark_text(dy=-10, fontSize=11)
            .encode(
                x="x:Q",
                y="y:Q",
                text="単語:N",
                color=alt.Color("クラスタ:N"),
            )
        )

        st.altair_chart((scatter + text).properties(height=420), use_container_width=True)

        # PCA 寄与率
        if projection_method == "pca" and proj_result.explained_variance:
            c1, c2 = st.columns(2)
            c1.metric(
                "第1主成分 寄与率",
                f"{proj_result.explained_variance[0]:.1%}",
            )
            c2.metric(
                "第2主成分 寄与率",
                f"{proj_result.explained_variance[1]:.1%}",
            )

        st.divider()

        # クラスタ別単語一覧
        st.subheader("クラスタ別の単語グループ")
        cluster_groups: dict[int, list[str]] = {}
        for word, label in zip(target_words, proj_result.cluster_labels.tolist()):
            cluster_groups.setdefault(int(label), []).append(word)

        cluster_df_rows = []
        for cid in sorted(cluster_groups.keys()):
            cluster_df_rows.append(
                {
                    "クラスタID": cid,
                    "単語数": len(cluster_groups[cid]),
                    "単語": " / ".join(cluster_groups[cid]),
                }
            )
        st.dataframe(
            pd.DataFrame(cluster_df_rows),
            hide_index=True,
            use_container_width=True,
        )


# ---------- エントリポイント ----------

if __name__ == "__main__":
    main()
