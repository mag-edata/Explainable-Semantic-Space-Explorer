"""
ui/app.py
=========
Explainable Semantic Space Explorer — Streamlit UI

A UI that visualizes and analyzes word embedding spaces and explains
"why words are close to each other."

Design policy:
    - This file holds no business logic. It only calls into ``core/``
      and ``transforms/``.
    - All computation, transformation, and decision-making is delegated
      to the core / transforms layers.
    - ``@st.cache_resource`` initializes the engines exactly once
      (preventing reloads of the large vector matrices).

Tab layout:
    Tab 1: Similar-word search — Top-K similar words, POS distribution,
                                 and the cosine formula display
    Tab 2: Static vs. contextual comparison — Compare both models,
                                               rank differences,
                                               neighborhood stability
    Tab 3: Distance distribution analysis — Histogram, Z-score,
                                            distribution statistics
    Tab 4: Projection / clustering — PCA / UMAP 2D scatter plot,
                                     cluster colorization
"""

from __future__ import annotations

import difflib
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
    ComparisonResult,
    SearchResult,
    SimilarityEngine,
    UnknownWordError,
)

logger = logging.getLogger(__name__)

# Curated example queries offered as one-click buttons in the sidebar (FR-23).
EXAMPLE_WORDS: List[str] = ["bank", "apple", "mouse", "run", "paris", "dog"]


def _use_example_word(word: str) -> None:
    """Sidebar-button callback: load an example or suggested word into the
    query box so the next rerun searches for it."""
    st.session_state.query_box = word

# ---------- Page configuration ----------

st.set_page_config(
    page_title="Explainable Semantic Space Explorer",
    page_icon="🔍",
    layout="wide",
)


# ---------- Cached resource initialization ----------


@st.cache_resource(show_spinner="Loading embedding vectors...")
def load_all_engines() -> tuple[EmbeddingLoader, SimilarityEngine, SimilarityEngine]:
    """Initialize ``EmbeddingLoader`` and the two ``SimilarityEngine`` instances.

    Thanks to ``@st.cache_resource``, this runs only once after the app
    starts. The vocabulary × 300 / 384-dim matrices stay resident in
    memory, and reinitialization is avoided for fast response times.

    Returns:
        tuple: ``(loader, static_engine, contextual_engine)``.

    Raises:
        EmbeddingLoaderError: If asset files are missing or an alignment
            check fails.
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

    logger.info("All engines initialized")
    return loader, static_engine, contextual_engine


# ---------- Cached per-query computations (FR-26) ----------
#
# Heavy engine calls are wrapped in ``@st.cache_data`` so that an identical
# query is not recomputed across the four tabs (previously the distance
# distribution was recomputed several times per query). The engine argument is
# prefixed with ``_`` so Streamlit skips hashing it (it is a cached singleton);
# an ``engine_tag`` string keeps the static and contextual engines distinct in
# the cache key instead.


@st.cache_data(show_spinner=False)
def cached_search(
    _engine: SimilarityEngine,
    engine_tag: str,
    query: str,
    top_k: int,
    pos_filter: str | None,
) -> List[SearchResult]:
    """Cache ``SimilarityEngine.search`` keyed on the query and parameters.

    ``engine_tag`` participates in the cache key (it is not otherwise used in
    the body) so the static and contextual engines do not collide.
    """
    logger.debug("cached_search miss: engine=%s query=%s top_k=%d", engine_tag, query, top_k)
    return _engine.search(query, top_k=top_k, pos_filter=pos_filter)


@st.cache_data(show_spinner=False)
def cached_distribution(
    _engine: SimilarityEngine,
    engine_tag: str,
    query: str,
) -> dict:
    """Cache ``SimilarityEngine.get_distance_distribution`` per (engine, query).

    ``engine_tag`` participates in the cache key (it is not otherwise used in
    the body) so the static and contextual engines do not collide.
    """
    logger.debug("cached_distribution miss: engine=%s query=%s", engine_tag, query)
    return _engine.get_distance_distribution(query)


@st.cache_data(show_spinner=False)
def cached_compare(
    _static_engine: SimilarityEngine,
    _contextual_engine: SimilarityEngine,
    query: str,
    top_k: int,
) -> ComparisonResult:
    """Cache the static-vs-contextual comparison per (query, top_k)."""
    return _static_engine.compare(query, other=_contextual_engine, top_k=top_k)


# ---------- Helper: SearchResult → DataFrame ----------


def results_to_df(scored: List[dict]) -> pd.DataFrame:
    """Convert the output of ``attach_z_scores()`` into a DataFrame for ``st.dataframe``.

    Args:
        scored: List returned by ``Analyzer.attach_z_scores()``.

    Returns:
        pd.DataFrame: DataFrame ready for display.
    """
    rows = []
    for item in scored:
        rows.append(
            {
                "Rank": item["rank"],
                "Word": item["word"],
                "Similarity": round(item["similarity"], 4),
                "POS": item["pos_tag"],
                "Rank in POS": item["pos_rank"],
                "Z-score": round(item["z_score"], 3),
            }
        )
    return pd.DataFrame(rows)


# ---------- Main ----------


def main() -> None:
    """Application entry point. Builds the sidebar and the four tabs."""

    st.title("Explainable Semantic Space Explorer")
    st.caption(
        "A tool that explains the difference between static (Word2Vec) "
        "and contextual (SBERT) embeddings numerically."
    )

    # --- Load engines ---
    try:
        loader, static_engine, contextual_engine = load_all_engines()
    except EmbeddingLoaderError as e:
        st.error(f"Failed to load asset files: {e}")
        st.stop()

    # Fetch POS labels dynamically (POSFilter.group_by_pos needs SearchResult, so read them directly)
    unique_pos: list[str] = sorted(set(loader.pos.tolist()))

    # ---------- Sidebar ----------
    pos_options = ["ALL"] + unique_pos
    # Source of truth for the search inputs; set once, then updated by the
    # search form (on submit) or the example / suggestion buttons (on click).
    st.session_state.setdefault("query_box", "king")
    st.session_state.setdefault("top_k_box", 10)
    st.session_state.setdefault("pos_box", "ALL")

    with st.sidebar:
        st.header("Search settings")

        # Example queries (FR-23): one click loads a curated word.
        st.caption("Try an example:")
        example_cols = st.columns(3)
        for i, word in enumerate(EXAMPLE_WORDS):
            example_cols[i % 3].button(
                word,
                key=f"example_{word}",
                on_click=_use_example_word,
                args=(word,),
                use_container_width=True,
            )

        # Search form (FR-26): heavy computation runs only on submit, so
        # editing the inputs no longer recomputes every tab on each keystroke.
        with st.form("search_form"):
            st.text_input(
                "Query word",
                key="query_box",
                help="Enter an English word that exists in the vocabulary",
            )
            st.slider(
                "Top-K (number of results to show)",
                min_value=1,
                max_value=50,
                step=1,
                key="top_k_box",
            )
            st.selectbox("POS filter", pos_options, key="pos_box")
            st.form_submit_button("🔍 Search", use_container_width=True)

        st.divider()
        st.header("Projection / cluster settings")

        projection_method: str = st.radio(
            "Projection method",
            options=["pca", "umap"],
            format_func=lambda x: x.upper(),
            horizontal=True,
        )

        n_clusters: int = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
        )

    # Resolve the active (submitted / example-selected) search parameters.
    query_word: str = st.session_state.query_box.strip().lower()
    top_k: int = st.session_state.top_k_box
    pos_selected: str = st.session_state.pos_box
    pos_filter: str | None = None if pos_selected == "ALL" else pos_selected

    # --- Wait when the query is empty ---
    if not query_word:
        st.info("Enter a query word or pick an example in the sidebar.")
        return

    # --- Vocabulary check with close-word suggestions (FR-24) ---
    if query_word not in loader.vocab:
        st.warning(f"\"{query_word}\" is not in the vocabulary.")
        suggestions = difflib.get_close_matches(
            query_word, list(loader.vocab.keys()), n=5, cutoff=0.6
        )
        if suggestions:
            st.write("Did you mean:")
            suggestion_cols = st.columns(len(suggestions))
            for col, word in zip(suggestion_cols, suggestions):
                col.button(
                    word,
                    key=f"suggest_{word}",
                    on_click=_use_example_word,
                    args=(word,),
                    use_container_width=True,
                )
        else:
            st.caption("No close matches found — try a different spelling.")
        return

    # ---------- Four tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Similar-word search", "Static vs. contextual", "Distance distribution", "Projection / clusters"]
    )

    # =========================================================
    # Tab 1: Similar-word search
    # =========================================================
    with tab1:
        st.subheader(f"Top-{top_k} similar words for \"{query_word}\"")

        # core: similarity search + distribution retrieval + Z-score augmentation
        try:
            static_results: list[SearchResult] = cached_search(
                static_engine, "static", query_word, top_k, pos_filter
            )
        except UnknownWordError as e:
            st.error(str(e))
            return

        static_dist = cached_distribution(static_engine, "static", query_word)
        scored = Analyzer.attach_z_scores(static_results, static_dist)

        # Result table
        df = results_to_df(scored)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Plain-language verdict for the Top-1 neighbor (FR-25):
        # answers "is this similarity actually high?" in words.
        if scored:
            top = scored[0]
            verdict = Analyzer.interpret_similarity(top["similarity"], static_dist)
            st.success(f"**{top['word']}** — {verdict.text}")

        # Cosine formula breakdown (top result)
        if scored:
            with st.expander("Show formula breakdown (Top-1)"):
                exp = scored[0]["explanation"]
                st.code(exp["formula"], language=None)
                col1, col2, col3 = st.columns(3)
                col1.metric("dot product dot(a,b)", f"{exp['dot_product']:.4f}")
                col2.metric("‖query‖", f"{exp['norm_a']:.4f}")
                col3.metric("‖target‖", f"{exp['norm_b']:.4f}")

        st.divider()

        # core: POS distribution
        pos_dist: dict[str, int] = POSFilter.pos_distribution(static_results)
        if pos_dist:
            st.subheader("POS distribution")
            pos_df = pd.DataFrame(
                [{"POS": k, "Count": v} for k, v in pos_dist.items()]
            )
            bar = (
                alt.Chart(pos_df)
                .mark_bar()
                .encode(
                    x=alt.X("POS:N", sort="-y"),
                    y=alt.Y("Count:Q"),
                    tooltip=["POS", "Count"],
                )
                .properties(height=200)
            )
            st.altair_chart(bar, use_container_width=True)

        # core: heterogeneity rate
        query_pos: str = loader.pos[loader.vocab[query_word]]
        hetero: float = POSFilter.heterogeneity_rate(static_results, query_pos)
        st.metric(
            "Heterogeneity rate (fraction with a POS different from the query)",
            f"{hetero:.1%}",
            help="Higher values mean more cross-POS similar words",
        )

    # =========================================================
    # Tab 2: Static vs. contextual comparison
    # =========================================================
    with tab2:
        st.subheader(f"\"{query_word}\" — Static (Word2Vec) vs. Contextual (SBERT)")

        # core: fetch the comparison result
        try:
            comparison = cached_compare(
                static_engine, contextual_engine, query_word, top_k
            )
        except UnknownWordError as e:
            st.error(str(e))
            return

        # core: neighborhood stability score
        stability: float = Analyzer.neighborhood_stability(
            comparison.static_results, comparison.contextual_results
        )

        # Aggregate metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Common words", len(comparison.common_words))
        m2.metric("Static-only words", len(comparison.static_only))
        m3.metric("Contextual-only words", len(comparison.contextual_only))
        m4.metric(
            "Neighborhood stability (Jaccard)",
            f"{stability:.3f}",
            help="1.0 = perfect agreement, 0.0 = no overlap",
        )

        st.divider()

        # Two columns: static Top-K / contextual Top-K
        col_s, col_b = st.columns(2)

        with col_s:
            st.write("**Static (Word2Vec) Top-K**")
            static_scored = Analyzer.attach_z_scores(
                comparison.static_results,
                cached_distribution(static_engine, "static", query_word),
            )
            st.dataframe(results_to_df(static_scored), hide_index=True)

        with col_b:
            st.write("**Contextual (SBERT) Top-K**")
            contextual_scored = Analyzer.attach_z_scores(
                comparison.contextual_results,
                cached_distribution(contextual_engine, "contextual", query_word),
            )
            st.dataframe(results_to_df(contextual_scored), hide_index=True)

        st.divider()

        # Rank-difference table (words shared by both models)
        if comparison.rank_diff:
            st.subheader("Rank difference (common words)")
            rd_rows = [
                {"Word": w, "Rank diff (Static − Contextual)": d}
                for w, d in sorted(
                    comparison.rank_diff.items(), key=lambda x: abs(x[1]), reverse=True
                )
            ]
            st.dataframe(pd.DataFrame(rd_rows), hide_index=True, use_container_width=True)

        # Unique-word badges
        if comparison.static_only:
            st.write("**Static-only words:**", "  ".join(f"`{w}`" for w in comparison.static_only))
        if comparison.contextual_only:
            st.write("**Contextual-only words:**", "  ".join(f"`{w}`" for w in comparison.contextual_only))

    # =========================================================
    # Tab 3: Distance distribution analysis
    # =========================================================
    with tab3:
        st.subheader(f"Distance distribution analysis for \"{query_word}\"")

        # core: distribution retrieval + stat enrichment + comparison
        static_dist = cached_distribution(static_engine, "static", query_word)
        contextual_dist = cached_distribution(contextual_engine, "contextual", query_word)

        static_stats = Analyzer.enrich_distribution(static_dist)
        contextual_stats = Analyzer.enrich_distribution(contextual_dist)
        dist_cmp = Analyzer.compare_distributions(query_word, static_dist, contextual_dist)

        # Per-model metrics
        col_s, col_b = st.columns(2)
        with col_s:
            st.write("**Static (Word2Vec)**")
            st.metric("Mean similarity", f"{static_stats.mean:.4f}")
            st.metric("Standard deviation", f"{static_stats.std:.4f}")
            st.metric("Top-1 similarity", f"{static_stats.top1_similarity:.4f}")
            st.metric("Z-score", f"{static_stats.z_score:.3f}")
            st.metric("Median", f"{static_stats.median:.4f}")
            st.caption(
                "Top-1 verdict — "
                + Analyzer.interpret_similarity(
                    static_stats.top1_similarity, static_dist
                ).text
            )

        with col_b:
            st.write("**Contextual (SBERT)**")
            st.metric("Mean similarity", f"{contextual_stats.mean:.4f}")
            st.metric("Standard deviation", f"{contextual_stats.std:.4f}")
            st.metric("Top-1 similarity", f"{contextual_stats.top1_similarity:.4f}")
            st.metric("Z-score", f"{contextual_stats.z_score:.3f}")
            st.metric("Median", f"{contextual_stats.median:.4f}")
            st.caption(
                "Top-1 verdict — "
                + Analyzer.interpret_similarity(
                    contextual_stats.top1_similarity, contextual_dist
                ).text
            )

        st.divider()

        # Histogram (static)
        st.subheader("Similarity distribution histogram (static)")

        hist_data = Analyzer.histogram(static_dist["histogram_data"], n_bins=50)
        bin_centers = [
            (hist_data.bin_edges[i] + hist_data.bin_edges[i + 1]) / 2
            for i in range(len(hist_data.counts))
        ]
        hist_df = pd.DataFrame(
            {
                "Similarity": bin_centers,
                "Frequency": hist_data.counts,
            }
        )

        # Vertical rule for the Top-1
        rule_df = pd.DataFrame({"x": [static_dist["top1_similarity"]]})
        base = alt.Chart(hist_df).mark_bar(opacity=0.7).encode(
            x=alt.X("Similarity:Q", bin=False, title="Cosine similarity"),
            y=alt.Y("Frequency:Q"),
            tooltip=["Similarity", "Frequency"],
        )
        rule = (
            alt.Chart(rule_df)
            .mark_rule(color="red", strokeDash=[4, 4], strokeWidth=2)
            .encode(x="x:Q")
        )
        st.altair_chart((base + rule).properties(height=250), use_container_width=True)
        st.caption(f"Red dashed line = Top-1 similarity ({static_dist['top1_similarity']:.4f})")

        # Distribution comparison metrics
        st.divider()
        st.subheader("Static vs. contextual distribution comparison")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean diff (Static − Contextual)", f"{dist_cmp.mean_diff:.4f}")
        c2.metric("Std diff", f"{dist_cmp.std_diff:.4f}")
        c3.metric("Z-score diff", f"{dist_cmp.z_score_diff:.3f}")

    # =========================================================
    # Tab 4: Projection / clustering
    # =========================================================
    with tab4:
        st.subheader(f"2D projection around \"{query_word}\" ({projection_method.upper()})")

        # Fetch Top-K indices around the query (core: search)
        try:
            proj_results: list[SearchResult] = cached_search(
                static_engine, "static", query_word, top_k, None
            )
        except UnknownWordError as e:
            st.error(str(e))
            return

        query_idx: int = loader.vocab[query_word]
        neighbor_indices: list[int] = [r.index for r in proj_results]
        # Only append the query itself if it isn't already a neighbor
        if query_idx not in neighbor_indices:
            target_indices = neighbor_indices + [query_idx]
        else:
            target_indices = neighbor_indices

        index_to_word: dict[int, str] = {v: k for k, v in loader.vocab.items()}
        target_words: list[str] = [index_to_word[i] for i in target_indices]

        target_vectors: np.ndarray = loader.static_vectors[target_indices]

        # analysis: clustering (restricted to the query's neighborhood)
        n_clust_actual = min(n_clusters, len(target_indices))
        clusterer = KMeansClusterer(n_clusters=n_clust_actual, seed=42)
        cluster_result = clusterer.fit(target_vectors)

        # analysis: projection (PCA or UMAP)
        projector = Projector(method=projection_method, seed=42)
        proj_result = projector.fit_transform(target_vectors)
        proj_result = projector.attach_clusters(proj_result, cluster_result.labels)

        # Similarity map (similarity to the query)
        sim_map: dict[str, float] = {r.word: r.similarity for r in proj_results}
        sim_map[query_word] = 1.0  # query itself

        # Scatter-plot data
        coords = proj_result.coords_2d
        scatter_df = pd.DataFrame(
            {
                "x": coords[:, 0].tolist(),
                "y": coords[:, 1].tolist(),
                "Word": target_words,
                "Cluster": [str(c) for c in proj_result.cluster_labels.tolist()],
                "Similarity": [round(sim_map.get(w, 0.0), 4) for w in target_words],
                "Role": ["✕ Query" if w == query_word else "Neighbor" for w in target_words],
            }
        )

        # Altair scatter plot
        axis_label = "PC" if projection_method == "pca" else "UMAP"
        scatter = (
            alt.Chart(scatter_df)
            .mark_point(size=120, filled=True)
            .encode(
                x=alt.X("x:Q", title=f"{axis_label}1"),
                y=alt.Y("y:Q", title=f"{axis_label}2"),
                color=alt.Color("Cluster:N", title="Cluster"),
                shape=alt.Shape(
                    "Role:N",
                    scale=alt.Scale(
                        domain=["✕ Query", "Neighbor"],
                        range=["cross", "circle"],
                    ),
                ),
                tooltip=["Word", "Similarity", "Cluster", "Role"],
            )
            .properties(height=400)
            .interactive()
        )

        # Word labels
        text = (
            alt.Chart(scatter_df)
            .mark_text(dy=-10, fontSize=11)
            .encode(
                x="x:Q",
                y="y:Q",
                text="Word:N",
                color=alt.Color("Cluster:N"),
            )
        )

        st.altair_chart((scatter + text).properties(height=420), use_container_width=True)

        # PCA contribution rates
        if projection_method == "pca" and proj_result.explained_variance:
            c1, c2 = st.columns(2)
            c1.metric(
                "PC1 contribution",
                f"{proj_result.explained_variance[0]:.1%}",
            )
            c2.metric(
                "PC2 contribution",
                f"{proj_result.explained_variance[1]:.1%}",
            )

        st.divider()

        # Per-cluster word listing
        st.subheader("Word groups per cluster")
        cluster_groups: dict[int, list[str]] = {}
        for word, label in zip(target_words, proj_result.cluster_labels.tolist()):
            cluster_groups.setdefault(int(label), []).append(word)

        cluster_df_rows = []
        for cid in sorted(cluster_groups.keys()):
            cluster_df_rows.append(
                {
                    "Cluster ID": cid,
                    "Word count": len(cluster_groups[cid]),
                    "Words": " / ".join(cluster_groups[cid]),
                }
            )
        st.dataframe(
            pd.DataFrame(cluster_df_rows),
            hide_index=True,
            use_container_width=True,
        )


# ---------- Entry point ----------

if __name__ == "__main__":
    main()
