# External Design Document — Explainable Semantic Space Explorer

**Project Name:** Explainable Semantic Space Explorer  
**Version:** 1.0  
**Date:** 2026-05-05

---

## 1. Screen Composition

| No. | Screen Name | Description |
|-----|-------------|-------------|
| 1 | Search Tab | Query word input / Top-K similar word display / cosine similarity breakdown and Z-score inspection |
| 2 | Comparison Tab | Differential comparison between Word2Vec and SBERT for the same query |
| 3 | Projection Tab | Overview of the 2D embedding space via PCA / UMAP |
| 4 | Cluster Tab | Color-coded scatter plot of KMeans clustering results |

> This system is implemented as a single-page application (SPA). All operations are completed via tab switching without page transitions.

---

## 2. Screen Design

### 2.1 Search Tab (Top-K Similar Word Search)

**Layout**

- Page title: `Explainable Semantic Space Explorer`
- Subtitle: a one-line summary of the system

**Input Area**

- Label: "Enter a query word"
- Component: `st.text_input` (placeholder: e.g., bank, king, computer)
- Top-K slider: `st.slider` (range: 5–30, default: 10)
- POS filter: `st.selectbox` (All POS / Noun / Verb / Adjective, etc.)
- Button: "Run search" (`st.button`)

**Result Display Area**

- Shown after search execution (with loading spinner)
- Word2Vec and SBERT results displayed side by side using `st.columns(2)`
- Per-row items: rank, similar word, similarity score, dot product, norm (cosine similarity breakdown), within-POS rank
- Distance distribution: histogram via `st.altair_chart`; vocabulary-wide mean, standard deviation, and Z-score presented with `st.metric`

### 2.2 Comparison Tab (Word2Vec vs SBERT Differences)

**Input Area**

- Label: "Enter a query word to compare"
- Component: `st.text_input`
- Top-K slider: `st.slider`

**Result Display Area**

- Counts of shared words, Word2Vec-only words, and SBERT-only words displayed with `st.metric`
- Rank differences and distance differences for shared words shown via `st.dataframe` (sorted in descending order by magnitude of difference)

### 2.3 Projection Tab (2D Space Projection)

**Input Area**

- Query word: `st.text_input`
- Projection method: `st.selectbox` (PCA / UMAP)
- Top-K: `st.slider`
- seed: `st.number_input`

**Result Display Area**

- 2D scatter plot displayed via `st.altair_chart`
- When PCA is selected: explained variance ratio of the 1st and 2nd principal components shown with `st.metric`
- The query word is shown with a highlighted marker

### 2.4 Cluster Tab (KMeans Clustering)

**Input Area**

- Query word: `st.text_input`
- Number of clusters: `st.slider` (default: 8)
- seed: `st.number_input`

**Result Display Area**

- Color-coded cluster scatter plot displayed via `st.altair_chart`
- Cluster IDs identified by color and legend
- Representative word list for each cluster shown via `st.expander`

---

## 3. Input / Output Definitions

### 3.1 Input (Query Word)

| Item | Details |
|------|---------|
| Field name | Query word |
| Input method | Manual entry into the text box |
| Data type | `str` |
| Required / Optional | Required |
| Case handling | Normalized to lowercase before searching |
| Validation | Empty strings and out-of-vocabulary words display an error message and abort processing |

**Input Example**

```
bank
```

### 3.2 Output (Similar Word List)

Output format: tables and metric displays in the Streamlit UI

| Field name | Type | Description |
|---|---|---|
| rank | int | Similarity rank (1-indexed) |
| word | str | Similar word |
| similarity | float | Cosine similarity (-1.0 to 1.0) |
| dot_product | float | Dot product a · b |
| norm_a | float | Norm of the query word ‖a‖ |
| norm_b | float | Norm of the similar word ‖b‖ |
| pos_tag | str | POS label (e.g., NN, VB, JJ) |
| pos_rank | int | Rank within the same POS |
| z_score | float | Z-score within the vocabulary-wide distribution |

**Output Example (UI display image)**

```
Query word: bank

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Word2Vec Results (Top-5)
1. river    | cos: 0.712 | dot: 14.2 | ‖a‖: 4.47, ‖b‖: 4.46 | NN / within-POS rank 1
2. shore    | cos: 0.698 | dot: 13.9 | ...
3. creek    | cos: 0.681 | ...
Z-score: 3.21 (vocabulary mean: 0.12, std: 0.18)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SBERT Results (Top-5)
1. financial | cos: 0.724 | dot: 18.1 | ...
2. credit    | cos: 0.711 | ...
3. loan      | cos: 0.698 | ...
Z-score: 3.47 (vocabulary mean: 0.09, std: 0.18)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> For the same word "bank", Word2Vec surfaces geographic senses (river, shore) at the top, while SBERT surfaces financial senses (financial, credit). This contrast is the central phenomenon this system aims to explain.

---

## 4. Error Message List

| Code | Trigger Condition | Displayed Message |
|------|-------------------|-------------------|
| E001 | Input is an empty string | "Please enter a query word." |
| E002 | Input word is not in the vocabulary | "'{word}' is not included in the vocabulary. Please try a different word." |
| E003 | `.npy` / `.json` files under `data/` are not found | "Asset files not found. Please check the setup instructions." |
| E004 | shape / dtype mismatch between `manifest.json` and actual files | "Data file consistency check failed. Please regenerate manifest.json." |
| E005 | Invalid Top-K value (0 or below, etc.) | "Top-K must be an integer of 1 or greater." |
| E006 | Invalid number of clusters (1 or below, etc.) | "Number of clusters must be an integer of 2 or greater." |

> Errors are displayed in red via the `st.error()` component.

---

## 5. System Architecture

### 5.1 Layered Structure and Dependency Direction

The system is composed of four layers: **data → core → transforms → ui**. Dependencies flow in one direction only; backward dependencies are forbidden.

```
data/ ──► core/ ──► transforms/ ──► ui/app.py
```

| Layer | Directory | Responsibility | Forbidden |
|---|---|---|---|
| Data layer | `data/` | Holds pre-generated vectors, vocabulary, POS tags, and consistency specifications | Dynamic generation / editing |
| Logic layer | `core/` | Embedding loading, distance computation, similarity search, POS filtering, distance distribution analysis | Importing Streamlit, calling external APIs |
| Preprocessing layer | `transforms/` | Dimensionality reduction (PCA / UMAP), clustering (KMeans) | Depending on the UI |
| UI layer | `ui/` | Screen construction, input handling, and result display via Streamlit | Inlining analysis logic |

### 5.2 Main Class List

**`core/` — Core Logic Layer**

| Class | Responsibility |
|---|---|
| `EmbeddingLoader` | Loads vectors, vocabulary, POS tags, and `manifest.json` under `data/`, and verifies consistency |
| `DistanceMetrics` | Provides a custom implementation of cosine similarity (the `cosine_similarity` of scipy / sklearn is not used) |
| `SimilarityEngine` | 1 instance = 1 model (Word2Vec or SBERT). Handles Top-K search and cross-model comparison |
| `POSFilter` | Performs POS-label-based filtering, within-POS ranking, and heterogeneity rate computation |
| `Analyzer` | Provides distance distribution statistics (mean, standard deviation, Z-score, raw histogram data) |

**`transforms/` — Preprocessing Layer**

| Class | Responsibility |
|---|---|
| `Projector` | Provides 2D projection via PCA / UMAP. Guarantees reproducibility through a fixed seed |
| `KMeansClusterer` | Performs cosine-based clustering via KMeans after L2 normalization |

### 5.3 Asset File Specifications

| File | shape | dtype | Approximate size |
|---|---|---|---|
| `data/embeddings/static_vectors.npy` | (83823, 300) | float32 | approx. 96 MB |
| `data/embeddings/contextual_vectors.npy` | (83823, 384) | float32 | approx. 123 MB |
| `data/metadata/vocab.json` | 83823 entries | UTF-8 JSON | approx. 1.2 MB |
| `data/metadata/vocab_pos.npy` | (83823,) | string array | approx. 2.9 MB |
| `data/manifest.json` | — | UTF-8 JSON | a few KB |

The premise of consistency verification is that all files agree on the vocabulary size N = 83823. At startup, shape / dtype are cross-checked against `manifest.json`, and any mismatch is immediately detected as `ManifestViolationError`.
