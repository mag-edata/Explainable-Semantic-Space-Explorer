# Internal Design Document — Explainable Semantic Space Explorer

**Project name:** Explainable Semantic Space Explorer  
**Version:** 1.0  
**Date:** 2026-05-05

---

## 1. System Architecture Diagram

```
+----------------------------------------------------------------------+
|                                                                      |
|                          User (Browser)                              |
|                              |                                       |
|                              v                                       |
|     +------------------------------------------------------+         |
|     |          ui/app.py   Streamlit 4 tabs                |         |
|     +------------------------------------------------------+         |
|                              |                                       |
|         +-----------+--------+--------+----------+                   |
|         |           |        |        |          |                   |
|         v           v        v        v          v                   |
|   +-----------+ +-------+ +---------+ +----------+ +-----------+     |
|   |similarity_| |pos_   | |analyzer | |clustering| |projection |     |
|   |engine.py  | |filter | |  .py    | |  .py     | |  .py      |     |
|   +-----+-----+ |  .py  | +---------+ +----------+ +-----------+     |
|         |       +-------+                                            |
|         v                                                            |
|   +---------------------+                                            |
|   | distance_metrics.py |                                            |
|   +----------+----------+                                            |
|              |                                                       |
|              v                                                       |
|   +---------------------+                                            |
|   | embedding_loader.py |                                            |
|   +----------+----------+                                            |
|              |                                                       |
|              v                                                       |
|   +------------------------------------------------------+           |
|   |  data/                                               |           |
|   |    static_vectors.npy / contextual_vectors.npy       |           |
|   |    vocab.json / vocab_pos.npy / manifest.json        |           |
|   +------------------------------------------------------+           |
|                                                                      |
|          * No network communication at inference time                |
|                                                                      |
+----------------------------------------------------------------------+
```

**Data flow summary**

| Flow | Path |
|------|------|
| Startup / load | `data/*.npy/.json` → `EmbeddingLoader` → `SimilarityEngine` × 2 |
| Similar-word search | Query word → `SimilarityEngine.search` → `Analyzer.attach_z_scores` → UI display |
| Comparison | Query word → `SimilarityEngine.compare` → `Analyzer.neighborhood_stability` → UI display |
| Projection / clustering | Neighbor vectors → `KMeansClusterer.fit` → `Projector.fit_transform` → Altair scatter plot |

---

## 2. Module Composition

### 2.1 `core/embedding_loader.py`

**Role:** Loads embedding vectors, vocabulary, and POS data under `data/`, and validates consistency using `manifest.json`.

**Main methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(data_root: Path) -> None` | Receives the Path to `data/`. Raises `FileNotFoundError` if absent |
| `load_all` | `(self) -> None` | The sole public API. Executes in the order manifest → embeddings → metadata → validate |
| `_load_manifest` | `(self) -> None` | Holds `manifest.json` as a dict |
| `_load_embeddings` | `(self) -> None` | `np.load`s `static_vectors.npy` / `contextual_vectors.npy` |
| `_load_metadata` | `(self) -> None` | Converts `vocab.json` (list format) into `Dict[str,int]` and loads `vocab_pos.npy` |
| `_validate` | `(self) -> None` | Validates N alignment, manifest matching, and contiguity of vocab indices |
| `_validate_against_manifest` | `(self, array: np.ndarray, key: str) -> None` | Checks shape / dtype against `manifest[key]` |

**Instance variables (after `load_all()`)**

| Variable | Type | Content |
|----------|------|---------|
| `static_vectors` | `np.ndarray` | shape (83823, 300), float32 |
| `contextual_vectors` | `np.ndarray` | shape (83823, 384), float32 |
| `vocab` | `Dict[str, int]` | word → index |
| `pos` | `np.ndarray` | shape (83823,), POS labels |
| `manifest` | `dict` | Contents of manifest.json |

**Raised exceptions:** `FileNotFoundError` / `IndexAlignmentError` / `ManifestViolationError`  
**External dependencies:** `numpy`, `json`, `pathlib`

---

### 2.2 `core/similarity_engine.py`

**Role:** Top-K similar-word search, cross-model comparison, and distance distribution computation against a single embedding matrix. One instance = one model (Word2Vec or SBERT).

**Main methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(vectors: np.ndarray, vocab: Dict[str,int], pos_tags: np.ndarray, metrics: DistanceMetrics) -> None` | Type-validates each argument. Builds the `_index_to_word` list |
| `search` | `(self, query_word: str, top_k: int = 10, pos_filter: str \| None = None) -> List[SearchResult]` | Returns Top-K similar words in descending order. POS filter is applied after Top-K extraction |
| `compare` | `(self, query_word: str, other: "SimilarityEngine", top_k: int = 10) -> ComparisonResult` | Compares Top-K of self and other, returning common, unique, and diff words |
| `get_distance_distribution` | `(self, query_word: str) -> dict` | Similarity distribution statistics over the full vocabulary (mean / std / top1 / z_score / histogram_data) |
| `word_to_index` | `(self, word: str) -> int` | Raises `UnknownWordError` if out of vocabulary |

**Dataclasses**

`SearchResult`: `word, index, similarity, rank, pos_tag, pos_rank, explanation`  
`ComparisonResult`: `query_word, static_results, contextual_results, common_words, static_only, contextual_only, rank_diff, similarity_diff`

**Raised exceptions:** `UnknownWordError` / `InvalidTopKError` / `TypeError`  
**External dependencies:** `numpy`, `DistanceMetrics`

---

### 2.3 `core/distance_metrics.py`

**Role:** Fully custom implementation of cosine similarity. Does not use `scipy` / `sklearn.cosine_similarity`. All methods are `staticmethod`.

**Main methods**

| Method | Signature | Formula |
|--------|-----------|---------|
| `l2_norm` | `(vector: np.ndarray) -> float` | `‖v‖ = √(v · v)` |
| `cosine_similarity` | `(vec_a: np.ndarray, vec_b: np.ndarray) -> float` | `(a · b) / (‖a‖ × ‖b‖)` |
| `cosine_similarity_batch` | `(query: np.ndarray, matrix: np.ndarray) -> np.ndarray` | Computes all rows at once. Guards against zero division with `_EPSILON = 1e-10` |
| `explain` | `(vec_a: np.ndarray, vec_b: np.ndarray) -> Dict[str, Any]` | Returns `dot_product / norm_a / norm_b / denominator / similarity / formula` as a dict |

**Batched computation strategy**

```python
dot_products = matrix @ query
row_norms    = sqrt(sum(matrix * matrix, axis=1))   # np.linalg.norm not used
denominators = row_norms * query_norm
similarities = dot_products / where(zero_mask, 1.0, denominators)
similarities[zero_mask] = 0.0
```

**Raised exceptions:** `TypeError` / `VectorDimensionError`  
**External dependencies:** `numpy` only

---

### 2.4 `core/pos_filter.py`

**Role:** Applies POS filters to `SearchResult` lists, groups them, and computes the heterogeneity rate. All methods are `staticmethod`.

**Main methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `filter` | `(results: List[SearchResult], pos_tag: str) -> List[SearchResult]` | Retains only entries with the specified POS. Empty string raises `ValueError`; no matches raises `UnknownPOSTagError` |
| `group_by_pos` | `(results: List[SearchResult]) -> Dict[str, List[SearchResult]]` | Groups by POS (preserves original order) |
| `pos_distribution` | `(results: List[SearchResult]) -> Dict[str, int]` | Counts entries per POS (descending by count) |
| `heterogeneity_rate` | `(results: List[SearchResult], query_pos: str) -> float` | Heterogeneity rate = ratio of words whose POS differs from the query word (range: [0.0, 1.0]) |
| `assign_pos_ranks` | `(results: List[SearchResult]) -> List[SearchResult]` | Assigns sequential ranks within the same POS, starting from 1 |
| `top_pos` | `(results: List[SearchResult], n: int = 3) -> List[str]` | Returns the top n POS by occurrence count |

**Heterogeneity rate formula**

```
heterogeneity_rate = |{r ∈ results | r.pos_tag ≠ query_pos}| / |results|
```

`0.0` = all results share the same POS (fully POS-dependent); `1.0` = all results have a different POS (POS-independent).

**External dependencies:** `SearchResult` (defined in `similarity_engine.py`)

---

### 2.5 `core/analyzer.py`

**Role:** Statistical augmentation of distance distributions, histogram generation, cross-model distribution comparison, and neighborhood stability computation. All methods are `staticmethod`.

**Main methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `enrich_distribution` | `(distribution: dict) -> DistributionStats` | Returns a statistical summary augmented with median and quartiles |
| `histogram` | `(data: List[float], n_bins: int = 50) -> HistogramData` | Bins the data with `np.histogram`. Returns `bin_edges / counts / data_min / data_max` |
| `attach_z_scores` | `(results: List[SearchResult], distribution: dict) -> List[Dict]` | Returns a list of dicts with a Z-score attached to each result |
| `compare_distributions` | `(query_word: str, static_dist: dict, contextual_dist: dict) -> DistributionComparison` | Returns the distribution diff between the two models (mean_diff / std_diff / z_score_diff) |
| `neighborhood_stability` | `(static_results, contextual_results) -> float` | Jaccard coefficient = ratio of common words across both models' Top-K |

**Dataclasses:** `DistributionStats` / `HistogramData` / `DistributionComparison`  
**External dependencies:** `numpy`, `SearchResult`, `ComparisonResult`

---

### 2.6 `transforms/clustering.py`

**Role:** Cosine-distance-based clustering via L2 normalization + `sklearn.KMeans`.

**Main methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(n_clusters: int = 8, seed: int = 42, max_iter: int = 300) -> None` | Validates arguments. Raises `InvalidClusterCountError` if `n_clusters < 1` |
| `fit` | `(self, vectors: np.ndarray) -> ClusterResult` | L2-normalize → `KMeans.fit()` → returns a `ClusterResult` |
| `get_labels` | `(self) -> np.ndarray` | Raises `NotFittedError` if not yet fit |
| `get_result` | `(self) -> ClusterResult` | Raises `NotFittedError` if not yet fit |

**Dataclass `ClusterResult`:** `labels, n_clusters, inertia, seed, n_samples`

**External dependencies:** `numpy`, `sklearn.cluster.KMeans`

---

### 2.7 `transforms/projection.py`

**Role:** 2D projection via PCA / UMAP. One instance = one method.

**Main methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(method: str = "pca", seed: int = 42) -> None` | Raises `InvalidMethodError` if `method` is not `"pca"` / `"umap"` |
| `fit_transform` | `(self, vectors: np.ndarray) -> ProjectionResult` | Calls `_fit_pca` or `_fit_umap` |
| `attach_clusters` | `(self, result: ProjectionResult, cluster_labels: np.ndarray) -> ProjectionResult` | Returns a new `ProjectionResult` with cluster IDs attached (immutable operation) |

**Dataclass `ProjectionResult`:** `coords_2d, explained_variance, method, cluster_labels, n_samples, seed`

For UMAP, `explained_variance = []` (no corresponding contribution-rate concept).

**External dependencies:** `numpy`, `sklearn.decomposition.PCA`, `umap.UMAP`

---

### 2.8 `ui/app.py`

**Role:** Builds the 4-tab UI with Streamlit. Holds no business logic.

**Processing flow**

1. Page setup and title display
2. Initialize `EmbeddingLoader` and two `SimilarityEngine`s (Word2Vec / SBERT) only once per process via `@st.cache_resource`
3. Sidebar: accepts query word, Top-K, POS filter, projection method, and cluster count
4. Tab 1 "Search": `engine.search` → `Analyzer.attach_z_scores` → tabular display
5. Tab 2 "Compare": `static_engine.compare` → diff table + Jaccard coefficient
6. Tab 3 "Distance distribution": `get_distance_distribution` × 2 → histograms + `compare_distributions`
7. Tab 4 "Projection / clusters": neighbor vectors → `KMeansClusterer.fit` → `Projector.fit_transform` / `attach_clusters` → Altair scatter plot

**External dependencies:** `streamlit`, `altair`, `pandas`, `core/`, `transforms/`

---

## 3. Data Flow

### Startup / load flow

1. `ui/app.py`: caches `load_all_engines()` with `@st.cache_resource` (executed only once per process)
2. `EmbeddingLoader(Path("data")).load_all()`:
   - Reads `manifest.json` and builds the alignment-check configuration
   - `np.load`s `static_vectors.npy` / `contextual_vectors.npy`
   - Converts `vocab.json` (list format) to `Dict[str, int]`
   - Loads `vocab_pos.npy`
   - Validates shape / dtype / N alignment (interrupts processing and raises on mismatch)
3. `static_engine = SimilarityEngine(loader.static_vectors, loader.vocab, loader.pos, metrics)`
4. `contextual_engine  = SimilarityEngine(loader.contextual_vectors, loader.vocab, loader.pos, metrics)`
5. Load complete → Streamlit UI rendering begins

### Similar-word search flow

1. The user enters the query word, Top-K, and POS filter, then presses the button
2. `SimilarityEngine.word_to_index(query_word)` → if out of vocabulary, display E002 error and abort
3. `DistanceMetrics.cosine_similarity_batch(query_vec, vectors)` → compute similarities for all N entries
4. Extract Top-K with `np.argpartition` (O(N + k log k))
5. Assign within-POS ranks via `POSFilter.assign_pos_ranks`
6. Store the dot-product / norm breakdown into `SearchResult.explanation` via `DistanceMetrics.explain`
7. Attach Z-scores against the full-vocabulary distribution via `Analyzer.attach_z_scores`
8. UI: tabular display of similar-word list, cosine breakdown, and Z-score

### Comparison flow (Word2Vec vs SBERT)

1. `static_engine.search(query_word, top_k)` → `static_results`
2. `contextual_engine.search(query_word, top_k)` → `contextual_results`
3. `static_engine.compare(query_word, other=contextual_engine, top_k)`:
   - Generates `common_words` / `static_only` / `contextual_only` via `set` operations
   - Computes `rank_diff` / `similarity_diff` per common word
4. `Analyzer.neighborhood_stability(static_results, contextual_results)` → Jaccard coefficient
5. UI: displays counts of common and unique words, and the diff ranking table

### Projection / cluster flow

1. `static_engine.search(query_word, top_k)` → obtain neighbor word indices
2. `target_indices = neighbor_indices + [query_idx]` (always include the query itself)
3. `target_vectors = loader.static_vectors[target_indices]`
4. `KMeansClusterer(n_clusters=k, seed=42).fit(target_vectors)`:
   - L2-normalize → `sklearn.KMeans.fit` → obtain `ClusterResult.labels`
5. `Projector(method=method, seed=42).fit_transform(target_vectors)`:
   - Generate 2D coordinates via PCA or UMAP
6. `projector.attach_clusters(projection_result, cluster_labels)` → result with cluster IDs attached
7. UI: render scatter plot with Altair `mark_point` + `mark_text`

---

## 4. Data Design

### Asset file specifications

| File | shape | dtype | Approx. size | Content |
|------|-------|-------|--------------|---------|
| `data/embeddings/static_vectors.npy` | (83823, 300) | float32 | ~96 MB | Word2Vec static embeddings |
| `data/embeddings/contextual_vectors.npy` | (83823, 384) | float32 | ~123 MB | SBERT contextual embeddings |
| `data/metadata/vocab.json` | 83823 entries | UTF-8 JSON | ~1.2 MB | `{"vocab": ["word0", ...]}` format |
| `data/metadata/vocab_pos.npy` | (83823,) | `<U9` etc. | ~2.9 MB | POS label array |
| `data/manifest.json` | — | UTF-8 JSON | a few KB | Expected shape / dtype values |

**`manifest.json` schema**

```json
{
  "static_vectors": { "shape": [83823, 300], "dtype": "float32" },
  "contextual_vectors":  { "shape": [83823, 384], "dtype": "float32" }
}
```

A shape / dtype mismatch raises `ManifestViolationError`. When a key is missing from the manifest, `logger.warning` is emitted and the check is skipped.

### Main dataclasses

**`SearchResult`**

| Field | Type | Meaning |
|-------|------|---------|
| `word` | `str` | Similar word |
| `index` | `int` | Vocabulary index (0-based) |
| `similarity` | `float` | Cosine similarity [-1.0, 1.0] |
| `rank` | `int` | Overall rank (1-based) |
| `pos_tag` | `str` | POS label |
| `pos_rank` | `int` | Rank within the same POS (1-based) |
| `explanation` | `dict` | `dot_product / norm_a / norm_b / denominator / similarity / formula` |

**`ComparisonResult`**

| Field | Type | Meaning |
|-------|------|---------|
| `query_word` | `str` | Query word |
| `static_results` | `List[SearchResult]` | Word2Vec Top-K |
| `contextual_results` | `List[SearchResult]` | SBERT Top-K |
| `common_words` | `List[str]` | Common words (sorted) |
| `static_only` | `List[str]` | Words unique to Word2Vec |
| `contextual_only` | `List[str]` | Words unique to SBERT |
| `rank_diff` | `Dict[str, int]` | Rank difference per common word (static_rank - contextual_rank) |
| `similarity_diff` | `Dict[str, float]` | Similarity difference per common word (static_sim - contextual_sim) |

**`ClusterResult`**

| Field | Type | Meaning |
|-------|------|---------|
| `labels` | `np.ndarray` | Cluster IDs, shape (N,) |
| `n_clusters` | `int` | Specified cluster count |
| `inertia` | `float` | Within-cluster sum of squares (in the normalized space) |
| `seed` | `int` | Random seed used |
| `n_samples` | `int` | Number of target words |

**`ProjectionResult`**

| Field | Type | Meaning |
|-------|------|---------|
| `coords_2d` | `np.ndarray` | 2D coordinates, shape (N, 2) |
| `explained_variance` | `List[float]` | Explained variance ratio (PCA only; `[]` for UMAP) |
| `method` | `str` | `"pca"` or `"umap"` |
| `cluster_labels` | `np.ndarray \| None` | Cluster IDs (attached via `attach_clusters`) |
| `n_samples` | `int` | Number of projected entries |
| `seed` | `int` | Random seed used |

---

## 5. Algorithm Design

### Top-K search (O(N + k log k) via `argpartition`)

```python
all_sims = DistanceMetrics.cosine_similarity_batch(query_vec, vectors)  # O(N·D)
all_sims[query_idx] = -np.inf                                           # exclude self
effective_k = min(top_k, n_vocab - 1)
top_k_unordered = np.argpartition(all_sims, -effective_k)[-effective_k:]  # O(N)
sorted_order    = np.argsort(all_sims[top_k_unordered])[::-1]             # O(k log k)
top_k_indices   = top_k_unordered[sorted_order]
```

Avoids a full O(N log N) sort over the vocabulary. `effective_k = min(top_k, N-1)` keeps it safe even when `top_k > N-1`.

### Cosine KMeans (equivalence under L2 normalization)

For unit vectors `â = a/‖a‖` and `b̂ = b/‖b‖`:

```
‖â - b̂‖² = ‖â‖² - 2·â·b̂ + ‖b̂‖² = 2 - 2·cos(a, b)
```

In the normalized space, minimizing Euclidean distance is equivalent to minimizing cosine distance, so cosine KMeans is achievable using `sklearn.KMeans` (which assumes Euclidean distance).

### PCA contribution rate

Using the eigenvalues `λ_i` of the covariance matrix of the input matrix `X ∈ R^{N×D}`:

```
Z = X · V^T             (V: matrix whose rows are the top 2 principal components)
contribution_rate_i = λ_i / Σ_j λ_j
```

`Projector._fit_pca` uses `sklearn.PCA(n_components=2, random_state=seed)` and stores `explained_variance_ratio_.tolist()` in `ProjectionResult.explained_variance`.

### Z-score computation

```
z = (top1_similarity - mean) / std
```

When `std == 0.0`, `z_score = 0.0` (avoids zero division). Computed against the similarity distribution over all N-1 entries (self excluded).

---

## 6. Constants and Configuration Values

### Main constants

| Constant | Value | Defined in | Description |
|----------|-------|------------|-------------|
| `_EPSILON` | `1e-10` | `distance_metrics.py` | Zero-division guard threshold |
| `SUPPORTED_METHODS` | `("pca", "umap")` | `projection.py` | Valid projection method names |
| `DEFAULT_TOP_K` | `10` | `similarity_engine.py` | Default Top-K for `search()` |
| `DEFAULT_N_CLUSTERS` | `8` | `clustering.py` | Default KMeans cluster count |
| `DEFAULT_SEED` | `42` | `clustering.py`, `projection.py` | Default random seed |
| `DEFAULT_MAX_ITER` | `300` | `clustering.py` | KMeans maximum iterations |
| `N_HISTOGRAM_BINS` | `50` | `analyzer.py` | Default histogram bin count |

### Seed fixing points

| Location | How the seed is passed |
|----------|------------------------|
| `KMeansClusterer.__init__` | `KMeans(random_state=self._seed)` |
| `Projector.__init__` (PCA) | `PCA(random_state=self._seed)` |
| `Projector.__init__` (UMAP) | `UMAP(random_state=self._seed)` |
| `ui/app.py` Tab 4 | Passed explicitly via `KMeansClusterer(seed=42)` / `Projector(seed=42)` |

`ClusterResult.seed` / `ProjectionResult.seed` retain the seed used, so reproducibility can be verified from the result.

### Exception class hierarchy

```
Exception
├── EmbeddingLoaderError
│   ├── IndexAlignmentError       # N (vocabulary size) mismatch / non-contiguous indices
│   └── ManifestViolationError    # shape / dtype diverges from manifest
├── SimilarityEngineError
│   ├── UnknownWordError          # Out-of-vocabulary word
│   └── InvalidTopKError          # top_k < 1
├── DistanceMetricsError
│   └── VectorDimensionError      # Invalid dimensionality / shape
├── POSFilterError
│   └── UnknownPOSTagError        # POS not present in the results
├── AnalyzerError
│   └── InsufficientDataError     # Statistical computation impossible (e.g., empty histogram_data)
├── ClusterError
│   ├── NotFittedError            # Access before fit
│   ├── InvalidClusterCountError  # Invalid n_clusters
│   └── UnfitVectorError          # Invalid input vector shape
└── ProjectionError
    ├── NotFittedError            # Access before fit (a separate class from clustering.py)
    ├── InvalidMethodError        # Method other than "pca" / "umap" specified
    └── InvalidVectorError        # Invalid input vector shape
```
