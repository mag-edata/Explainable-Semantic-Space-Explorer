# Test Cases Document — Explainable Semantic Space Explorer

**Project name:** Explainable Semantic Space Explorer  
**Version:** 1.0  
**Date created:** 2026-05-05

**Legend**

| Symbol | Meaning |
|--------|---------|
| Priority: H / M / L | High / Medium / Low |
| Type: UT / ST | Unit test / System test |
| Result: O / X / △ / - | Pass / Fail / Conditional pass / Not executed |

---

## 1. DistanceMetrics Test Cases

**Target files:** `tests/test_distance_metrics.py` → `core/distance_metrics.py`

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| DM-01 | `l2_norm([3.0, 4.0])` returns `5.0` (Pythagorean theorem) | H | UT | O |
| DM-02 | `l2_norm([1.0, 0.0, 0.0])` returns `1.0` (unit vector) | H | UT | O |
| DM-03 | `l2_norm(np.zeros(4))` returns `0.0` (zero vector) | H | UT | O |
| DM-04 | `l2_norm([1, 1, 1, 1])` returns `2.0` (= √4) | M | UT | O |
| DM-05 | The return type of `l2_norm` is Python `float` (not a numpy scalar) | M | UT | O |
| DM-06 | Passing a Python list to `l2_norm` raises `TypeError` | M | UT | O |
| DM-07 | Passing a 2D array of shape (2, 2) to `l2_norm` raises `VectorDimensionError` | H | UT | O |
| DM-08 | For a 300-dimensional random vector, `l2_norm` matches `sqrt(sum(v²))` (places=4) | H | UT | O |
| DM-09 | `cosine_similarity([1,2,3], [2,4,6])` returns `1.0` (parallel vectors) | H | UT | O |
| DM-10 | `cosine_similarity([1,2,3], [-1,-2,-3])` returns `-1.0` (anti-parallel vectors) | H | UT | O |
| DM-11 | `cosine_similarity([1,0], [0,1])` returns `0.0` (orthogonal vectors) | H | UT | O |
| DM-12 | When one side is a zero vector, `cosine_similarity` returns `0.0` (no NaN) | H | UT | O |
| DM-13 | For 20 pairs of 50-dimensional random vectors, all `cosine_similarity` results fall within `[-1.0, 1.0]` | H | UT | O |
| DM-14 | `cosine_similarity(a, b) == cosine_similarity(b, a)` holds (symmetry) | H | UT | O |
| DM-15 | Passing vectors with different dimensions raises `VectorDimensionError` | H | UT | O |
| DM-16 | `cosine_similarity_batch(query(3,), matrix(4,3))` outputs shape `(4,)` | H | UT | O |
| DM-17 | When batch row 0 is identical to the query, `result[0] == 1.0` | H | UT | O |
| DM-18 | When batch row 1 is orthogonal, `result[1] == 0.0` | H | UT | O |
| DM-19 | When batch row 2 is anti-parallel, `result[2] == -1.0` | H | UT | O |
| DM-20 | When batch row 3 is at 45 degrees, `result[3] == 1/√2` | M | UT | O |
| DM-21 | Even when the matrix contains zero rows, no NaN occurs and the zero-row result is `0.0` | H | UT | O |
| DM-22 | `cosine_similarity_batch` results match the single-pair computation (places=4, 10-row batch) | H | UT | O |
| DM-23 | When the query and matrix dimensions do not match, `VectorDimensionError` is raised | H | UT | O |
| DM-24 | The dictionary returned by `explain` contains all 6 keys: `dot_product`, `norm_a`, `norm_b`, `denominator`, `similarity`, `formula` | M | UT | O |
| DM-25 | The `formula` field in `explain` is of type `str` | M | UT | O |
| DM-26 | Calling `explain` with orthogonal vectors returns a dictionary with `similarity == 0.0` | H | UT | O |
| DM-27 | Calling `explain` with parallel vectors returns a dictionary with `similarity == 1.0` | H | UT | O |
| DM-28 | The `dot_product` field of `explain` matches `np.dot(a, b)` | H | UT | O |
| DM-29 | The `norm_a` and `norm_b` fields of `explain` are `>= 0.0` (non-negativity of norm) | M | UT | O |

---

## 2. EmbeddingLoader Test Cases

**Target files:** `tests/test_embedding_loader.py` → `core/embedding_loader.py`

The tests use the `_build_mock_assets()` helper to dynamically generate small mock assets (n=5, static_dim=4, contextual_dim=8) inside a `tempfile.TemporaryDirectory`.

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| EL-01 | After `load_all()`, `static_vectors.shape == (5, 4)` (n=5, dim=4 mock) | H | UT | O |
| EL-02 | After `load_all()`, `contextual_vectors.shape == (5, 8)` (n=5, contextual_dim=8 mock) | H | UT | O |
| EL-03 | After `load_all()`, `vocab` is of type `dict` (converted from list form) | H | UT | O |
| EL-04 | After `load_all()`, `len(vocab) == 5` (n=5 mock) | H | UT | O |
| EL-05 | After `load_all()`, `pos.shape == (5,)` | H | UT | O |
| EL-06 | After `load_all()`, `manifest` is a dict and contains the `"static_vectors"` key | M | UT | O |
| EL-07 | The set of index values in `vocab` equals `set(range(N))` (zero-based contiguous indices) | H | UT | O |
| EL-08 | `static_vectors.dtype == np.float32` | H | UT | O |
| EL-09 | Overwriting `manifest`'s static_shape with `(99, 4)` raises `ManifestViolationError` | H | UT | O |
| EL-10 | Overwriting `manifest`'s contextual_shape with `(5, 999)` raises `ManifestViolationError` | H | UT | O |
| EL-11 | Overwriting `manifest`'s dtype with `"float64"` (while the actual data is float32) raises `ManifestViolationError` | H | UT | O |
| EL-12 | With a vocab of 5 entries, reducing `vocab_pos.npy` to 3 entries raises `IndexAlignmentError` (N mismatch) | H | UT | O |
| EL-13 | When vocab indices have gaps (`{"word0":0,"word1":1,"word2":9}`), `IndexAlignmentError` is raised | H | UT | O |
| EL-14 | When vocab indices have duplicates (`{"word0":0,"word1":0}`), `IndexAlignmentError` is raised | H | UT | O |
| EL-15 | Creating an `EmbeddingLoader` with a non-existent path raises `FileNotFoundError` | H | UT | O |
| EL-16 | Calling `load_all()` with `manifest.json` missing raises `FileNotFoundError` | H | UT | O |
| EL-17 | Calling `load_all()` with `static_vectors.npy` missing raises `FileNotFoundError` | H | UT | O |

---

## 3. SimilarityEngine Test Cases

**Target files:** `tests/test_similarity_engine.py` → `core/similarity_engine.py`

The tests use the `_make_engine()` helper to build a small engine with n=6 (or 10) and dim=4 via `numpy.random.default_rng(seed)`.

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| SE-01 | The return value of `search("word0", top_k=3)` is `list[SearchResult]` | H | UT | O |
| SE-02 | The number of results returned by `search("word0", top_k=3)` is `<= 3` | H | UT | O |
| SE-03 | The query word itself (`"word0"`) is not included in the `search` results (self-exclusion) | H | UT | O |
| SE-04 | The list returned by `search` is sorted by `similarity` in descending order | H | UT | O |
| SE-05 | In the `search` results, `results[0].rank == 1` (rank starts at 1) | H | UT | O |
| SE-06 | The `explanation` of the `search` results contains the `"formula"` and `"dot_product"` keys | M | UT | O |
| SE-07 | For `search("word0", pos_filter="NOUN")`, every result has `pos_tag == "NOUN"` | M | UT | O |
| SE-08 | `search("nonexistent_word")` raises `UnknownWordError` | H | UT | O |
| SE-09 | `search("word0", top_k=0)` raises `InvalidTopKError` | H | UT | O |
| SE-10 | `search("word0", top_k=-1)` raises `InvalidTopKError` | H | UT | O |
| SE-11 | The return value of `compare` is `ComparisonResult` | M | UT | O |
| SE-12 | The `query_word` field of the `compare` result matches the query word | M | UT | O |
| SE-13 | In the `compare` result, both `static_results` and `contextual_results` have a count `<= top_k` | M | UT | O |
| SE-14 | In the `compare` result, `common_words` equals `static_words ∩ contextual_words` | M | UT | O |
| SE-15 | In the `compare` result, the key set of `rank_diff` matches `common_words` | M | UT | O |
| SE-16 | `compare("no_such_word", ...)` raises `UnknownWordError` | H | UT | O |
| SE-17 | The return value of `get_distance_distribution("word0")` is a `dict` | M | UT | O |
| SE-18 | The returned dictionary contains all 6 keys: `query_word`, `mean`, `std`, `top1_similarity`, `z_score`, `histogram_data` | H | UT | O |
| SE-19 | The `query_word` field of the returned dictionary matches the query word | M | UT | O |
| SE-20 | `top1_similarity >= mean - 1e-6` (Top-1 is at least the mean) | H | UT | O |
| SE-21 | `len(histogram_data) == n - 1` (excluding the query itself, n=10) | H | UT | O |
| SE-22 | `std >= 0.0` (standard deviation is non-negative) | H | UT | O |
| SE-23 | `get_distance_distribution("no_such_word")` raises `UnknownWordError` | H | UT | O |
| SE-24 | `word_to_index` returns the correct index for every word in the vocab | H | UT | O |
| SE-25 | `word_to_index("unknown_word_xyz")` raises `UnknownWordError` | H | UT | O |

---

## 4. POSFilter Test Cases

**Target files:** `tests/test_pos_filter.py` → `core/pos_filter.py`

The tests use `_sample_results()`, which directly constructs 6 `SearchResult` objects (NOUN×3 / VERB×2 / ADJ×1).

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| PF-01 | `filter("NOUN")` returns only SearchResults with the NOUN tag | H | UT | O |
| PF-02 | The return value of `filter` preserves the order of the original results | M | UT | O |
| PF-03 | The return value of `filter` preserves the original `rank` (overall rank) | M | UT | O |
| PF-04 | `filter("")` raises `ValueError` | M | UT | O |
| PF-05 | Specifying a POS not present in the results raises `UnknownPOSTagError` | H | UT | O |
| PF-06 | The return value of `group_by_pos` is of type `dict` | M | UT | O |
| PF-07 | The key set of `group_by_pos` matches every POS appearing in the results | H | UT | O |
| PF-08 | The count per POS group is correct (NOUN=3 / VERB=2 / ADJ=1) | H | UT | O |
| PF-09 | The order within each group preserves the original results order | M | UT | O |
| PF-10 | `group_by_pos` returns an empty dict for empty results | M | UT | O |
| PF-11 | `pos_distribution` correctly aggregates the count for each POS | H | UT | O |
| PF-12 | `pos_distribution` is sorted by count in descending order | H | UT | O |
| PF-13 | The total of `pos_distribution` matches the length of the results | M | UT | O |
| PF-14 | `pos_distribution` returns an empty dict for empty results | M | UT | O |
| PF-15 | `heterogeneity_rate(query_pos="NOUN")` returns 3/6=0.5 | H | UT | O |
| PF-16 | `heterogeneity_rate(query_pos="VERB")` returns 4/6 | H | UT | O |
| PF-17 | When every result shares the same POS, `heterogeneity_rate` returns 0.0 | H | UT | O |
| PF-18 | When every result has a different POS, `heterogeneity_rate` returns 1.0 | H | UT | O |
| PF-19 | The return value of `heterogeneity_rate` falls within `[0.0, 1.0]` | M | UT | O |
| PF-20 | Calling `heterogeneity_rate` with empty results raises `ValueError` | H | UT | O |
| PF-21 | Calling `heterogeneity_rate` with `query_pos=""` raises `ValueError` | M | UT | O |
| PF-22 | After `assign_pos_ranks`, the first `pos_rank` of each POS group starts at 1 | H | UT | O |
| PF-23 | After `assign_pos_ranks`, `pos_rank` within the same POS is consecutive: 1, 2, 3 ... | H | UT | O |
| PF-24 | `assign_pos_ranks` does not break the order of the original results | M | UT | O |
| PF-25 | The number of items returned by `assign_pos_ranks` matches the input | M | UT | O |
| PF-26 | `top_pos()` (n=3) returns the top 3 most frequent POS in descending order | M | UT | O |
| PF-27 | `top_pos(n=1)` returns only the most frequent POS | M | UT | O |
| PF-28 | Even when n exceeds the number of distinct POS, the returned count does not exceed the available POS | M | UT | O |
| PF-29 | `top_pos(n=0)` raises `ValueError` | M | UT | O |

---

## 5. Analyzer Test Cases

**Target files:** `tests/test_analyzer.py` → `core/analyzer.py`

The tests use `_make_distribution()` to build a dict that imitates the return value of `get_distance_distribution()`, and `_make_search_result()` to construct a `SearchResult`.

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| AN-01 | The return value of `enrich_distribution` is `DistributionStats` | H | UT | O |
| AN-02 | `query_word` is carried through | M | UT | O |
| AN-03 | The `median` field matches `numpy.median` | H | UT | O |
| AN-04 | The relationship `q25 <= median <= q75` holds | H | UT | O |
| AN-05 | `n_samples` matches the length of `histogram_data` | M | UT | O |
| AN-06 | When a required key is missing from the distribution, `KeyError` is raised | H | UT | O |
| AN-07 | Empty `histogram_data` raises `InsufficientDataError` | H | UT | O |
| AN-08 | The return value of `histogram` is `HistogramData` | M | UT | O |
| AN-09 | The length of `bin_edges` is `n_bins + 1` | H | UT | O |
| AN-10 | The length of `counts` matches `n_bins` | H | UT | O |
| AN-11 | The sum of `counts` matches the number of data points | H | UT | O |
| AN-12 | `data_min` / `data_max` match the minimum and maximum of the data | M | UT | O |
| AN-13 | Calling `histogram` with empty data raises `InsufficientDataError` | H | UT | O |
| AN-14 | Calling `histogram` with `n_bins=0` raises `ValueError` | M | UT | O |
| AN-15 | The return value of `attach_z_scores` is a list of `dict` | M | UT | O |
| AN-16 | The number of items returned by `attach_z_scores` matches the results | H | UT | O |
| AN-17 | Each element contains the required keys (word/rank/similarity/pos_tag/pos_rank/z_score/explanation) | H | UT | O |
| AN-18 | `z_score` matches `(similarity - mean) / std` | H | UT | O |
| AN-19 | When `std=0`, `z_score=0.0` is returned safely (zero-division guard) | H | UT | O |
| AN-20 | The return value of `compare_distributions` is `DistributionComparison` | M | UT | O |
| AN-21 | The `query_word` of `compare_distributions` is carried through | M | UT | O |
| AN-22 | `mean_diff = static.mean - contextual.mean` matches | H | UT | O |
| AN-23 | `std_diff = static.std - contextual.std` matches | H | UT | O |
| AN-24 | `z_score_diff = static.z_score - contextual.z_score` matches | H | UT | O |
| AN-25 | `static_stats` / `contextual_stats` are stored as the `DistributionStats` type | M | UT | O |
| AN-26 | `neighborhood_stability` returns 1.0 when there is full agreement | H | UT | O |
| AN-27 | `neighborhood_stability` returns 0.0 when there is no agreement | H | UT | O |
| AN-28 | Returns the correct Jaccard coefficient on partial overlap (intersection=2, union=4 → 0.5) | H | UT | O |
| AN-29 | The return value of `neighborhood_stability` falls within `[0.0, 1.0]` | M | UT | O |
| AN-30 | Calling `neighborhood_stability` with empty `static_results` raises `ValueError` | H | UT | O |
| AN-31 | Calling `neighborhood_stability` with empty `contextual_results` raises `ValueError` | H | UT | O |

---

## 6. KMeansClusterer Test Cases

**Target files:** `tests/test_clustering.py` → `transforms/clustering.py`

The tests use `_random_vectors()` for random generation and `_separable_vectors()` with 3 centers.

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| CL-01 | Can be instantiated with default arguments | M | UT | O |
| CL-02 | Can be instantiated with custom arguments (n_clusters=5, seed=7, max_iter=100) | M | UT | O |
| CL-03 | `n_clusters=0` raises `InvalidClusterCountError` | H | UT | O |
| CL-04 | `n_clusters=-1` raises `InvalidClusterCountError` | H | UT | O |
| CL-05 | When `n_clusters` is not an int, `TypeError` is raised | M | UT | O |
| CL-06 | When `seed` is not an int, `TypeError` is raised | M | UT | O |
| CL-07 | The return value of `fit` is `ClusterResult` | H | UT | O |
| CL-08 | The shape of `labels` is `(N,)` | H | UT | O |
| CL-09 | Values in `labels` fall within `[0, n_clusters - 1]` | H | UT | O |
| CL-10 | The `n_samples` field matches the input count | M | UT | O |
| CL-11 | The `n_clusters` field matches the specified value | M | UT | O |
| CL-12 | The `seed` field matches the specified value | M | UT | O |
| CL-13 | `inertia` is non-negative | H | UT | O |
| CL-14 | On clearly separated data, nearby points belong to the same cluster | H | UT | O |
| CL-15 | Anything other than `np.ndarray` raises `UnfitVectorError` | H | UT | O |
| CL-16 | Passing a 1-dimensional array raises `UnfitVectorError` | H | UT | O |
| CL-17 | `n_clusters > N` raises `InvalidClusterCountError` | H | UT | O |
| CL-18 | After `fit`, labels can be obtained via `get_labels` | H | UT | O |
| CL-19 | `get_labels` before `fit` raises `NotFittedError` | H | UT | O |
| CL-20 | After `fit`, a `ClusterResult` can be obtained via `get_result` | H | UT | O |
| CL-21 | `get_result` before `fit` raises `NotFittedError` | H | UT | O |
| CL-22 | `_l2_norm_batch([[3,4],[0,0],[1,0]])` returns `[5,0,1]` | H | UT | O |
| CL-23 | The output shape of `_l2_norm_batch` is `(N,)` | M | UT | O |
| CL-24 | The norms produced by `_l2_norm_batch` are non-negative | M | UT | O |
| CL-25 | After `_normalize_rows`, the norm of every non-zero row is 1.0 | H | UT | O |
| CL-26 | Zero-vector rows remain zero after normalization (no NaN) | H | UT | O |
| CL-27 | `_normalize_rows` does not mutate the input matrix | M | UT | O |
| CL-28 | With the same seed and same data, `labels` match exactly (reproducibility) | H | UT | O |
| CL-29 | With the same seed, `inertia` also matches exactly (reproducibility) | H | UT | O |

---

## 7. Projector Test Cases

**Target files:** `tests/test_projection.py` → `transforms/projection.py`

The tests use `_random_vectors(n=20, dim=8)`. UMAP is kept to a small scale to limit computational cost.

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| PR-01 | The default method is `pca` | M | UT | O |
| PR-02 | Can be instantiated with `method='pca'` | M | UT | O |
| PR-03 | Can be instantiated with `method='umap'` | M | UT | O |
| PR-04 | An unsupported method name (e.g., "tsne") raises `InvalidMethodError` | H | UT | O |
| PR-05 | When `method` is not a str, `TypeError` is raised | M | UT | O |
| PR-06 | When `seed` is not an int, `TypeError` is raised | M | UT | O |
| PR-07 | The return value of PCA is `ProjectionResult` | H | UT | O |
| PR-08 | The shape of `coords_2d` for PCA is `(N, 2)` | H | UT | O |
| PR-09 | The `method` field of PCA is "pca" | M | UT | O |
| PR-10 | The number of elements in `explained_variance` for PCA is 2 | H | UT | O |
| PR-11 | Each PCA explained-variance ratio falls within `[0.0, 1.0]` | H | UT | O |
| PR-12 | The sum of PCA explained-variance ratios is at most 1.0 | H | UT | O |
| PR-13 | The first principal component's variance ratio is at least that of the second | H | UT | O |
| PR-14 | The `n_samples` of PCA matches the input count | M | UT | O |
| PR-15 | The `seed` field of PCA matches the specified value | M | UT | O |
| PR-16 | Immediately after `fit_transform`, `cluster_labels` is None | M | UT | O |
| PR-17 | The shape of `coords_2d` for UMAP is `(N, 2)` | H | UT | O |
| PR-18 | The `method` field of UMAP is "umap" | M | UT | O |
| PR-19 | `explained_variance` for UMAP is an empty list | H | UT | O |
| PR-20 | Anything other than `np.ndarray` raises `InvalidVectorError` | H | UT | O |
| PR-21 | A 1-dimensional array raises `InvalidVectorError` | H | UT | O |
| PR-22 | A single sample raises `InvalidVectorError` | H | UT | O |
| PR-23 | The return value of `attach_clusters` is `ProjectionResult` | M | UT | O |
| PR-24 | After `attach_clusters`, `cluster_labels` matches the specified label array | H | UT | O |
| PR-25 | `attach_clusters` does not mutate the original result (immutable) | H | UT | O |
| PR-26 | The return value of `attach_clusters` is a different object from the original result | M | UT | O |
| PR-27 | `attach_clusters` carries over `coords_2d` / `explained_variance` / `method` | M | UT | O |
| PR-28 | When the label length does not match `n_samples`, `InvalidVectorError` is raised | H | UT | O |
| PR-29 | When result is not a `ProjectionResult`, `TypeError` is raised | M | UT | O |
| PR-30 | When `cluster_labels` is not an `np.ndarray`, `TypeError` is raised | M | UT | O |
| PR-31 | With the same seed for PCA, `coords_2d` matches exactly (reproducibility) | H | UT | O |
| PR-32 | With the same seed for UMAP, `coords_2d` matches exactly (reproducibility) | H | UT | O |

---

## 8. UI Test Cases

**Target file:** `ui/app.py` (manual verification in the browser)

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| A-01 | The Streamlit app starts and displays the title and 4 tabs | H | ST | O |
| A-02 | On the search tab, entering and submitting a query word displays Top-K similar words for Word2Vec / SBERT in two columns | H | ST | O |
| A-03 | Each result row displays the similarity score, dot product, norm (cosine breakdown), and POS label | H | ST | O |
| A-04 | The compare tab displays the number of common words, the number of unique words, and the rank-diff ranking | M | ST | O |
| A-05 | The projection tab displays a 2D scatter plot corresponding to the selected PCA / UMAP method | M | ST | O |
| A-06 | The cluster tab displays a scatter plot color-coded by cluster and a list of representative words | M | ST | O |
| A-07 | A loading spinner is displayed while a search is running | M | ST | O |

---

## 9. Error Case Test Cases

**Target:** Error messages E001–E006 (red text display via `st.error()`, manual verification)

| No. | Test summary | Priority | Type | Result |
|-----|--------------|----------|------|--------|
| X-01 | Submitting a search with an empty string displays E001 "Please enter a query word." | H | ST | O |
| X-02 | Entering an out-of-vocabulary word displays E002 "<word> is not in the vocabulary." | H | ST | O |
| X-03 | When `.npy` / `.json` files under `data/` are not found at startup, E003 is displayed | H | ST | - |
| X-04 | When the shape / dtype of `manifest.json` does not match the actual files, E004 is displayed | H | ST | - |
| X-05 | Specifying a Top-K value of 0 or less displays E005 "Top-K must be specified as an integer of 1 or greater." | M | ST | O |
| X-06 | Specifying a cluster count of 1 or less displays E006 "The cluster count must be specified as an integer of 2 or greater." | M | ST | O |

---

## Test Execution Summary

| Item | Detail |
|------|--------|
| Execution date | 2026-05-06 (UT expansion), 2026-05-05 (ST manual) |
| Executor | mag |
| Total tests | 205 (UT: 192, ST: 13) |
| Pass (O) | 203 (DM-01–29, EL-01–17, SE-01–25, PF-01–29, AN-01–31, CL-01–29, PR-01–32, A-01–07, X-01, X-02, X-05, X-06) |
| Not executed (-) | 2 (X-03, X-04) |
| Fail (X) | 0 |

**Notes**

- All 71 UT cases (§1–§3) have been executed with `venv/bin/python3 -m unittest discover tests/ -v` and all pass.
- ST (§4 / §5) is manual verification in the browser. Because automated tests for Streamlit session state are not in place, verification is by visual inspection.
- X-03 / X-04 require the operation of intentionally removing or tampering with `data/`, so they will be executed manually after the production assets are deployed.
