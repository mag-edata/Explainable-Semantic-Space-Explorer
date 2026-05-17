# Test Design Document — Explainable Semantic Space Explorer

**Project Name:** Explainable Semantic Space Explorer
**Version:** 1.0
**Date:** 2026-05-05

---

## 1. Testing Policy

- The top priorities of this project are "explainability" and "reproducibility." Therefore, rather than UI appearance or speed, we focus on **numerical correctness of logic, exception boundaries, and guarantees of index consistency**.
- Since no external APIs are used, the purpose of mocks is not to block network access but is limited to speeding up tests by excluding large real assets (83,823 vocabulary × 300 / 384 dimensions). All test data is dynamically generated in memory.
- Because project constraints prohibit the use of external libraries for cosine similarity, the correctness of the custom implementation is ensured by **matching analytical solutions and verifying mathematical properties** using known vectors (comparisons with `scipy` / `sklearn` are not used).
- Tests are conducted by a single developer using `unittest` and manual verification. External test libraries such as `pytest` are not introduced.

---

## 2. Test Scope

**In Scope**

- `core/embedding_loader.py` (asset loading and manifest consistency check logic)
- `core/distance_metrics.py` (custom implementation of cosine similarity and norms)
- `core/similarity_engine.py` (Top-K search, model-to-model comparison, distance distribution calculation)
- `core/pos_filter.py` (POS filtering, grouping, cross-POS ratio, intra-POS ranking)
- `core/analyzer.py` (distribution statistics extension, histogram, Z-score annotation, neighborhood stability)
- `transforms/clustering.py` (cosine KMeans, custom norm/normalization, reproducibility)
- `transforms/projection.py` (PCA / UMAP, explained variance ratio, attach_clusters, reproducibility)
- `ui/app.py` (UI operation flow — manual verification only)
- Error cases (unknown words, shape mismatch, invalid `top_k` values, non-contiguous indices, etc.)

**Out of Scope**

- Behavior of the Streamlit framework itself
- Contents of actual data files under `data/` (tests use dynamically generated mocks)

---

## 3. Test Level Definitions

### Unit Tests

| Item | Details |
|------|---------|
| Target | Individual methods of each class in `core/` |
| Method | `unittest.TestCase` (covering happy path, error path, and boundary values within a single file) |
| Timing | Upon completion of each module's implementation |
| Pass Criteria | All test cases PASS |

### Integration Tests

| Item | Details |
|------|---------|
| Target | Combination of `DistanceMetrics` → `SimilarityEngine` |
| Method | Inject a `DistanceMetrics` instance within `test_similarity_engine.py` for verification |
| Timing | Upon completion of each module's implementation |
| Pass Criteria | Search results are correctly returned and scores fall within the `[-1.0, 1.0]` range |

### System Tests

| Item | Details |
|------|---------|
| Target | All UI flows across the 4 Streamlit tabs (Search, Compare, Projection, Cluster) |
| Method | Manual operation in a browser |
| Timing | Final verification before deployment |
| Pass Criteria | Main use cases (query word input → similar word display) complete successfully |

---

## 4. Test Environment

**Local Environment**

| Item | Details |
|------|---------|
| OS | macOS (development machine) |
| Python | 3.12 (venv) |
| Real assets | Not required at test execution time (dynamically generated mocks are used) |
| External APIs | Not used |

**Test Configuration**

- Use `tempfile.TemporaryDirectory` to generate temporary mock assets (`embeddings/`, `metadata/`, `manifest.json`), achieving test isolation (see `setUp` / `tearDown` in `tests/test_embedding_loader.py`).
- The execution command is standardized as follows.

```bash
venv/bin/python3 -m unittest discover tests/ -v
```

**Post-Streamlit Deployment Verification**

- After deployment, system tests are conducted by manually operating the UI.

---

## 5. Test Data Policy

### Data Generation Method per Test File

| Test File | Data Generation Method |
|-----------|------------------------|
| `tests/test_distance_metrics.py` | No real assets used. A mix of known vectors (with determinate analytical solutions) and random vectors generated via `numpy.random.default_rng(seed)`. |
| `tests/test_embedding_loader.py` | The `_build_mock_assets()` helper dynamically generates a mini asset of size `n=5` in a `tempfile.TemporaryDirectory`. shape / dtype / vocab can be overridden individually via arguments, allowing error-path scenarios to be described declaratively. |
| `tests/test_similarity_engine.py` | The `_make_engine()` helper generates small-scale vectors (e.g., `n=6, dim=4`) using `numpy.random.default_rng(seed)` and assembles `SimilarityEngine` directly. |
| `tests/test_pos_filter.py` | The `_make_result()` / `_sample_results()` helpers directly construct SearchResults with NOUN×3 / VERB×2 / ADJ×1. No real assets used. |
| `tests/test_analyzer.py` | The `_make_distribution()` helper generates a dict mimicking the return value of `get_distance_distribution()`, and combinations with SearchResult are verified. |
| `tests/test_clustering.py` | Random generation via `_random_vectors()` and three-center `_separable_vectors()` are used to verify clustering correctness. Full reproducibility under the same seed is confirmed. |
| `tests/test_projection.py` | `_random_vectors()` generates a small matrix of n=20, dim=8. UMAP is restricted to small scale to limit computational cost, and reproducibility under a fixed seed is confirmed for both PCA and UMAP. |

**Test Data Principles**

- Do not create a persistent `fixtures/` directory (reproducibility is ensured by fixing seeds).
- Temporary files are reliably disposed of in `setUp` / `tearDown`.

---

## 6. Pass Criteria

### Unit Tests

- All test cases PASS (0 errors, 0 failures).
- `l2_norm`, `cosine_similarity`, and `cosine_similarity_batch` match analytical solutions (norm of `[3,4]` = `5.0`, parallel vectors = `1.0`, orthogonal = `0.0`, opposite direction = `-1.0`).
- The following custom exceptions are reliably raised under the expected boundary conditions.

| Exception Class | Test Target Scenario |
|---|---|
| `VectorDimensionError` | Dimension mismatch, input other than 1-dimensional array |
| `ManifestViolationError` | shape / dtype mismatch with `manifest.json` |
| `IndexAlignmentError` | vocab count ≠ pos count, non-contiguous indices, duplicate indices |
| `FileNotFoundError` | `manifest.json` missing, `static_vectors.npy` missing, root directory absent |
| `UnknownWordError` | Search / comparison / distance distribution request for an out-of-vocabulary word |
| `InvalidTopKError` | `top_k=0`, specifying a negative value |

### Integration Tests

- The number of items returned by `SimilarityEngine.search` matches the requested `top_k`.
- Similarity scores fall within the `[-1.0, 1.0]` range.
- Search results are sorted in descending order of similarity.

### System Tests

- The full flow of query word input → similar word display → cosine breakdown display completes without errors.
- An appropriate error message (E002) is displayed when an out-of-vocabulary word is entered.
- All 4 Streamlit tabs can be navigated to.

---

## 7. Quality Metrics

### Number of Tests and PASS Rate

| Item | Details |
|------|---------|
| Definition | Number of test cases passing in full and the PASS rate |
| Measurement Method | Output count from `unittest discover tests/ -v` |
| Target | Maintain 192 cases at a 100% PASS rate |

The breakdown is as follows.

| Test File | Count |
|---|---|
| `tests/test_distance_metrics.py` | 29 |
| `tests/test_similarity_engine.py` | 26 |
| `tests/test_embedding_loader.py` | 16 |
| `tests/test_pos_filter.py` | 29 |
| `tests/test_analyzer.py` | 31 |
| `tests/test_clustering.py` | 29 |
| `tests/test_projection.py` | 32 |
| Total | 192 |

### Numerical Reproducibility

| Item | Details |
|------|---------|
| Definition | Whether the same Top-K results, cluster IDs, and projection coordinates are obtained under the same seed |
| Measurement Method | Verification of agreement through repeated executions with a fixed seed |
| Target | 100% reproducibility (CONST-06: fix the seed when using random numbers) |

### Test Execution Time

| Item | Details |
|------|---------|
| Definition | Time required for full execution via `unittest discover` |
| Measurement Method | Actual measured value during local execution |
| Reference Value | Approximately 3.7 seconds (most of which is occupied by reproducibility verification of UMAP projection; the rest is under 0.2 seconds) |

> Automated measurement of end-to-end performance (target: within 5 seconds) at real asset scale (83,823 vocabulary) is left as future work.
