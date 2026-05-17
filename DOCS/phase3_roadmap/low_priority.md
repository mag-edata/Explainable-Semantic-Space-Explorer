# (Low) Planned Improvements

This document consolidates the 25 low-priority issues (refactoring / code quality). For details and the rationale behind the priority decisions, refer to `overview.md`.

---

## 1. Data Consistency (remaining)

### 1.3 `training_date` Placeholder Left in `manifest.json`

- **Location:** `data/manifest.json:9`
- **Symptom:** `"training_date": "YYYY-MM-DD"` is left as-is. This is the literal in the generation script `data_pipeline/manifest.py:71`.
- **Impact:** Unrelated to the alignment check, but it diminishes the completeness of the file as a publicly released artifact.
- **Action:** Embed `datetime.date.today().isoformat()` in `manifest.py`, or remove the field itself.

---

## 3. UI / Streamlit (remaining)

### 3.2 Tab 1: `heterogeneity_rate` Exception when POS Filter is Specified

- **Location:** `ui/app.py:209-261`
- **Symptom:**
  - When `pos_filter` is specified as "verb" etc. in the sidebar and the result is 0 items, `static_engine.search` returns an empty list.
  - The subsequent `POSFilter.heterogeneity_rate(static_results, query_pos)` then halts the UI with `ValueError("results is empty")`.
- **`UnknownPOSTagError`** is a similar concern: when narrowing by POS via `POSFilter.filter`, an exception is raised by design when there are no matches (the Engine side silently filters, so this does not occur when going through the Engine. Be careful about mixing them).
- **Action:** Handle the 0-result case with `st.warning` on the UI side first, then proceed to downstream processing.

### 3.3 Tab 1: Semantic Twist of "Heterogeneity Rate"

- **Location:** `ui/app.py:255-261`
- **Symptom:** When narrowing by POS with `pos_filter`, all search results have the same POS. Computing `heterogeneity_rate(...)` in that state always yields 0.0 or always yields 1.0 (depending on whether `query_pos` matches the filter target).
- **Impact:** A value that does not carry meaning as a "heterogeneity rate" metric is displayed. This causes confusion for users.
- **Action:** Display the heterogeneity rate only when `pos_filter is None` (or compute it on the results before `pos_filter` is applied).

### 3.5 Tab 4: Duplicate Construction of `index_to_word`

- **Location:** `ui/app.py:429`
- **Symptom:** `index_to_word: dict[int, str] = {v: k for k, v in loader.vocab.items()}` is constructed in a loop every time.
- **Issue:** `SimilarityEngine._index_to_word` already holds an equivalent list (although it is not exposed because it begins with `_`).
- **Action:** Either expose an `index_to_word(idx)` method on `SimilarityEngine`, or construct this dictionary once on the UI side using `@st.cache_resource`.

### 3.6 Type of `loader.pos[loader.vocab[query_word]]`

- **Location:** `ui/app.py:255`
- **Symptom:** A `numpy.str_` type is being assigned as-is to `query_pos: str`. The comparison inside `POSFilter.heterogeneity_rate` works without issue, but there is an inconsistency with the type hint.
- **Action:** Explicitly cast with `str(loader.pos[...])`.

---

## 4. Projection / Clustering Layer

### 4.1 Unused import in `transforms/projection.py`

- **Location:** `transforms/projection.py:26`
- **Symptom:** `from dataclasses import dataclass, field` imports `field`, but it is unused.
- **Action:** Remove `field`.

### 4.2 PCA Input Dimension D < 2 is Not Validated

- **Location:** `transforms/projection.py:313-338`
- **Symptom:** `_validate_inputs` validates `vectors.ndim != 2` and `shape[0] >= 2`, but does not validate `shape[1] >= 2`.
- **Impact:** Passing a matrix with D=1 to `PCA(n_components=2)` raises an exception on the sklearn side (`n_components > min(n_samples, n_features)`). The error message is hard to understand.
- **Action:** Add an explicit check for `vectors.shape[1] >= 2` in `_validate_inputs`.

### 4.3 float64 Upcast in `KMeansClusterer._normalize_rows`

- **Location:** `transforms/clustering.py:291`
- **Symptom:** `unit_matrix: np.ndarray = matrix.copy().astype(np.float64)` expands float32 to float64.
- **Impact:**
  - When the full vocabulary (83823 × 384) is passed in, approximately 256 MB inflates to 512 MB. This is borderline within Streamlit Cloud's memory limit (free tier: 1 GB).
  - However, UI Tab 4 only passes the Top-K neighbors, so this is not critical.
- **Action:** Normalize while keeping float32. Or explicitly comment on the reason for the `astype`.

### 4.4 sklearn `KMeans` `n_init` Unspecified

- **Location:** `transforms/clustering.py:191-195`
- **Symptom:** `KMeans(n_clusters=..., random_state=..., max_iter=...)` does not specify `n_init`.
- **Impact:** Since sklearn 1.4, `n_init='auto'` is the default, but behavior may change in future sklearn versions.
- **Action:** Explicitly specify `n_init=10` (the existing customary value).

### 4.5 Duplicate Distance Computation Logic

- **Location:** `transforms/clustering.py:248-267` and `core/distance_metrics.py:191-198`
- **Symptom:** Both modules independently implement `np.sqrt((matrix * matrix).sum(axis=1))`.
- **Impact:** A hotbed for forgetting to fix one side when fixing a bug.
- **Action:** Delegate from `clustering.py` to the equivalent method in `DistanceMetrics` (the dependency direction is OK: transforms → core is plausible, but currently there is no reverse reference).

---

## 5. core/ Layer

### 5.1 dtype Inconsistency in `cosine_similarity_batch` Docstring

- **Location:** `core/distance_metrics.py:179`
- **Symptom:** The docstring says `"dtype is float64"`, but the implementation follows the output dtype of `matrix @ query`. If `matrix` is float32, the result is also float32.
- **Action:** Correct the docstring to "dtype follows the input", or explicitly call `.astype(np.float64)`.

### 5.2 `rank` in `SimilarityEngine.search`'s `pos_filter` Return Value is Not Contiguous

- **Location:** `core/similarity_engine.py:255-260`
- **Symptom:** When filtered with `pos_filter`, the `rank` field of the remaining elements retains the original Top-K rank (non-contiguous) (e.g., rank=1, 4, 7, ...).
- **Judgment:** This is by design intent (preserving the overall rank), and is addressed by holding a separate `pos_rank` field. However, **on the UI display, the "rank" column becomes non-contiguous, which may confuse users**.
- **Action:** Either provide separate columns on the UI for "overall rank" and "filtered rank", or add help text supplementing the meaning of rank.

### 5.3 Type Hint of `EmbeddingLoader.vocab` is Optional

- **Location:** `core/embedding_loader.py:118`
- **Symptom:** It is initialized with `self.vocab: Dict[str, int] | None = None`, and only becomes non-None after `load_all()`. The caller must check for None every time (in practice, the check is omitted on the assumption that `load_all` has been called).
- **Action:** Call `load_all()` inside `__init__`, or provide a `from_path` class method that exposes only the fully constructed state.

---

## 6. data_pipeline/ Layer

### 6.1 `merge()` is Re-executed inside `train_w2v.py`

- **Location:** `data_pipeline/train/train_w2v.py:114`
- **Symptom:** `load_corpus()` re-invokes `merge.py` with `vocab = set(merge())`. Meanwhile, the README setup procedure runs `python -m data_pipeline.vocab.merge` beforehand.
- **Impact:** Brown corpus loading and Wikipedia download run once more at train execution time (merge.py is designed to reconstruct without reading `vocab.json`).
- **Action:** Either change `merge()` into a lightweight function that reads `vocab.json`, or have `load_corpus()` read `vocab.json` directly.

### 6.2 Incorrect Execution Order Documented in `data_pipeline/__init__.py`

- **Location:** `data_pipeline/__init__.py:6-10`
- **Symptom:** Step 3 of the docstring lists `export.static_vectors / contextual_vectors / vocab_pos` in parallel, but `static_vectors.py` depends on `train_w2v.model`, so it **cannot be executed until training is complete**.
- **Action:** Make the steps in the docstring explicit: "2. train → 3. export.static_vectors → 4. export.contextual_vectors → 5. export.vocab_pos".

### 6.3 Documentation Inconsistency of vocab.json Structure

- **Location:** `core/embedding_loader.py:213-220` and `data_pipeline/vocab/merge.py:50`
- **Symptom:**
  - Format written by merge.py: `{"vocab": ["word0", "word1", ...]}`
  - EmbeddingLoader supports both formats (list form and `{word: idx}` dictionary form)
  - However, no script writes out the dictionary form
- **Action:** Remove the fallback branch on the EmbeddingLoader side (`else: self.vocab = raw`) and explicitly state that only the list form is supported (YAGNI).

---

## 7. Test Coverage

### 7.1 0 Tests for `ui/`

- **Target:** `ui/app.py` (537 lines)
- **Gap:** Even pure functions like `results_to_df` are untested.
- **Judgment:** The Streamlit UI itself is hard to spu-test, but data-transformation helpers can be tested independently.

### 7.2 0 Tests for `data_pipeline/`

- **Targets:** `data_pipeline/_common/tokenizer.py`, `_common/token_definition.py`, `vocab/`, `train/`, `export/vocab_pos.map_pos_tag`, etc.
- **Gap examples:**
  - Behavior test of `normalize_tokens(["Hello", "WORLD", "123abc", "!"])`
  - vocab filter behavior of `tokenize_text`
  - Mapping table tests such as `map_pos_tag("NN") == "noun"`
- **Action:** Add unit tests for pure functions (only those with no external dependencies).

### 7.3 Risk of Updating the Label "192 Tests All Passing"

- **Location:** README.md, CLAUDE.md, detailed-design document.md
- **Symptom:** The number "192 tests" is hard-coded in multiple places. Three places must be updated when tests are added.
- **Action:** Loosen the README expression from a "number" to something like "core / transforms layers covered by unit tests", or auto-update the number in CI.

---

## 8. Other Minor Points

### 8.1 `.DS_Store` Exists throughout the Repository

- **Locations:** `core/`, `tests/`, `transforms/`, `data_pipeline/`, `data/`, etc.
- These remain with extended attributes. Confirm whether they are excluded by `.gitignore` (macOS-specific metadata file).

### 8.2 `core/__init__.py` is Empty

- The public API is not aggregated. `from core import EmbeddingLoader, SimilarityEngine, DistanceMetrics, ...` is not possible. If this is intentional, it is fine, but since the design document presents a list of classes in `core/`, making them explicit with `__all__` would reduce the import burden.

### 8.3 `unique_pos` UI Display

- **Location:** `ui/app.py:140`
- The value range of `vocab_pos` is only the 5 values `["adjective", "adverb", "any", "noun", "verb"]`. `"any"` is a coarse label that includes conjunctions, prepositions, determiners, pronouns, and so on; it has weak meaning as a filter. From the user's perspective, "which POS are included" is opaque.
- **Action:** Either add help text on the UI saying "any = POS other than the above 4 (conjunctions, prepositions, determiners, etc.)", or prepare a more granular coarse-grained POS (such as `"function"`).

### 8.4 No Search Button (Auto Search)

- **Location:** `ui/app.py`
- **Basic Design Document §2.1** states "Button: Execute Search", but the implementation **searches instantly on input**. There is a discrepancy between the design document and the implementation.
- **Action:** Align with either one. Instant search has good UX, but a 5-10 second computation runs on every keystroke, which may be counterproductive. Debouncing or a button is preferable.

### 8.5 Data Expansion

- **Target:** `data/metadata/vocab.json` (83823 words)
- **Status:** Not started
- **Contents:** Expansion of vocabulary count and corpus. An item for Phase 3 (improvement).
