# (High) Planned Improvements

This document consolidates the 4 high-priority issues. For details and the rationale behind the priority decisions, refer to `overview.md`.

---

## 1.1 Risk of Count Mismatch between `vocab.json` and `static_vectors.npy`

- **Location:** `data_pipeline/export/static_vectors.py:72-77`
- **Symptom:** `for word in vocab: if word in model.wv:` **skips words not in the Word2Vec vocabulary**.
  - As a result, `static_vectors.npy.shape[0]` may become smaller than `len(vocab)`.
  - Since `EmbeddingLoader._validate()` requires the N values to match, the application halts at startup with `IndexAlignmentError`.
- **Triggering condition:** When the `min_count=5` filter in `train_w2v.py` drops words from `vocab.json` from the Word2Vec training result.
- **Reason the issue is currently avoided:** By coincidence, `min_freq=10` (vocab generation side) ≥ `min_count=5` (Word2Vec side), so almost all words are likely present in `wv` in the current data. However, this is not guaranteed.
- **Recommended action:**
  - Write back `valid_words` as `vocab.json` (align vocab with the Word2Vec vocabulary)
  - Or insert a guard `assert set(vocab).issubset(model.wv.key_to_index.keys())` at the start of `static_vectors.py`

---

## 3.1 Dependence on the Current Working Directory at Startup

- **Location:** `ui/app.py:71`
- **Symptom:** `data_root = Path("data")` is a **relative path**. It fails when launched from Streamlit Cloud or from a different directory.
- **Action:** Make it an absolute path with `Path(__file__).resolve().parent.parent / "data"` (the same resolution method used by scripts under `data_pipeline/`).
- **Candidate deployment blocker:** The application becomes unbootable on configurations where the Streamlit Cloud working directory is not the project root.

---

## 3.4 Duplicate Calls to `get_distance_distribution` across Tabs

- **Location:** `ui/app.py:216, 303, 310, 341, 342`
- **Symptom:** For the same query word, `static_engine.get_distance_distribution(query_word)` is called a total of 5 times across **Tab1 / Tab2 (twice) / Tab3 (twice)**. `compare()` also calls `search` twice internally.
- **Impact:** One call equals a batch operation of 83823 × 300/384, so on a local CPU, **one query takes 5 to 10 seconds**. This may exceed the NFR "Top-K plus distance distribution computation within 5 seconds" defined in the requirements document.
- **Action:**
  - Cache with `@st.cache_data` keyed by `(query_word, engine_id)`
  - Or compute once each at the top of `main()` and pass to each tab.
- **The existing `@st.cache_resource` caches the engine instance itself, not the computation results.**

---

## 9.2 Streamlit Cloud Deployment Blocker due to `data/` Large-File Issue

- **Targets:** `data/embeddings/static_vectors.npy` (approx. 96 MB), `data/embeddings/contextual_vectors.npy` (approx. 123 MB)
- **Symptom:** Because the embedding files exceed Git's standard managed size, they are excluded by `.gitignore`. As a result, simply pushing to Streamlit Cloud does not deliver the data, and deployment cannot complete.
- **Impact:** Streamlit Cloud deployment is currently incomplete. The most critical blocker of Phase 2 (deployment).
- **Solutions under consideration:**
  - Manage large files with Git LFS
  - Upload assets to HuggingFace Hub and fetch them at Streamlit Cloud startup
  - Fetch from out-of-repo cloud storage (S3 etc.) at startup
- **Policy:** Deploy immediately once a solution is finalized.
