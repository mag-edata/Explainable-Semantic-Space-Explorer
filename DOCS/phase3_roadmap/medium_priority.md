# (Medium) Planned Improvements

This document consolidates the 5 medium-priority issues. For details and the rationale behind the priority decisions, refer to `overview.md`.

---

## 1.2 "Loss of Context" in POS Tagging

- **Location:** `data_pipeline/export/vocab_pos.py:124`
- **Symptom:** `pos_tag(vocab)` applies the NLTK POS tagger to a sorted word list (`['a', 'aaron', 'abandon', ...]`).
- **Issue:** NLTK's `averaged_perceptron_tagger` is **context-dependent** (it infers POS from surrounding words), so tagging on a list of words sorted in dictionary order produces erroneous judgments.
  - Example: even if "bank" is placed in the dictionary-order context of "balance, bank, banker", this is not a natural-language context.
  - To obtain the "most typical POS" on a per-word basis, one must either call `nltk.pos_tag([word])` per word, or devise an approach such as identifying occurrences within multiple sentences and voting.
- **Impact:** The accuracy of `loader.pos` degrades, weakening the meaning of the UI's POS filter and the heterogeneity-rate metric.
- **Caution is required regarding the reliability of POS information as an explainability tool, although this is not documented.**

---

## 1.4 Asymmetric Normalization State between Static and Contextual Vectors

- **Location:** `data_pipeline/export/contextual_vectors.py:82` vs. `data_pipeline/export/static_vectors.py`
- **Symptom:**
  - `contextual_vectors.npy` is saved **L2-normalized** with `normalize_embeddings=True`.
  - `static_vectors.npy` is **unnormalized** (raw Word2Vec vectors).
- **Impact:**
  - The cosine similarity values themselves are computed correctly because `cosine_similarity_batch` recomputes the norm of each row every time.
  - However, the **absolute values** of `dot_product` / `norm_a` / `norm_b` output in the `explanation` dictionary become **asymmetric between the two models**.
    - Contextual side: `norm_a ≈ norm_b ≈ 1.0`, `dot_product ≈ similarity`
    - Static side: `norm_a / norm_b` are real values, and `dot_product` is a large value
  - When comparing the "breakdown" on the UI, this is hard for users to read. There is an aspect that contradicts the claim of being an explainability tool.
- **Candidate policies:**
  - Unify both (normalize both on save, or leave both unnormalized)
  - Or explicitly state in the UI / docstring that "the contextual side is normalized in memory"

---

## 2.1 `seed` Unspecified + `workers=4` during Word2Vec Training

- **Location:** `data_pipeline/train/train_w2v.py:126-133`
- **Symptom:**
  ```python
  model = Word2Vec(
      sentences=corpus,
      vector_size=300,
      window=5,
      min_count=5,
      workers=4,
      sg=1,
  )
  ```
  - The `seed=` parameter is not specified (gensim's default is `seed=1`, but explicit specification is required by CONST-06 — "fix the seed when using random numbers" — defined in `requirements_definition.md`)
  - With `workers=4`, the **gensim official documentation states "full reproducibility is not guaranteed even when seed is specified"**. To guarantee full reproducibility, `workers=1` is required, along with fixing `PYTHONHASHSEED` as an environment variable.
- **Impact:** Regenerating `static_vectors.npy` may produce subtly different vectors. The phenomenon of "results change on regeneration" could occur after deployment.
- **Action:** Specify `seed=42, workers=1` explicitly (trade-off with training time).

---

## 2.2 UMAP `n_neighbors` Default vs. Sample Size

- **Location:** `transforms/projection.py:297`
- **Symptom:** In `UMAP(n_components=2, random_state=self._seed)`, `n_neighbors` is the UMAP default (15).
- **Issue:** In Tab 4 of `ui/app.py`, only "query + Top-K neighbors" are projected. When `top_k=10`, the sample size = 11 < 15, so UMAP either **emits a warning (`n_neighbors is larger than the dataset size`)** or auto-adjusts internally.
- **Impact:** Warnings appear in the UI logs, and the projection result becomes unstable.
- **Action:** Specify `n_neighbors=min(15, n_samples - 1)` dynamically inside `_fit_umap`.

---

## 9.1 [UI] Projection/Cluster Tab: Mismatch between the Legend of the Query-Word Marker and the Figure

- **File:** `ui/app.py` (the scatter plot in the projection/cluster tab, Tab 4)
- **Symptom:** Although the query word is specified as `"cross"` using `alt.Shape` + `alt.Scale(domain, range)`, the legend display in Altair does not match the marker shape on the figure.
- **Failed attempts:**
  - `range=["star", "circle"]` → becomes an invalid value in Vega-Lite and the legend itself disappears
  - Splitting the chart into layers (neighbor words and query word in separate `alt.Chart`) → the rendering itself disappeared
- **Current state:** Held with `range=["cross", "circle"]`
- **Next candidate solutions:**
  - Drop the `shape` encoding and emphasize the query with `size` + `color`
  - Pass Vega-Lite valid values (`"diamond"` / `"triangle-up"` etc.) to `range`
  - Pass Vega-Lite SVG path strings directly to `range`
  - Combine `mark_rule` / `mark_point` to overlay the query point separately
