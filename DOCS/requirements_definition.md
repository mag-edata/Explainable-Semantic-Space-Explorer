# Requirements Definition — Explainable Semantic Space Explorer

**Project name:** Explainable Semantic Space Explorer
**Version:** 2.0
**Date:** 2026-07-03

**Revision note (v2.0):** v2.0 defines the system as a real product for NLP learners. Major changes: target-user redefinition (§2), honest repositioning of the cosine decomposition (computational transparency, not a semantic explanation), vocabulary quality promoted to a functional requirement (FR-18/19), sentence-context mode added (FR-20–22), guided UX added (FR-23–27), CONST-03 amended, phasing added (§7). v1.0 remains available in git history.

---

## 1. Purpose and Background

- Textbooks and blog posts state claims about word embeddings — "semantically similar words are placed close together," "static embeddings assign one vector per word and cannot distinguish word senses, while contextual embeddings can," "a similarity score has no absolute meaning" — but learners rarely get a chance to verify these claims against real data.
- This system is an **interactive laboratory for verifying textbook claims about word embeddings**: browse real neighborhoods, compare a static space (Word2Vec) and an SBERT space side by side, watch a word's representation move when its sentence changes, and learn to read similarity scores relative to the whole-vocabulary distribution.
- The system explains **how** every displayed number is computed (cosine decomposition, distribution statistics) and **how to interpret it** (Z-score, histogram, plain-language verdicts). It deliberately does **not** claim to explain *why* two words are semantically related — the semantic "why" lives in corpus co-occurrence, not in inner-product arithmetic. Keeping that boundary honest is part of the design.

---

## 2. Target Users and Problem Definition

### 2.1 Target users

- **Primary:** NLP learners — students, junior engineers, and engineers who entered the field through LLM applications and are back-filling fundamentals. They have read about embeddings but have never interacted with an embedding space directly.
- **Secondary:** instructors and mentors who want a live, hands-on demo for teaching embeddings.
- **Explicit non-target:** practitioners selecting an embedding model for production. Model selection should be done with downstream-task benchmarks, which this tool does not provide and does not claim to replace.

### 2.2 Problems — the learner questions this system answers

- **Q1 "Are similar words really close?"** Learners cannot easily browse a *trustworthy* neighborhood. A neighborhood polluted with non-words ("aabach", "aaahh", …) destroys the learning experience, so vocabulary quality is a requirement in itself, not an implementation detail (→ FR-18/19).
- **Q2 "Does context really change meaning?"** The claim "contextual models disambiguate senses in context" can only be believed when seen. The user gives one word in two different sentences and observes the vectors and neighborhoods diverge (→ FR-20–22, sentence-context mode).
- **Q3 "Is 0.82 high?"** Similarity scores are routinely misread as absolute values. Every score shall be relativized against the whole-vocabulary distribution (Z-score, histogram) and the verdict stated in plain language (→ FR-08/09/25). This relativization is the system's core original asset.
- **Q4 "Where does the number come from?"** The cosine computation is exposed as inner product and norms — as teaching material for *how* similarity is computed. This is computational transparency, not a semantic explanation, and the UI wording shall reflect that distinction (→ FR-17, NFR-08).

---

## 3. System Overview

The system offers two modes:

| Mode | Input | Vectors used |
|---|---|---|
| Word mode | a query word | pre-generated local `.npy` assets |
| Sentence-context mode (Phase B) | a sentence containing the query word | in-context token vector computed at inference time by the locally deployed transformer model |

- Output: Top-K similar words (static / SBERT), side-by-side model comparison (the primary view), distance-distribution statistics with plain-language interpretation, cosine-similarity breakdown, and 2D projection / cluster scatter plots.
- Backend: `core/` (similarity logic) and `transforms/` (PCA / UMAP / KMeans). Frontend: Streamlit.
- Data: pre-generated local files. V denotes the vocabulary size recorded in `manifest.json` (v1 assets: 83,823; after curation the target is roughly 20,000–30,000 high-quality words).

| File | shape | dtype | Description |
|---|---|---|---|
| `static_vectors.npy` | (V, 300) | float32 | Word2Vec static embeddings (self-trained) |
| `contextual_vectors.npy` | (V, 384) | float32 | SBERT word-anchor embeddings (single-word encodes; labeled accurately in the UI, see FR-02) |
| `vocab.json` | V entries | — | Curated vocabulary list |
| `vocab_pos.npy` | (V,) | — | POS label array |
| `manifest.json` | — | — | Expected shape / dtype values |

No external APIs are used. The system runs entirely on a local CPU environment.

---

## 4. Functional Requirements

FR-01–17 carry over from v1.0 (amendments noted); FR-18 and later are new in v2.0.

### 4.1 Similar-Word Search

- For a word entered by the user, the system shall output the Top-K similar words under the Word2Vec embedding (FR-01).
- For the same input word, the system shall output the Top-K similar words under the SBERT embedding (FR-02, **amended**): word-mode SBERT vectors are single-word encodes and shall **not** be presented as context-resolved; the UI shall label them accurately (e.g., "SBERT (word anchor)").
- For each similar word, the system shall present the similarity score, rank, within-POS rank, and heterogeneity rate alongside (FR-03).

### 4.2 Static vs SBERT Difference Comparison

- For the Top-K similar words from Word2Vec and SBERT for the same query word, the system shall present the number of shared words and the number of words unique to each side (FR-04).
- For the shared words, the system shall present the per-model rank difference and distance difference (FR-05).
- The system shall output a difference ranking sorted by the magnitude of the differences (FR-06).
- The comparison view is the system's primary view and shall be the first screen the user encounters (FR-27, **new**).

### 4.3 Distance Distribution Analysis

- The system shall visualize the distribution of cosine similarities between the query word and the entire vocabulary in histogram form (FR-07).
- The system shall output the vocabulary-wide mean, standard deviation, and Top-1 similarity as numerical values (FR-08).
- The system shall compute the anomaly degree of the Top-1 similarity as a Z-score (FR-09).

### 4.4 2D Projection

- The system shall project embedding vectors into two dimensions, with the user able to choose between PCA and UMAP (FR-10).
- When PCA is used, the system shall also display the contribution rate of each principal component (FR-11).
- The system shall annotate each point on the projection with its cluster ID (FR-12).

### 4.5 Clustering

- The system shall cluster the embedding vectors using KMeans and assign a cluster ID to each word (FR-13).
- The user shall be able to specify the number of clusters and the random seed; identical specifications shall yield identical results (FR-14).

### 4.6 POS Filtering

- The system shall allow the similar-word search results to be filtered by POS (FR-15).
- The system shall compute the heterogeneity rate of the search results (the proportion of words whose POS differs from that of the query word) (FR-16).

### 4.7 Provision of Explanatory Information

- For each output, the system shall annotate the applicable items among the distance formula, the deviation from the vocabulary-wide mean, the presence of POS influence, and cluster membership (FR-17, **amended**): the wording shall present these as *how the number was computed and how it compares*, never as a claim of semantic causation.

### 4.8 Vocabulary Quality (new)

- The vocabulary shall contain only words that pass explicit curation rules — lowercase alphabetic form, corpus frequency at or above a documented threshold, and membership in an English lexical resource (e.g., WordNet) — so that non-words, abbreviation noise, and proper-noun fragments are excluded (FR-18).
- Curation shall be implemented as reproducible scripts under `data_pipeline/` with the rules documented, and POS labels shall be produced by a method that preserves tagging validity (not by tagging a sorted word list) (FR-19).

### 4.9 Sentence-Context Mode (new, Phase B)

- The user shall be able to input a sentence containing the query word; the system shall compute the query's in-context token vector at inference time using the locally deployed model, with no network access and no training (FR-20).
- The system shall display the nearest words of the in-context vector and contrast them with the static neighborhood of the same word (FR-21).
- The user shall be able to input two sentences for the same word and see (a) the similarity between the two in-context vectors and (b) both neighborhoods, so that sense separation (e.g., "bank": financial vs. river) is directly observable (FR-22).

### 4.10 Guided UX (new)

- One-click example queries curated to showcase interesting phenomena (e.g., bank, apple, mouse, run) shall be available on the first screen (FR-23).
- When the input word is out of vocabulary, the system shall suggest close alternatives (e.g., spelling-distance candidates) rather than terminating with an error alone (FR-24).
- Every primary metric shall be accompanied by a plain-language verdict (e.g., "0.82 — top 0.1% of 24,000 words: unusually close") (FR-25).
- Heavy computation shall run on explicit user action (search button / form), and intermediate results shall be cached so that identical queries are not recomputed across tabs (FR-26).

---

## 5. Non-Functional Requirements

### Performance

| Process | Target |
|---|---|
| Top-K similar-word search + distance distribution computation | within 5 seconds (assuming vocabulary size of 100,000 or less) |
| Repeated identical query (cache hit) | within 1 second |
| Sentence-context encoding (Phase B) | within 5 seconds on CPU |
| PCA 2D projection | within 10 seconds |
| KMeans clustering (k=8, full vocabulary) | within 30 seconds |

### Reproducibility

- All processes that use randomness shall fix the seed and guarantee 100% reproducibility for identical inputs. This **includes the training pipeline**: Word2Vec shall be trained with a fixed seed and a single worker (e.g., `seed=42, workers=1`) (NFR-01, **amended**).
- A shape / dtype consistency check against `manifest.json` shall be performed at startup, and any inconsistency shall be explicitly detected (NFR-02).

### Runtime Environment

- The system shall be guaranteed to run only in a local CPU environment and shall not require a GPU (NFR-05).
- The runtime language shall be Python 3.12 (NFR-06).
- The system shall not require network connectivity at inference time (NFR-07).

### Interpretability (renamed from "Explainability")

- Every primary output shall be annotated with its computation formula, its relative statistics (deviation from the vocabulary-wide mean, Z-score), and a plain-language reading. The wording shall claim computational transparency and distributional interpretation only — never semantic causation (NFR-08, **amended**).

### Usability (new)

- The first screen shall be understandable without reading the README: a one-line purpose statement and example-query buttons shall be present (NFR-12).
- All displayed metrics shall carry help text (NFR-13).

### Maintainability

- The core logic shall prioritize class-based design over a functional style, yielding a more structured implementation (NFR-09).
- Type hints shall be provided for all arguments and return values, and docstrings shall be written for all classes and methods (NFR-10).
- Exceptions shall be defined as purpose-specific custom exception classes; broadly raising generic exceptions is prohibited (NFR-11).

---

## 6. Constraints

| Item | Content |
|---|---|
| Language | Python 3.12 |
| Static embeddings | Word2Vec (self-trained by `data_pipeline/`, locally deployed) |
| Contextual embeddings | SBERT (all-MiniLM-L6-v2, locally deployed) |
| External APIs | Prohibited (CONST-01) |
| HuggingFace downloads | Prohibited at inference time. Permitted only when running asset-generation scripts under `data_pipeline/` and for pre-deploying the model (CONST-02) |
| Cosine similarity | Custom implementation required. Use of scipy / sklearn `cosine_similarity` is prohibited (CONST-04) |
| Inference-time vectors | Word mode: pre-saved vectors only. Sentence-context mode: encoding the user's input sentence with the locally deployed model is permitted; network access and training remain prohibited (CONST-03, **amended in v2.0**) |
| Training | Prohibited at inference time (CONST-05) |
| Randomness | Seed must always be fixed when randomness is used, including in the training pipeline (CONST-06) |
| Data consistency | shape / dtype check via `manifest.json` is required (CONST-07) |

---

## 7. Phasing

| Phase | Contents | Requirements covered |
|---|---|---|
| **A — Data quality & UX repair** | Vocabulary curation and full asset regeneration (Word2Vec retraining with fixed seed), context-valid POS regeneration, normalization unification, cross-tab computation caching + form-based execution, example buttons / OOV suggestions / verdict lines. *Conditional:* corpus expansion (e.g., WikiText-103) only if neighborhood quality is still insufficient after curation | FR-18/19, FR-23–26, NFR-01 |
| **B — Sentence-context mode** | In-context token vector extraction, two-sentence comparison, comparison-first screen restructure | FR-20–22, FR-27 |
| **C — Release** | Streamlit Cloud deployment (assets shrink after Phase A), user-facing README rewrite | — |

---

## 8. Glossary

| Term | Definition |
|---|---|
| Word embedding | A representation that maps a word to a real-valued vector. Trained so that semantically similar words lie close to each other in vector space |
| Static embedding (Word2Vec) | An embedding method that assigns a single fixed vector per word, independent of context |
| Word-anchor vector (SBERT) | The vector obtained by encoding a word in isolation with SBERT. Fixed per word; **not** context-resolved |
| In-context (token) vector | The vector a transformer assigns to a specific occurrence of a word inside a sentence; it changes when the sentence changes |
| Sentence-context mode | The mode in which the user supplies a sentence and the system computes the in-context vector of the query word at inference time |
| Vocabulary curation | Rule-based filtering of the vocabulary (frequency threshold, lexical-resource membership, form constraints) to exclude non-words and noise |
| Verdict (plain-language) | A one-line natural-language interpretation attached to a metric, e.g., "top 0.1% of the vocabulary — unusually close" |
| Cosine similarity | The cosine of the angle between two vectors. Ranges from -1 to 1 and expresses the closeness of vector directions. `cos(a,b) = (a · b) / (‖a‖ · ‖b‖)` |
| Z-score | A measure of how far a value lies from the distribution mean in units of standard deviation. `z = (x - mean) / std` |
| Top-K similar words | The set of the top K words extracted in order of decreasing similarity to the query word |
| POS (part-of-speech) | A grammatical classification of a word such as noun, verb, or adjective |
| PCA | Principal Component Analysis. A linear dimensionality-reduction method that selects new axes along directions of maximum variance |
| UMAP | Uniform Manifold Approximation and Projection. A non-linear dimensionality-reduction method that preserves manifold structure |
| KMeans | A clustering method that partitions data into K clusters. In this system, L2 normalization is applied so that the partitioning is effectively based on cosine distance |
| manifest.json | A consistency-check file that records the expected shape / dtype of the asset files |
