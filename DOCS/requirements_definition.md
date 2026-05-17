# Requirements Definition — Explainable Semantic Space Explorer

**Project name:** Explainable Semantic Space Explorer  
**Version:** 1.0  
**Date:** 2026-05-05

---

## 1. Purpose and Background

- Word embeddings are widely used in NLP practice, yet there is no straightforward numerical means to explain *why* certain words are deemed close, which tends to render the models a black box.
- This system compares static embeddings (Word2Vec) and contextual embeddings (SBERT) side by side, and exposes the similarity computation as a decomposition into inner-product and norm terms, thereby making the behavior of the embedding space explainable.
- The objective is not to maximize NLP accuracy but to demonstrate **decision-making ability in designing explainable NLP systems**.

---

## 2. Problem Definition

- **Status quo:** Practitioners make judgments based on similarity scores produced by embedding models, but cannot justify those scores numerically.

- **Problem 1 (lack of explainability):** While it is well known that "queen" appears in the neighborhood of "king," practitioners cannot explain *why* they are close at the level of inner products and norms.

- **Problem 2 (difficulty of cross-model comparison):** Word2Vec (static) and SBERT (contextual) place the same word at different positions in their respective semantic spaces, but there are few tools to quantitatively compare this difference.

- **Problem 3 (difficulty of relative score evaluation):** Whether a similarity score of 0.82 is "high" or merely "average" cannot be judged without reference to the distribution across the entire vocabulary.

This system addresses Problem 1 through a decomposed display of cosine similarity together with Z-scores, Problem 2 through side-by-side comparison of static and contextual embeddings, and Problem 3 through distance-distribution statistics (histogram, mean, standard deviation).

---

## 3. System Overview

- Input: a query word (any English word)
- Output: Top-K similar words for each of Word2Vec and SBERT, cross-model difference comparison, breakdown of cosine similarity, distance-distribution statistics, and 2D projection / cluster scatter plots
- Backend: `core/` (similarity logic) and `transforms/` (PCA / UMAP / KMeans)
- Frontend: Streamlit (4-tab layout)
- Data: pre-generated local `.npy` files (vocabulary size 83,823)

| File | shape | dtype | Description |
|---|---|---|---|
| `static_vectors.npy` | (83823, 300) | float32 | Word2Vec static embeddings |
| `contextual_vectors.npy` | (83823, 384) | float32 | SBERT contextual embeddings |
| `vocab.json` | 83,823 entries | — | Vocabulary list |
| `vocab_pos.npy` | (83823,) | — | POS label array |
| `manifest.json` | — | — | Expected shape / dtype values |

No external APIs are used. The system runs entirely on a local CPU environment.

---

## 4. Functional Requirements

### 4.1 Similar-Word Search

- For a word entered by the user, the system shall output the Top-K similar words under the Word2Vec embedding (FR-01).
- For the same input word, the system shall output the Top-K similar words under the SBERT embedding (FR-02).
- For each similar word, the system shall present the similarity score, rank, within-POS rank, and heterogeneity rate alongside (FR-03).

### 4.2 Static vs Contextual Difference Comparison

- For the Top-K similar words from Word2Vec and SBERT for the same query word, the system shall present the number of shared words and the number of words unique to each side (FR-04).
- For the shared words, the system shall present the per-model rank difference and distance difference (FR-05).
- The system shall output a difference ranking sorted by the magnitude of the differences (FR-06).

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

- For each output, the system shall annotate the applicable items among the distance formula, the deviation from the vocabulary-wide mean, the presence of POS influence, cluster membership, and corpus frequency, so as to answer "why are they close?" with numerical evidence (FR-17).

---

## 5. Non-Functional Requirements

### Performance

| Process | Target |
|---|---|
| Top-K similar-word search + distance distribution computation | within 5 seconds (assuming vocabulary size of 100,000 or less) |
| PCA 2D projection | within 10 seconds |
| KMeans clustering (k=8, full vocabulary) | within 30 seconds |

### Reproducibility

- All processes that use randomness (clustering, UMAP, etc.) shall fix the seed and guarantee 100% reproducibility for identical inputs (NFR-01).
- A shape / dtype consistency check against `manifest.json` shall be performed at startup, and any inconsistency shall be explicitly detected (NFR-02).

### Runtime Environment

- The system shall be guaranteed to run only in a local CPU environment and shall not require a GPU (NFR-05).
- The runtime language shall be Python 3.12 (NFR-06).
- The system shall not require network connectivity at inference time (NFR-07).

### Explainability

- For every primary output, the distance computation formula, the related statistics (deviation from the vocabulary-wide mean, Z-score), and the presence of POS influence shall be annotated so that the system does not become a black box (NFR-08).

### Maintainability

- The core logic shall prioritize class-based design over a functional style, yielding a more structured implementation (NFR-09).
- Type hints shall be provided for all arguments and return values, and docstrings shall be written for all classes and methods (NFR-10).
- Exceptions shall be defined as purpose-specific custom exception classes; broadly raising generic exceptions is prohibited (NFR-11).

---

## 6. Constraints

| Item | Content |
|---|---|
| Language | Python 3.12 |
| Static embeddings | Word2Vec (pretrained model, locally deployed) |
| Contextual embeddings | SBERT (all-MiniLM-L6-v2, locally deployed) |
| External APIs | Prohibited (CONST-01) |
| HuggingFace downloads | Prohibited at inference time. Permitted only when running asset-generation scripts under `data_pipeline/` (CONST-02) |
| Cosine similarity | Custom implementation required. Use of scipy / sklearn `cosine_similarity` is prohibited (CONST-04) |
| Inference-time vectors | Limited to pre-saved ones (CONST-03) |
| Training | Prohibited at inference time (CONST-05) |
| Randomness | Seed must always be fixed when used (CONST-06) |
| Data consistency | shape / dtype check via `manifest.json` is required (CONST-07) |

---

## 7. Glossary

| Term | Definition |
|---|---|
| Word embedding | A representation that maps a word to a real-valued vector. Trained so that semantically similar words lie close to each other in vector space |
| Static embedding (Word2Vec) | An embedding method that assigns a single fixed vector per word, independent of context |
| Contextual embedding (SBERT) | Sentence-BERT. A method that generates context-aware embeddings using a Transformer-based model |
| Cosine similarity | The cosine of the angle between two vectors. Ranges from -1 to 1 and expresses the closeness of vector directions. `cos(a,b) = (a · b) / (‖a‖ · ‖b‖)` |
| Z-score | A measure of how far a value lies from the distribution mean in units of standard deviation. `z = (x - mean) / std` |
| Top-K similar words | The set of the top K words extracted in order of decreasing similarity to the query word |
| POS (part-of-speech) | A grammatical classification of a word such as noun, verb, or adjective |
| PCA | Principal Component Analysis. A linear dimensionality-reduction method that selects new axes along directions of maximum variance |
| UMAP | Uniform Manifold Approximation and Projection. A non-linear dimensionality-reduction method that preserves manifold structure |
| KMeans | A clustering method that partitions data into K clusters. In this system, L2 normalization is applied so that the partitioning is effectively based on cosine distance |
| manifest.json | A consistency-check file that records the expected shape / dtype of the asset files |
