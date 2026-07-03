# Explainable Semantic Space Explorer

> Explainability-focused NLP analysis tool for analyzing why embeddings are semantically close.

A tool that visualizes and analyzes word embedding spaces, **explaining numerically "why words are close to each other."**

- Quantitatively compares the differences between static embeddings (Word2Vec) and contextual embeddings (SBERT)
- Decomposes cosine similarity into dot product, norm, and formula form — no black boxes
- No external APIs; runs fully on local CPU

---

## Demo

> Screenshots / GIFs will be added.

Example comparing static and contextual embeddings for the query word `"bank"`:

**Word2Vec (static) neighbors**

```
river, shore, creek, lake, ...
```

→ Without context, the geographical sense dominates.

**SBERT (contextual) neighbors**

```
financial, credit, loan, fund, ...
```

→ The financial sense becomes dominant based on corpus context.

The same word `"bank"` occupies **different positions in semantic space depending on the embedding model**, and this can be confirmed numerically.

---

## Key Features

- **Similarity breakdown**: Decomposes cosine similarity into `dot(a,b) / (‖a‖·‖b‖)` form, showing the dot product and norms individually.
- **Relative evaluation via Z-score**: Quantifies how much of an outlier a similarity score is within the overall vocabulary distribution, answering "is 0.82 actually high?"
- **Side-by-side static vs. contextual comparison**: Symmetrically compares neighbors, rank differences, shared words, and unique words between Word2Vec and SBERT.
- **2D projection + clustering**: Overviews the embedding space with PCA / UMAP and identifies semantic groups via KMeans.
- **Part-of-speech filtering**: Filters neighbors by POS (noun, verb, adjective, etc.) to isolate syntactic bias.

---

## Relevance to LLM / RAG Systems

Embeddings are widely used as the foundation of RAG, semantic search, and recommendation, but their behavior is easily black-boxed. This tool addresses problems directly tied to these use cases:

- **RAG retrieval quality evaluation**: If you can't explain why a document was retrieved, you can't diagnose retrieval failures.
- **Semantic search debugging**: Viewing the breakdown of cosine similarity makes it possible to identify the cause of mis-hits.
- **Justifying model selection**: Quantitatively comparing static vs. contextual embeddings enables explainable per-use-case model choices.
- **Embedding drift analysis**: Observes how the distance distribution shifts between static and contextual representations of the same vocabulary.

---

## Architecture

Dependencies flow in one direction only. Reverse dependencies are forbidden.

```
data/ ──► core/ ──► transforms/ ──► ui/app.py
```

| Layer | Role |
|---|---|
| `data/` | Embedding data (read-only) |
| `core/` | Pure logic: distance computation, search, statistics |
| `transforms/` | Vector transformations such as PCA, UMAP, KMeans |
| `ui/` | Streamlit display only. No business logic allowed. |

---

## Design Rationale

**Why implement cosine similarity from scratch?**
Using `cosine_similarity` from scipy / sklearn would be a one-liner. Implementing it by hand makes the computation steps (dot product, norms, division) explicit in the UI, so users can see *what cosine similarity actually is*.

**Why handle both static and contextual embeddings?**
Word2Vec assigns a fixed vector per word (it cannot distinguish polysemy). SBERT generates vectors dynamically based on context. Showing both side by side lets users understand *what differs* between them, rather than asking *which is better*.

**Why no external APIs?**
Reproducibility and transparency are the top priorities, so external service dependencies are eliminated. The tool runs entirely on local CPU, and every behavior can be traced by reading the code.

**Why place NLTK data inside the repository (`data/nltk_data/`)?**
NLTK's default location (`~/nltk_data`) sits in the home directory, making it hard to track which project owns the data and polluting the environment. The Brown corpus and tagger are project-specific resources, so placing them inside the repo makes dependencies self-contained.

**Why leave the HuggingFace cache at its default (`~/.cache/huggingface/`)?**
The HuggingFace cache is standard ML infrastructure shared across projects, designed so that multiple projects don't re-download identical datasets. Confining it inside the repo would lose this sharing benefit, so we follow the standard.

---

## Tech Stack

| Role | Technology |
|---|---|
| Static embeddings | Word2Vec (pre-trained model, locally hosted) |
| Contextual embeddings | SBERT (all-MiniLM-L6-v2, locally hosted) |
| Dimensionality reduction | PCA / UMAP |
| Clustering | KMeans (cosine distance) |
| UI | Streamlit + Altair |
| Language | Python 3.12 |
| Testing | unittest (192 tests, all passing) |

---

## What This Project Demonstrates

- How to bake "explainability" into the design of an embedding space exploration tool
- Building a fully local NLP pipeline that does not depend on external APIs
- An architecture that quantitatively visualizes and compares static vs. contextual embeddings
- Quality assurance of core logic via 192 tests (covering both `core/` and `transforms/`)

---

## Quick Start

```bash
git clone https://github.com/mag-edata/Explainable-Semantic-Space-Explorer.git
cd Explainable-Semantic-Space-Explorer
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run ui/app.py
```

> **Note:** Before the first run, you need to generate the embedding files under `data/`. See [Full Data Pipeline Setup](#full-data-pipeline-setup) below.

---

## Full Data Pipeline Setup

<details>
<summary>First-time setup instructions (click to expand)</summary>

```bash
# Download NLTK data (saved to data/nltk_data/)
# brown / tagger: corpus and POS tagging. wordnet / stopwords: vocabulary curation.
python -c "import nltk; [nltk.download(p, download_dir='data/nltk_data') for p in ['brown', 'averaged_perceptron_tagger_eng', 'wordnet', 'stopwords']]"

# Generate asset files (requires HuggingFace access).
# Run in this exact order: static_vectors finalizes vocab.json, so it must
# precede contextual_vectors and vocab_pos to keep all assets index-aligned.
# PYTHONHASHSEED=0 makes the Word2Vec training fully reproducible (CONST-06).
export PYTHONHASHSEED=0
python -m data_pipeline.vocab.merge                   # → data/metadata/vocab.json (curated candidate vocab)
python -m data_pipeline.train.train_w2v               # → models/w2v_brown10_simplewiki10_sg_300d_w5.model
python -m data_pipeline.export.static_vectors         # → data/embeddings/static_vectors.npy (+ finalizes vocab.json)
python -m data_pipeline.export.contextual_vectors     # → data/embeddings/contextual_vectors.npy
python -m data_pipeline.export.vocab_pos              # → data/metadata/vocab_pos.npy
python -m data_pipeline.manifest                      # → data/manifest.json
```

> **Vocabulary curation (v2.0):** `merge` now filters the corpus union down to
> real English words via WordNet + stopword membership (see
> `data_pipeline/vocab/curate.py`), removing non-words and proper-noun
> fragments. Re-running the pipeline above regenerates all assets against the
> curated, index-aligned vocabulary.

### Verified Environment

**Runtime**

| Item | Version / Details |
|---|---|
| OS | macOS Tahoe 26.3.1 |
| Python | 3.12 |
| Virtual environment | venv (standard library) |

**Core dependencies**

| Package | Version | Purpose |
|---|---|---|
| numpy | 2.4.2 | Vector operations |
| scipy | 1.17.1 | Numerical computation (auxiliary) |
| scikit-learn | 1.8.0 | KMeans / PCA |
| umap-learn | 0.5.11 | Non-linear dimensionality reduction |
| numba | 0.64.0 | Dependency of umap-learn |
| pandas | 2.3.3 | Data wrangling |

**NLP / embeddings**

| Package | Version | Purpose |
|---|---|---|
| gensim | ≥4.3 | Word2Vec training |
| sentence-transformers | (latest) | SBERT inference |
| nltk | ≥3.8 | Brown corpus, POS tagger |
| datasets | ≥2.16, <3.0 | Fetching Simple Wikipedia |

**UI / visualization**

| Package | Version |
|---|---|
| streamlit | 1.54.0 |
| altair | 6.0.0 |

**External assets**

| Item | Details |
|---|---|
| Word2Vec model | Trained in-house (Brown + Simple Wikipedia, sg=1, dim=300, win=5, min_count=5) |
| SBERT model | `all-MiniLM-L6-v2` (HuggingFace) |
| NLTK data | `brown`, `averaged_perceptron_tagger_eng` (placed under `data/nltk_data/`) |

**Tests**

| Item | Details |
|---|---|
| Framework | `unittest` (standard library) |
| Test count | 192 |
| Run command | `python -m unittest discover tests` |

</details>

---

## Project Structure

```
Explainable-Semantic-Space-Explorer/
├── data/                        # Embedding vectors and metadata (not tracked by Git)
│   ├── embeddings/
│   │   ├── static_vectors.npy     # Word2Vec  shape (83823, 300)
│   │   └── contextual_vectors.npy # SBERT     shape (83823, 384)
│   ├── metadata/
│   │   ├── vocab.json             # Vocabulary list
│   │   └── vocab_pos.npy          # POS label array, shape (83823,)
│   ├── nltk_data/                 # NLTK corpora (not tracked by Git)
│   └── manifest.json              # Used for shape / dtype consistency checks
│
├── core/                          # Pure logic layer
│   ├── embedding_loader.py        # Vector loading and consistency checks
│   ├── similarity_engine.py       # Similarity search and comparison
│   ├── distance_metrics.py        # Custom cosine similarity implementation
│   ├── pos_filter.py              # POS-based filtering
│   └── analyzer.py                # Statistical analysis of distance distributions
│
├── transforms/                    # Vector transformation layer
│   ├── clustering.py              # KMeans clustering
│   └── projection.py              # 2D projection via PCA / UMAP
│
├── ui/
│   └── app.py                     # Streamlit UI (4 tabs)
│
├── tests/                         # Unit test suite (192 tests, all passing)
│
├── data_pipeline/                 # Asset generation pipeline (run only at setup)
│   ├── _common/
│   │   ├── nltk_setup.py          # NLTK data path management and auto-download
│   │   ├── token_definition.py    # Token normalization rules
│   │   └── tokenizer.py           # Shared tokenizer
│   ├── vocab/
│   │   ├── gen_brown.py           # Generates vocabulary from the Brown corpus
│   │   ├── gen_wiki.py            # Generates vocabulary from Simple Wikipedia
│   │   └── merge.py               # Merges and sorts vocab → vocab.json
│   ├── export/
│   │   ├── static_vectors.py      # Exports Word2Vec vectors to .npy
│   │   ├── contextual_vectors.py  # Exports SBERT vectors to .npy
│   │   └── vocab_pos.py           # Exports POS label array to .npy
│   ├── train/
│   │   └── train_w2v.py           # Trains the Word2Vec model
│   └── manifest.py                # Generates manifest.json and runs consistency checks
│
├── models/                        # Word2Vec model location (not tracked by Git)
│
├── DOCS/                          # Design documentation
│
├── requirements.txt
└── README.md
```

---

## Testing

- 192 unit tests passing
- Core logic and transform layers fully covered
- Deterministic local execution (fixed random seed)

---

## Background

My interest in NLP with Python began with research that treats word semantic relationships statistically.

In recent LLM systems, embeddings serve as the foundation for RAG, search, and recommendation, yet the behavior of embedding spaces tends to be black-boxed. It is difficult to explain **why two words are close to each other, or what differs from model to model** — and this makes model selection, retrieval quality evaluation, and failure analysis hard.

This project treats embeddings not as something "to be used" but as something "to be analyzed and explained," designed as an NLP system with explainability built in.
