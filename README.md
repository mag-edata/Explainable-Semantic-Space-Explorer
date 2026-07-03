# Explainable Semantic Space Explorer

> An interactive lab for NLP learners to check textbook claims about word embeddings against real data.

A tool that visualizes and analyzes word embedding spaces, showing numerically **how** word similarity is computed and **how to read** it — without claiming to explain *why* two words are semantically related (that lives in corpus co-occurrence, not in the arithmetic).

- Compares static embeddings (Word2Vec) and single-word SBERT embeddings side by side
- Decomposes cosine similarity into dot product, norm, and formula form — no black boxes
- No external APIs; runs fully on local CPU

---

## Demo

> Screenshots / GIFs will be added.

**Q1 — are similar words really close?** Query `"king"` under Word2Vec returns:

```
queen (0.67), ethelred (0.65), athelstan (0.62), canute (0.61), ...
```

The textbook claim that "queen" sits near "king" checks out — and the model has also gathered a cluster of historical kings nearby.

**Q3 — is 0.67 actually high?** Against the whole-vocabulary similarity distribution, 0.67 lands far out in the right tail (a large Z-score), so "queen" is *genuinely* close rather than merely above average. The distribution histogram shows this at a glance.

**Static vs. contextual.** The same word can sit in different positions under Word2Vec and single-word SBERT, and the tool compares them side by side. In the current **word mode** both models use one fixed vector per word; a **sentence mode** that shows a word's vector shifting with its surrounding context (e.g. `"bank"` by a river vs. a bank loan) is planned — see the roadmap.

---

## Key Features

- **Similarity breakdown**: Decomposes cosine similarity into `dot(a,b) / (‖a‖·‖b‖)` form, showing the dot product and norms individually.
- **Relative evaluation via Z-score**: Quantifies how much of an outlier a similarity score is within the overall vocabulary distribution, answering "is 0.82 actually high?"
- **Side-by-side static vs. contextual comparison**: Symmetrically compares neighbors, rank differences, shared words, and unique words between Word2Vec and SBERT.
- **2D projection + clustering**: Overviews the embedding space with PCA / UMAP and identifies semantic groups via KMeans.
- **Part-of-speech filtering**: Filters neighbors by POS (noun, verb, adjective, etc.) to isolate syntactic bias.

---

## Who this is for & how to use it

This tool is an interactive lab for **NLP learners** — students, junior engineers, and anyone back-filling embedding fundamentals — to check textbook claims about word embeddings against real data. It is **not** a production model-selection benchmark.

Four questions it helps you answer:

- **Q1. Are similar words really close?** Type a word and read its nearest neighbors.
- **Q2. Does context change a word's meaning?** *In word mode both models use one fixed vector per word, so this is not yet demonstrated; a sentence-input mode that shows meaning shifting with context is planned (see the roadmap).*
- **Q3. Is a similarity of 0.82 actually high?** Read the Z-score and the distribution histogram, not the raw score alone.
- **Q4. Where does the number come from?** Open the cosine breakdown to see the dot product and norms.

### Feature guide — what to try, and how to read it

- **Similar-word search (Tab 1)** — *Try it when* you want to see a word's neighborhood. *Read it as:* the similarity column answers Q1; the Z-score column answers Q3 (a higher value = more of an outlier = genuinely close); open **"formula breakdown"** for Q4.
- **Static vs. contextual (Tab 2)** — *Try it when* you want to see how two different models place the same word. *Read it as:* the shared / unique neighbors and the Jaccard stability score show how much the two models agree. *Note:* in word mode the SBERT vectors are single-word **anchor** embeddings, so this compares two models' word vectors — the "meaning shifts with context" demonstration is planned for sentence mode.
- **Distance distribution (Tab 3)** — *Try it when* you want to judge whether a similarity is high. *Read it as:* the histogram is the query's similarity against the whole vocabulary, and the red line marks the top-1. If it sits far out in the right tail, the match is genuinely close — this is the visual form of Q3.
- **Projection / clusters (Tab 4)** — *Try it when* you want a spatial picture of the neighborhood. *Read it as:* nearby points are similar and clusters are sub-themes. Watch the PCA contribution rate — a low percentage means the 2-D picture is a lossy shadow of the 300-D space.

### How to read the numbers

- **Cosine similarity** — direction agreement, ranging −1 to 1. Because both models' vectors are stored as unit-length vectors, the similarity equals the dot product.
- **Z-score** — how many standard deviations the similarity sits above the vocabulary-wide average. A large positive Z means "unusually close," and *that* is the real signal — the raw 0.82 on its own is not.
- **Part-of-speech / heterogeneity** — how many neighbors carry a different part of speech than your query, giving a rough sense of whether the neighborhood is driven by meaning or by grammar.

---

## Why it matters when you work with LLMs

Embeddings sit under semantic search, RAG, and recommendation, yet their behavior is easy to treat as a black box. Building intuition for what a similarity score actually means, how the embedding space is shaped, and how two models can disagree — by inspecting real vectors instead of trusting a single number — is the groundwork that makes those systems easier to reason about later. This tool is a place to build that intuition, not a benchmark for choosing a production model.

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
Word2Vec assigns a fixed vector per word (it cannot distinguish polysemy). SBERT *can* generate context-dependent vectors; in the current word mode it is used to produce one anchor vector per word, with a sentence mode that exercises its context-sensitivity planned. Showing both side by side lets users understand *what differs* between them, rather than asking *which is better*.

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
| Testing | unittest (208 tests, all passing) |

---

## What This Project Demonstrates

- How to bake "explainability" into the design of an embedding space exploration tool
- Building a fully local NLP pipeline that does not depend on external APIs
- An architecture that quantitatively visualizes and compares static vs. contextual embeddings
- Quality assurance of core logic via 208 tests (covering both `core/` and `transforms/`)

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
| Test count | 208 |
| Run command | `python -m unittest discover tests` |

</details>

---

## Project Structure

```
Explainable-Semantic-Space-Explorer/
├── data/                        # Embedding vectors and metadata (not tracked by Git)
│   ├── embeddings/
│   │   ├── static_vectors.npy     # Word2Vec  shape (40032, 300)
│   │   └── contextual_vectors.npy # SBERT     shape (40032, 384)
│   ├── metadata/
│   │   ├── vocab.json             # Vocabulary list
│   │   └── vocab_pos.npy          # POS label array, shape (40032,)
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
├── tests/                         # Unit test suite (208 tests, all passing)
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

- 208 unit tests passing
- Core logic and transform layers fully covered
- Deterministic local execution (fixed random seed)

---

## Background

My interest in NLP with Python began with research that treats word semantic relationships statistically.

In recent LLM systems, embeddings serve as the foundation for RAG, search, and recommendation, yet the behavior of embedding spaces tends to be black-boxed. Learners rarely get to see *how* a similarity score is computed, how the space is shaped, or how two models differ — so the intuition stays abstract.

This project treats embeddings not as something "to be used" but as something "to be analyzed and explained," designed as an NLP system with explainability built in.
