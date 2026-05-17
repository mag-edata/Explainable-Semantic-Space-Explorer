"""
static_vectors.py

Inputs:
    ``data/metadata/vocab.json``
    ``models/w2v_brown10_simplewiki10_sg_300d_w5.model``

Outputs:
    ``data/embeddings/static_vectors.npy``  shape=(N, 300)  dtype=float32

This module provides the preprocessing function that, against the merged
vocabulary finalized by ``vocab/merge.py``, builds the static word
vector matrix (``STATIC_VECTORS``) using a trained Word2Vec model and
saves it in NumPy format.

The goal is to pre-export static word embeddings for the merged
vocabulary so that inference no longer depends on gensim, enabling fast
and consistent vocabulary / vector lookups.

The vocabulary actually saved is restricted to words for which a vector
can be retrieved from the Word2Vec model, guaranteeing index alignment
between the vocabulary array and the vector matrix.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
from gensim.models import Word2Vec

# ---------- I/O paths (specific to this script) ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
STATIC_VECTORS: Path = PROJECT_ROOT / "data" / "embeddings" / "static_vectors.npy"
W2V_MODEL: Path = PROJECT_ROOT / "models" / "w2v_brown10_simplewiki10_sg_300d_w5.model"


def load_vocab() -> List[str]:
    """
    Returns
    -------
    list[str]
        Sorted merged vocabulary list.
    """
    with VOCAB_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "vocab" not in data:
        raise ValueError(
            f"{VOCAB_JSON} is missing the required 'vocab' key"
        )

    return data["vocab"]


def export_static_vectors() -> None:
    """
    Output:
    - STATIC_VECTORS: shape = (N, dim), dtype = float32

    Indices match ``vocab.json`` exactly.
    """
    vocab = load_vocab()
    model = Word2Vec.load(str(W2V_MODEL))
    valid_words: List[str] = []
    vectors: List[np.ndarray] = []

    # Collect word vectors that correspond to the vocabulary
    for word in vocab:
        # gensim Word2Vec keeps the vocabulary on model.wv
        if word in model.wv:
            valid_words.append(word)
            vectors.append(model.wv[word])

    if not vectors:
        raise RuntimeError(
            "No valid word vectors were found. "
            "Verify the consistency between the vocab and the Word2Vec model."
        )

    # Convert to a numpy array and save (dtype matches EmbeddingLoader's expected float32)
    vectors_array = np.stack(vectors).astype(np.float32)

    STATIC_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(STATIC_VECTORS, vectors_array)

    print("Static word vector matrix export complete")
    print(f"- output: {STATIC_VECTORS}")
    print(f"- count: {len(valid_words)}")


if __name__ == "__main__":
    export_static_vectors()
