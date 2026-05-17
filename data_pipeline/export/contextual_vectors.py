"""
contextual_vectors.py

Inputs:
    ``data/metadata/vocab.json``

Outputs:
    ``data/embeddings/contextual_vectors.npy``  shape=(N, 384)  dtype=float32

This module provides the preprocessing function that, against the merged
vocabulary finalized by ``vocab/merge.py``, builds the contextual
embedding vector matrix (``CONTEXTUAL_VECTORS``) using
Sentence-BERT (SBERT) and saves it in NumPy format.

The goal is to pre-compute high-quality semantic representations for
the merged vocabulary, so that downstream tasks (similar-word search,
distance computation, etc.) can perform fast, reproducible vector
operations without invoking the model at inference time.

The vectors produced use SBERT embeddings derived from single-word
inputs, treated as a representative of the contextual embedding
paradigm for contrast with static embeddings (Word2Vec).

Assumes HuggingFace network access is available at setup time
(see CONST-02 in the requirements document).
"""

import json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- I/O paths (specific to this script) ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
CONTEXTUAL_VECTORS: Path = PROJECT_ROOT / "data" / "embeddings" / "contextual_vectors.npy"


def load_vocab() -> List[str]:
    """
    Returns
    -------
    List[str]
        Sorted merged vocabulary list finalized by ``vocab/merge.py``.
    """
    with VOCAB_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "vocab" not in data:
        raise ValueError("vocab.json has an invalid format")

    return data["vocab"]


def export_contextual_vectors() -> None:
    """
    Generate and save contextual embedding vectors for the merged vocabulary using SBERT.

    Output
    --------
    - CONTEXTUAL_VECTORS: shape = (N, dim), dtype = float32

    Indices match ``vocab.json`` exactly.
    Vectors are L2-normalized in advance, so cosine similarity can be
    computed by a plain dot product for speed.
    """
    vocab = load_vocab()

    # Lightweight, fast, general-purpose sentence embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode word strings as sentence embeddings with SBERT
    vectors = model.encode(
        vocab,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize (assumes cosine computation)
    )

    # Match EmbeddingLoader's expected dtype (float32)
    vectors = vectors.astype(np.float32)

    CONTEXTUAL_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(CONTEXTUAL_VECTORS, vectors)

    print("Contextual embedding vector matrix export complete")
    print(f"- output: {CONTEXTUAL_VECTORS}")
    print(f"- count: {len(vocab)}")
    print(f"- dim: {vectors.shape[1]}")


if __name__ == "__main__":
    export_contextual_vectors()
