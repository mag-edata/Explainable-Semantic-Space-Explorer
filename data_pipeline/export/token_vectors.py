"""
token_vectors.py

Inputs:
    ``data/metadata/vocab.json``

Outputs:
    ``data/embeddings/token_vectors.npy``  shape=(N, 384)  dtype=float32

This module builds the **isolated-word token-vector** matrix used by the
sentence-context mode (requirements v2.0, FR-20 / FR-21). Each row is the word
encoded on its own via ``ContextEncoder.encode_in_context(word, word)`` — that
is, the L2-normalized mean of the word's subword-token hidden states from the
transformer.

Why this asset (rather than reusing ``contextual_vectors.npy``):
    The runtime in-context vector is produced by the very same code path
    (``encode_in_context(sentence, word)``), so the isolated vectors here and
    the in-context vector live in the **same space**. Nearest-word search
    between an in-context vector and this matrix is therefore meaningful — a
    property the pooled SBERT word-anchor vectors do not guarantee.

Notes:
    - Encoding runs the transformer once per word on CPU, so a full export of
      the ~40k vocabulary takes on the order of 10-20 minutes. It is a
      one-time setup step.
    - Requires network access only the first time the model is fetched into the
      local cache (CONST-02); the export itself does no training.
"""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from inference.context_encoder import ContextEncoder
from inference.token_pooling import TokenPoolingError

logger = logging.getLogger(__name__)

# ---------- I/O paths (specific to this script) ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
TOKEN_VECTORS: Path = PROJECT_ROOT / "data" / "embeddings" / "token_vectors.npy"


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


def export_token_vectors() -> None:
    """
    Generate and save the isolated-word token-vector matrix.

    Output
    --------
    - TOKEN_VECTORS: shape = (N, dim), dtype = float32, L2-normalized.
      Indices match ``vocab.json`` exactly. A word that (very rarely) yields no
      usable token is left as a zero row.
    """
    vocab = load_vocab()
    if not vocab:
        raise ValueError("vocab.json is empty")

    encoder = ContextEncoder()
    encoder.load()

    # Infer the hidden dimension from the first word, then fill the matrix.
    first_vector = encoder.encode_in_context(vocab[0], vocab[0])
    dim = int(first_vector.shape[0])
    vectors = np.zeros((len(vocab), dim), dtype=np.float32)
    vectors[0] = first_vector

    skipped = 0
    for i in range(1, len(vocab)):
        word = vocab[i]
        try:
            vectors[i] = encoder.encode_in_context(word, word)
        except TokenPoolingError:
            skipped += 1
            logger.warning("token_vectors: no usable token for %r (left as zeros)", word)
        if i % 2000 == 0:
            print(f"  encoded {i}/{len(vocab)} words...")

    TOKEN_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(TOKEN_VECTORS, vectors)

    print("Token-vector matrix export complete")
    print(f"- output: {TOKEN_VECTORS}")
    print(f"- count: {len(vocab)}  (dim={dim}, zero-rows={skipped})")


if __name__ == "__main__":
    export_token_vectors()
