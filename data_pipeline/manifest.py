"""
manifest.py

Inputs:
    ``data/embeddings/static_vectors.npy``
    ``data/embeddings/contextual_vectors.npy``

Outputs:
    ``data/manifest.json``

This module reads the shape and dtype of
``data/embeddings/static_vectors.npy`` and ``contextual_vectors.npy``
and generates ``manifest.json``.

The goal is to auto-derive the reference values used for the shape /
dtype consistency check performed by
``EmbeddingLoader._validate_against_manifest()`` directly from the actual data.

``EmbeddingLoader`` only checks the keys ``"static_vectors"`` and
``"contextual_vectors"`` (other entries such as ``vocab_pos`` may be
omitted from the manifest with only a warning). This script follows the
structure of the current ``manifest.json``.
"""

import json
from pathlib import Path

import numpy as np

# ---------- I/O paths (specific to this script) ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
STATIC_VECTORS: Path = DATA_DIR / "embeddings" / "static_vectors.npy"
CONTEXTUAL_VECTORS: Path = DATA_DIR / "embeddings" / "contextual_vectors.npy"
MANIFEST_JSON: Path = DATA_DIR / "manifest.json"


def gen_manifest() -> dict:
    """
    Read shape / dtype from the actual ``.npy`` files and build the
    dictionary to be written out as ``manifest.json``.

    Returns
    -------
    dict
        Contents of ``manifest.json``.

    Raises
    ------
    FileNotFoundError
        If a required ``.npy`` file is missing.
    """
    if not STATIC_VECTORS.exists():
        raise FileNotFoundError(
            f"static_vectors が見つかりません: {STATIC_VECTORS}"
        )
    if not CONTEXTUAL_VECTORS.exists():
        raise FileNotFoundError(
            f"contextual_vectors が見つかりません: {CONTEXTUAL_VECTORS}"
        )

    static_arr = np.load(STATIC_VECTORS)
    contextual_arr = np.load(CONTEXTUAL_VECTORS)

    manifest = {
        "static_vectors": {
            "shape": list(static_arr.shape),
            "dtype": str(static_arr.dtype),
            "source": "Word2Vec trained on Brown + SimpleWiki",
            "training_date": "YYYY-MM-DD",
        },
        "contextual_vectors": {
            "shape": list(contextual_arr.shape),
            "dtype": str(contextual_arr.dtype),
            "model": "all-MiniLM-L6-v2",
            "encoding_type": "single-word embedding",
        },
        "vocab_alignment": "index matched across all files",
    }
    return manifest


if __name__ == "__main__":
    manifest = gen_manifest()

    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"manifest.json を生成しました: {MANIFEST_JSON}")
    print(f"- static_vectors:     shape={manifest['static_vectors']['shape']}, "
          f"dtype={manifest['static_vectors']['dtype']}")
    print(f"- contextual_vectors: shape={manifest['contextual_vectors']['shape']}, "
          f"dtype={manifest['contextual_vectors']['dtype']}")
