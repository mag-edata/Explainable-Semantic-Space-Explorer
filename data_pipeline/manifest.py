"""
manifest.py

入力:
    data/embeddings/static_vectors.npy
    data/embeddings/contextual_vectors.npy

出力:
    data/manifest.json

本モジュールは、
data/embeddings/static_vectors.npy / contextual_vectors.npy の
shape および dtype を読み取り、
manifest.json を生成する。

目的は、
EmbeddingLoader._validate_against_manifest() による
shape / dtype 整合チェックの基準を、
実データから自動生成することにある。

EmbeddingLoader が照合するキーは "static_vectors" と "contextual_vectors" の
2 つのみ（vocab_pos 等は manifest に記載しなくても警告のみで動作）。
本スクリプトは現行 manifest.json の構造を踏襲する。
"""

import json
from pathlib import Path

import numpy as np

# ---------- 入出力パス（このスクリプト固有）----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
STATIC_VECTORS: Path = DATA_DIR / "embeddings" / "static_vectors.npy"
CONTEXTUAL_VECTORS: Path = DATA_DIR / "embeddings" / "contextual_vectors.npy"
MANIFEST_JSON: Path = DATA_DIR / "manifest.json"


def gen_manifest() -> dict:
    """
    実データの .npy ファイルから shape / dtype を読み取り、
    manifest.json として出力する dict を構築する。

    Returns
    -------
    dict
        manifest.json の内容。

    Raises
    ------
    FileNotFoundError
        必要な .npy ファイルが見つからない場合。
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
