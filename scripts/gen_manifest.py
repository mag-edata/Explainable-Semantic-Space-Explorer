"""
gen_manifest.py

本モジュールは、
assets/embeddings/static_vectors.npy / sbert_vectors.npy の
shape および dtype を読み取り、
manifest.json を生成する。

目的は、
EmbeddingLoader._validate_against_manifest() による
shape / dtype 整合チェックの基準を、
実データから自動生成することにある。

EmbeddingLoader が照合するキーは "static_vectors" と "sbert_vectors" の
2 つのみ（vocab_pos 等は manifest に記載しなくても警告のみで動作）。
本スクリプトは現行 manifest.json の構造を踏襲する。
"""

import json
from pathlib import Path

import numpy as np

# ---------- 入出力パス（このスクリプト固有）----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
STATIC_VECTORS: Path = ASSETS_DIR / "embeddings" / "static_vectors.npy"
SBERT_VECTORS: Path = ASSETS_DIR / "embeddings" / "sbert_vectors.npy"
MANIFEST_JSON: Path = ASSETS_DIR / "manifest.json"


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
    if not SBERT_VECTORS.exists():
        raise FileNotFoundError(
            f"sbert_vectors が見つかりません: {SBERT_VECTORS}"
        )

    static_arr = np.load(STATIC_VECTORS)
    sbert_arr = np.load(SBERT_VECTORS)

    manifest = {
        "static_vectors": {
            "shape": list(static_arr.shape),
            "dtype": str(static_arr.dtype),
            "source": "Word2Vec trained on Brown + SimpleWiki",
            "training_date": "YYYY-MM-DD",
        },
        "sbert_vectors": {
            "shape": list(sbert_arr.shape),
            "dtype": str(sbert_arr.dtype),
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
    print(f"- static_vectors: shape={manifest['static_vectors']['shape']}, "
          f"dtype={manifest['static_vectors']['dtype']}")
    print(f"- sbert_vectors:  shape={manifest['sbert_vectors']['shape']}, "
          f"dtype={manifest['sbert_vectors']['dtype']}")
