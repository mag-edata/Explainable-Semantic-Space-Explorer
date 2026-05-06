"""
export_contextual_vectors.py

本モジュールは、
merge_vocab.pyにより確定した統合語彙に対し、
Sentence-BERT(SBERT)を用いた文脈埋め込みベクトルを生成し、
CONTEXTUAL_VECTORSとしてnumpy形式で保存する。

目的は、
統合語彙に対する高品質な意味表現を事前計算することで、
下流タスク(類似語検索、距離計算等)において
推論時にモデル呼び出しを行わず、
高速かつ再現性のあるベクトル演算を可能にすることである。

生成されるベクトルは、
単語単体入力に基づくSBERT埋め込みを用いており、
文脈埋め込みパラダイムの代表として静的埋め込み(Word2Vec)と対比する。

セットアップ時の HuggingFace ネットワークアクセスを前提とする
（CLAUDE.md 制約 #2: 「セットアップ時 data_pipeline/ 実行時のみ許可」に該当）。
"""

import json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- 入出力パス（このスクリプト固有）----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
CONTEXTUAL_VECTORS: Path = PROJECT_ROOT / "data" / "embeddings" / "contextual_vectors.npy"


def load_vocab() -> List[str]:
    """
    Returns
    -------
    List[str]
        merge_vocab.pyにより確定したソート済み統合語彙リスト。
    """
    with VOCAB_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "vocab" not in data:
        raise ValueError("Invalid vocab.json format")

    return data["vocab"]


def export_contextual_vectors() -> None:
    """
    SBERTを用いて統合語彙の文脈埋め込みベクトルを生成し保存する。

    保存物
    --------
    - CONTEXTUAL_VECTORS : shape = (N, dim)  dtype = float32

    インデックスは vocab.json と完全一致する。
    ベクトルは事前にL2正規化済みであり、
    コサイン類似度計算を内積により高速化可能とする。
    """
    vocab = load_vocab()

    # 軽量・高速な汎用文埋め込みモデル
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # SBERTを用いて単語列を文埋め込みとしてエンコード
    vectors = model.encode(
        vocab,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2正規化(コサイン計算前提)
    )

    # dtype は EmbeddingLoader の期待値 float32 に合わせる
    vectors = vectors.astype(np.float32)

    CONTEXTUAL_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(CONTEXTUAL_VECTORS, vectors)

    print("Contextual vector export completed.")
    print(f"Total words: {len(vocab)}")
    print(f"Dim: {vectors.shape[1]}")


if __name__ == "__main__":
    export_contextual_vectors()
