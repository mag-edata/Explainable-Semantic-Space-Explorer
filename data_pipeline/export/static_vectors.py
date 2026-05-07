"""
static_vectors.py

本モジュールは、
vocab/merge.py により確定した統合語彙に対し、
学習済みのWord2Vecモデルを用いて、
静的単語ベクトル行列(STATIC_VECTORS)を生成し、
numpy形式で保存する前処理関数を提供する。

目的は、
統合語彙に対する静的な単語分散表現を事前に書き出すことで、
推論時にgensimへ依存することなく、
高速かつ一貫した語彙・ベクトル参照を可能にすることである。

保存対象とする語彙は、
Word2Vecモデル上で実際にベクトルが取得可能な単語のみに限定し、
語彙配列とベクトル行列のインデックス対応を保証する設計とする。
"""

import json
from pathlib import Path
from typing import List

import numpy as np
from gensim.models import Word2Vec

# ---------- 入出力パス（このスクリプト固有）----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
STATIC_VECTORS: Path = PROJECT_ROOT / "data" / "embeddings" / "static_vectors.npy"
W2V_MODEL: Path = PROJECT_ROOT / "models" / "w2v_brown10_simplewiki10_sg_300d_w5.model"


def load_vocab() -> List[str]:
    """
    Returns
    -------
    list[str]
        統合語彙リスト(ソート済み)
    """
    with VOCAB_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "vocab" not in data:
        raise ValueError(
            f"{VOCAB_JSON} does not contain required key: 'vocab'"
        )

    return data["vocab"]


def export_static_vectors() -> None:
    """
    保存物:
    - STATIC_VECTORS : shape = (N, dim)  dtype = float32

    インデックスは vocab.json と完全一致する。
    """
    vocab = load_vocab()
    model = Word2Vec.load(str(W2V_MODEL))
    valid_words: List[str] = []
    vectors: List[np.ndarray] = []

    # 語彙に対応する単語ベクトルを収集
    for word in vocab:
        # gensim Word2Vec は model.wv に語彙を保持する
        if word in model.wv:
            valid_words.append(word)
            vectors.append(model.wv[word])

    if not vectors:
        raise RuntimeError(
            "No valid word vectors were found. "
            "Check consistency between vocab and Word2Vec model."
        )

    # numpy配列へ変換し保存（dtype は EmbeddingLoader の期待値 float32 に合わせる）
    vectors_array = np.stack(vectors).astype(np.float32)

    STATIC_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(STATIC_VECTORS, vectors_array)

    print(f"Exported static word vectors: {len(valid_words)} words")
    print(f"- vectors : {STATIC_VECTORS}")


if __name__ == "__main__":
    export_static_vectors()
