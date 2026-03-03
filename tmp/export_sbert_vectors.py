"""
export_sbert_vectors.py

本モジュールは、
merge_vocab.pyにより確定した統合語彙に対し、
Sentence-BERT(SBERT)を用いた文脈埋め込みベクトルを生成し、
SBERT_WORDSおよびSBERT_VECTORSとしてnumpy形式で保存する。

目的は，
統合語彙に対する高品質な意味表現を事前計算することで、
下流タスク(類似語検索，距離計算等)において
推論時にモデル呼び出しを行わず、
高速かつ再現性のあるベクトル演算を可能にすることである。

生成されるベクトルは，
単語単体入力に基づくSBERT埋め込みを用いており、
文脈を含まない静的な表現として扱う設計とする。
"""

import json
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.paths import VOCAB_JSON, SBERT_WORDS, SBERT_VECTORS

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

def export_sbert_vectors() -> None:
    """
    SBERTを用いて統合語彙の静的ベクトルを生成し保存する。

    保存物
    --------
    - SBERT_WORDS   : shape = (N,)
    - SBERT_VECTORS : shape = (N, dim)

    両者はインデックス対応を保証する。
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
        normalize_embeddings=True  # L2正規化(コサイン計算前提)
    )

    np.save(SBERT_WORDS, np.array(vocab))
    np.save(SBERT_VECTORS, vectors)

    print("SBERT vector export completed.")
    print(f"Total words: {len(vocab)}")
    print(f"Dim: {vectors.shape[1]}")


if __name__ == "__main__":
    export_sbert_vectors()