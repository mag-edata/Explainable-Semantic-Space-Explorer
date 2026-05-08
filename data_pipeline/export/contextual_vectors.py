"""
contextual_vectors.py

入力:
    data/metadata/vocab.json

出力:
    data/embeddings/contextual_vectors.npy  shape=(N, 384)  dtype=float32

本モジュールは、
vocab/merge.py により確定した統合語彙に対し、
Sentence-BERT(SBERT)を用いて、
文脈埋め込みベクトル行列(CONTEXTUAL_VECTORS)を生成し、
numpy形式で保存する前処理関数を提供する。

目的は、
統合語彙に対する高品質な意味表現を事前計算することで、
下流タスク(類似語検索、距離計算等)において
推論時にモデル呼び出しを行わず、
高速かつ再現性のあるベクトル演算を可能にすることである。

生成されるベクトルは、
単語単体入力に基づくSBERT埋め込みを用いており、
文脈埋め込みパラダイムの代表として静的埋め込み(Word2Vec)と対比する。

セットアップ時の HuggingFace ネットワークアクセスを前提とする
（要件定義書 CONST-02 を参照）。
"""

import json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- 入出力パス（このスクリプト固有）----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
CONTEXTUAL_VECTORS: Path = PROJECT_ROOT / "data" / "embeddings" / "contextual_vectors.npy"


def load_vocab() -> List[str]:
    """
    Returns
    -------
    List[str]
        vocab/merge.py により確定したソート済み統合語彙リスト。
    """
    with VOCAB_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "vocab" not in data:
        raise ValueError("vocab.json のフォーマットが不正です")

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

    print("文脈埋め込みベクトル行列の書き出しが完了しました")
    print(f"- 出力先: {CONTEXTUAL_VECTORS}")
    print(f"- 件数: {len(vocab)}")
    print(f"- 次元: {vectors.shape[1]}")


if __name__ == "__main__":
    export_contextual_vectors()
