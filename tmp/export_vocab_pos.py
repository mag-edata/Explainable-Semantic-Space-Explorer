"""
export_vocab_pos.py

本モジュールは、
merge_vocab.pyにより確定した統合語彙に対し、
品詞(POS:Part-of-Speech)ラベルを付与し、
語彙配列と完全にインデックス整合した
coarse-grained POS配列をnumpy形式で保存する。

目的は、
推論時に動的なPOSタグ付け処理を実行することなく、
事前計算済みのPOS情報を用いて
高速かつ一貫した品詞制約を適用可能とする設計を実現することである。

POS定義およびマッピング規則は本モジュール内に集約し、
語彙構成や下流タスクにおける再現性を保証する設計とする。
"""

import json
from typing import List
import numpy as np
import nltk
from nltk import pos_tag
from src.utils.paths import VOCAB_JSON, PROCESSED_DIR, VOCAB_POS

# POS mapping
POS_MAP = {
    # Noun
    "NN": "noun",
    "NNS": "noun",
    # "NNP": "noun",
    # "NNPS": "noun",

    # Verb
    "VB": "verb",
    "VBD": "verb",
    "VBG": "verb",
    "VBN": "verb",
    "VBP": "verb",
    "VBZ": "verb",

    # Adjective
    "JJ": "adjective",
    "JJR": "adjective",
    "JJS": "adjective",

    # Adverb
    "RB": "adverb",
    "RBR": "adverb",
    "RBS": "adverb",
}

def load_vocab() -> List[str]:
    """
    Returns
    -------
    List[str]
        merge_vocab.pyにより確定したソート済み語彙リスト。
    """
    with VOCAB_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "vocab" not in data:
        raise ValueError(
            f"{VOCAB_JSON} does not contain required key: 'vocab'"
        )

    return data["vocab"]

def map_pos_tag(tag: str) -> str:
    """
    Penn Treebank POSタグをcoarse-grained POSへ変換する。

    Parameters
    ----------
    tag : str
        Penn Treebank形式のPOSタグ

    Returns
    -------
    str
        coarse-grained POSラベル。
        以下のいずれかを返す:
        - "any"
        - "noun"
        - "verb"
        - "adjective"
        - "adverb"
    """
    return POS_MAP.get(tag, "any")

def export_vocab_pos() -> None:
    """
    統合語彙に対してPOSラベルを付与し、
    インデックス整合したPOS配列を保存する。

    保存物
    --------
    - VOCAB_POS : shape = (N,)
        各要素はcoarse-grained POSラベル。
        インデックスは語彙配列(VOCAB_WORDS / SBERT_WORDS等)と完全一致する。
    """
    # averaged_perceptron_tagger未取得の場合はダウンロード
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")

    vocab = load_vocab()

    # 単語単位でPOS taggingを実行
    tagged = pos_tag(vocab)
    # coarse-grained POS へマッピング
    pos_labels: List[str] = [
        map_pos_tag(tag) for _, tag in tagged
    ]

    pos_array = np.array(pos_labels)

    VOCAB_POS.parent.mkdir(parents=True, exist_ok=True)
    np.save(VOCAB_POS, pos_array)

    print("POS export completed.")
    print(f"Total words: {len(pos_array)}")
    print(f"Saved to: {VOCAB_POS}")


if __name__ == "__main__":
    export_vocab_pos()