"""
merge_vocab.py

本モジュールは、
以下2コーパスの統合語彙をJSON形式で書き出すことで
最終的に利用する統合語彙(vocabulary)を構築するための前処理関数を提供する。
- Brownコーパス由来の語彙
- Simple Wikipedia由来の語彙

目的は、異なる性質を持つ複数コーパス由来の語彙を統合し、
特定ドメインへの過度な依存を避けた汎用的な語彙集合を構築することであり、
類義語検索、単語分散表現、分類・生成モデル等において安定性と網羅性を両立させる。
"""

import json
from pathlib import Path

from data_pipeline.gen_brown_vocab import gen_brown_vocab
from data_pipeline.gen_wiki_vocab import gen_wiki_vocab

# ---------- 入出力パス（このスクリプト固有）----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"


def merge_vocab() -> list[str]:
    """
    Returns
    -------
    list[str]
        BrownとSimple Wikipediaの統合語彙をソートした語彙リスト。
    """
    brown_vocab = gen_brown_vocab()
    wiki_vocab = gen_wiki_vocab()

    merged_vocab = sorted(brown_vocab | wiki_vocab)
    return merged_vocab


if __name__ == "__main__":
    vocab = merge_vocab()

    output = {"vocab": vocab}

    VOCAB_JSON.parent.mkdir(parents=True, exist_ok=True)

    with VOCAB_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Merged vocab size: {len(vocab)}")
    print(f"Saved to {VOCAB_JSON}")
