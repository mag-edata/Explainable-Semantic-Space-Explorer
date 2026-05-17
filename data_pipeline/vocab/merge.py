"""
merge.py

Inputs:
    Return values of ``gen_brown()`` / ``gen_wiki()``
    (indirectly references the Brown corpus + Simple Wikipedia).

Outputs:
    ``data/metadata/vocab.json``

This module provides the preprocessing function that builds the final
merged vocabulary by writing the union of the two corpora's vocabularies
into JSON:
- Vocabulary from the Brown corpus
- Vocabulary from Simple Wikipedia

The goal is to merge vocabularies from multiple corpora of different
characteristics into a general-purpose vocabulary set that avoids
excessive dependence on any specific domain, balancing stability and
coverage in synonym search, word embeddings, and classification /
generation models.
"""

import json
from pathlib import Path

from data_pipeline.vocab.gen_brown import gen_brown
from data_pipeline.vocab.gen_wiki import gen_wiki

# ---------- I/O paths (specific to this script) ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"


def merge() -> list[str]:
    """
    Returns
    -------
    list[str]
        Sorted vocabulary list combining Brown and Simple Wikipedia.
    """
    brown_vocab = gen_brown()
    wiki_vocab = gen_wiki()

    merged_vocab = sorted(brown_vocab | wiki_vocab)
    return merged_vocab


if __name__ == "__main__":
    vocab = merge()

    output = {"vocab": vocab}

    VOCAB_JSON.parent.mkdir(parents=True, exist_ok=True)

    with VOCAB_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("統合語彙を生成しました")
    print(f"- 出力先: {VOCAB_JSON}")
    print(f"- 件数: {len(vocab)}")
