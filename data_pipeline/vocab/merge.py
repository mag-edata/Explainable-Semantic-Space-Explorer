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

The merged union is then passed through ``curate_vocab`` (requirements
v2.0, FR-18 / FR-19), which drops non-words, abbreviation noise, and
proper-noun fragments by requiring lexical-resource membership. The
returned list is the *candidate* vocabulary; it is finalized (restricted
to words that also have a Word2Vec vector) later in
``export/static_vectors.py`` to guarantee index alignment.
"""

import json
from pathlib import Path

from data_pipeline.vocab.curate import curate_vocab
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
        Sorted, curated vocabulary list combining Brown and Simple
        Wikipedia. Non-words and noise are removed by ``curate_vocab``.
    """
    brown_vocab = gen_brown()
    wiki_vocab = gen_wiki()

    merged_vocab = curate_vocab(brown_vocab | wiki_vocab)
    return merged_vocab


if __name__ == "__main__":
    vocab = merge()

    output = {"vocab": vocab}

    VOCAB_JSON.parent.mkdir(parents=True, exist_ok=True)

    with VOCAB_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Merged vocabulary generated")
    print(f"- output: {VOCAB_JSON}")
    print(f"- count: {len(vocab)}")
