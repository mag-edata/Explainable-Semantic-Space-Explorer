"""
vocab_pos.py

Inputs:
    ``data/metadata/vocab.json``

Outputs:
    ``data/metadata/vocab_pos.npy``  shape=(N,)  dtype=<U_>

This module assigns Part-of-Speech (POS) labels to the merged vocabulary
finalized by ``vocab/merge.py`` and saves the coarse-grained POS array
in NumPy format, with full index alignment to the vocabulary array.

The goal is a design where pre-computed POS information enables fast
and consistent POS-based filtering at inference time, without running
dynamic POS tagging at runtime.

POS definitions and mapping rules are consolidated within this module,
guaranteeing reproducibility for vocabulary composition and downstream
tasks.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
from nltk import pos_tag

from data_pipeline._common.nltk_setup import ensure_nltk_resource

# ---------- I/O paths (specific to this script) ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
VOCAB_POS: Path = PROJECT_ROOT / "data" / "metadata" / "vocab_pos.npy"

# ---------- POS mapping ----------
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
        Sorted vocabulary list finalized by ``vocab/merge.py``.
    """
    with VOCAB_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "vocab" not in data:
        raise ValueError(
            f"{VOCAB_JSON} is missing the required 'vocab' key"
        )

    return data["vocab"]


def map_pos_tag(tag: str) -> str:
    """
    Convert a Penn Treebank POS tag to a coarse-grained POS label.

    Parameters
    ----------
    tag : str
        POS tag in Penn Treebank format.

    Returns
    -------
    str
        Coarse-grained POS label. Returns one of:
        - "any"
        - "noun"
        - "verb"
        - "adjective"
        - "adverb"
    """
    return POS_MAP.get(tag, "any")


def export_vocab_pos() -> None:
    """
    Assign POS labels to the merged vocabulary and save an
    index-aligned POS array.

    Output
    --------
    - VOCAB_POS: shape = (N,)
        Each element is a coarse-grained POS label.
        Indices match ``vocab.json`` exactly.
    """
    ensure_nltk_resource("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")

    vocab = load_vocab()

    # Run POS tagging per word
    tagged = pos_tag(vocab)
    # Map to coarse-grained POS labels
    pos_labels: List[str] = [map_pos_tag(tag) for _, tag in tagged]

    pos_array = np.array(pos_labels)

    VOCAB_POS.parent.mkdir(parents=True, exist_ok=True)
    np.save(VOCAB_POS, pos_array)

    print("POS label array export complete")
    print(f"- output: {VOCAB_POS}")
    print(f"- count: {len(pos_array)}")


if __name__ == "__main__":
    export_vocab_pos()
