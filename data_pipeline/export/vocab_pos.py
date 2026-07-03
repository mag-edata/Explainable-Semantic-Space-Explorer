"""
vocab_pos.py

Inputs:
    ``data/metadata/vocab.json``

Outputs:
    ``data/metadata/vocab_pos.npy``  shape=(N,)  dtype=<U_>

This module assigns a coarse-grained Part-of-Speech (POS) label to each
word in the merged vocabulary finalized by ``vocab/merge.py`` and saves
the label array in NumPy format, index-aligned with the vocabulary array.

POS derivation (requirements v2.0, fixes roadmap item 1.2):
    POS labels are derived from **WordNet**, one word at a time. The
    previous implementation ran the NLTK perceptron tagger over the whole
    *sorted* vocabulary list (``pos_tag(['a', 'aaa', 'aachen', ...])``);
    because that tagger is context-sensitive, tagging an alphabetical list
    produced unreliable labels. WordNet gives each word its dictionary POS
    independently of any surrounding "sentence", removing that contamination
    and keeping the result deterministic and reproducible (CONST-06).

    For a word with multiple POS (e.g. "run" is both noun and verb) the
    coarse label with the most WordNet senses is chosen; ties are broken by
    a fixed priority. Words WordNet does not cover (e.g. kept stopwords) are
    labeled ``"any"``, matching the previous label vocabulary.

The goal is a design where pre-computed POS information enables fast and
consistent POS-based filtering at inference time, without running dynamic
POS tagging at runtime.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
from nltk.corpus import wordnet

from data_pipeline._common.nltk_setup import ensure_nltk_resource

# ---------- I/O paths (specific to this script) ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VOCAB_JSON: Path = PROJECT_ROOT / "data" / "metadata" / "vocab.json"
VOCAB_POS: Path = PROJECT_ROOT / "data" / "metadata" / "vocab_pos.npy"

# ---------- POS mapping ----------
# WordNet synset POS tags ('n', 'v', 'a', 's', 'r') → coarse-grained labels.
# 's' is the adjective-satellite tag, folded into "adjective".
_WN_POS_TO_COARSE: Dict[str, str] = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}

# Tie-break priority when a word has an equal number of senses across POS.
_POS_PRIORITY: Dict[str, int] = {"noun": 4, "verb": 3, "adjective": 2, "adverb": 1}


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


def wordnet_pos_label(
    word: str,
    synset_lookup: Callable[[str], Sequence[object]],
) -> str:
    """
    Derive a coarse-grained POS label for a single word from WordNet.

    The synset lookup is injected so the labelling logic can be unit-tested
    without any NLTK data.

    Parameters
    ----------
    word : str
        Word to label.
    synset_lookup : Callable[[str], Sequence[object]]
        Function returning the WordNet synsets for a word (e.g.
        ``wordnet.synsets``). Each synset must expose ``.pos()``.

    Returns
    -------
    str
        One of ``"noun"``, ``"verb"``, ``"adjective"``, ``"adverb"``, or
        ``"any"`` (when WordNet has no usable POS for the word).
    """
    counts: Dict[str, int] = {}
    for synset in synset_lookup(word):
        coarse = _WN_POS_TO_COARSE.get(synset.pos())
        if coarse is not None:
            counts[coarse] = counts.get(coarse, 0) + 1

    if not counts:
        return "any"

    return max(counts, key=lambda label: (counts[label], _POS_PRIORITY[label]))


def export_vocab_pos() -> None:
    """
    Assign a WordNet-derived POS label to each vocabulary word and save an
    index-aligned label array.

    Output
    --------
    - VOCAB_POS: shape = (N,)
        Each element is a coarse-grained POS label.
        Indices match ``vocab.json`` exactly.
    """
    ensure_nltk_resource("corpora/wordnet", "wordnet")

    vocab = load_vocab()

    pos_labels: List[str] = [wordnet_pos_label(word, wordnet.synsets) for word in vocab]
    pos_array = np.array(pos_labels)

    VOCAB_POS.parent.mkdir(parents=True, exist_ok=True)
    np.save(VOCAB_POS, pos_array)

    # Small summary so the label distribution is visible at generation time.
    labels, counts = np.unique(pos_array, return_counts=True)
    dist = ", ".join(f"{lbl}={cnt}" for lbl, cnt in zip(labels.tolist(), counts.tolist()))

    print("POS label array export complete")
    print(f"- output: {VOCAB_POS}")
    print(f"- count: {len(pos_array)}")
    print(f"- distribution: {dist}")


if __name__ == "__main__":
    export_vocab_pos()
