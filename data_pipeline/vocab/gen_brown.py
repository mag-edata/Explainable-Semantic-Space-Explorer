"""
gen_brown.py

Inputs:
    ``data/nltk_data/corpora/brown`` (the NLTK Brown corpus).

Outputs:
    ``set[str]`` (no file output; intermediate data passed to ``merge.py``).

This module provides the preprocessing function for building a vocabulary
suitable for NLP tasks from the NLTK Brown corpus.

Excluding low-frequency words suppresses noise and stabilizes training
for word embeddings, classifiers, and language models.
"""

from collections import Counter

from nltk.corpus import brown

from data_pipeline._common.nltk_setup import ensure_nltk_resource
from data_pipeline._common.tokenizer import normalize_tokens


def gen_brown(min_freq: int = 10) -> set[str]:
    """
    Build a vocabulary set containing only words above a minimum frequency.

    Parameters
    ----------
    min_freq : int, optional
        Minimum occurrence count required for inclusion in the vocabulary.

    Returns
    -------
    set[str]
        Set of words that satisfy the frequency condition.
    """
    # Fetch every word from the Brown corpus
    ensure_nltk_resource("corpora/brown", "brown")
    words_raw = brown.words()

    # Apply the tokenizer
    words = normalize_tokens(words_raw)

    # Count word occurrences
    counter = Counter(words)
    # Return words occurring at least min_freq times
    return {w for w, f in counter.items() if f >= min_freq}


if __name__ == "__main__":
    vocab = gen_brown()
    print("Brownコーパスから語彙を生成しました")
    print(f"- 件数: {len(vocab)}")
