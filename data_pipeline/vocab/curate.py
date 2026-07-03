"""
curate.py

Inputs:
    An iterable of candidate words (the Brown ∪ Simple-Wikipedia union
    produced inside ``merge.py``).

Outputs:
    A curated, sorted ``list[str]`` (no file output; consumed by ``merge()``).

Purpose (requirements v2.0, FR-18 / FR-19):
    The v1 vocabulary (83,823 words) was a raw frequency-filtered union and
    contained non-words, abbreviation noise, and proper-noun fragments
    ("aabach", "aaahh", "aad", "aacta", "aachen", ...). Such neighbors
    destroy the learning experience the product now targets, so vocabulary
    quality is enforced here as an explicit, reproducible curation step.

Curation rules:
    A candidate word is KEPT if and only if all of the following hold.

    1. Form: the word is lowercase ASCII alphabetic only (``^[a-z]+$``).
       The upstream tokenizer already guarantees this; it is re-checked here
       so curation is correct and testable in isolation.

    2. Membership in an English lexical resource — either:
         - WordNet: ``wordnet.synsets(word)`` is non-empty. This admits
           content words (nouns, verbs, adjectives, adverbs); WordNet's
           ``morphy`` also resolves inflected forms (plurals, verb forms),
           so pedagogically useful variants such as "running" are kept.
         - NLTK English stopwords: high-value function words (he, she, of,
           the, not, ...) that WordNet omits but which are legitimate,
           frequent vocabulary worth exploring.

    Words that pass neither membership test are dropped as noise.

    The *frequency* component of FR-18 is enforced upstream by the per-corpus
    ``min_freq`` threshold in ``gen_brown()`` / ``gen_wiki()``; curation here
    adds the lexical-membership component.

Reproducibility:
    Curation is deterministic (membership tests + ``sorted``), so re-running
    the pipeline yields an identical vocabulary (CONST-06).
"""

import re
from typing import Callable, Iterable, List, Sequence, Set

from nltk.corpus import stopwords, wordnet

from data_pipeline._common.nltk_setup import ensure_nltk_resource

# Lowercase ASCII alphabetic only. Upstream normalization already lowercases
# and restricts to [a-zA-Z]+; after lowering, the valid form is [a-z]+.
_LOWER_ALPHA_PATTERN = re.compile(r"^[a-z]+$")


def _ensure_resources() -> None:
    """Ensure the NLTK corpora required for curation are available locally.

    Downloads ``wordnet`` and ``stopwords`` if missing. ``omw-1.4`` is
    fetched on a best-effort basis: English ``synsets`` lookups do not
    require it, but some NLTK versions load it lazily.
    """
    ensure_nltk_resource("corpora/wordnet", "wordnet")
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    try:
        ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    except Exception:
        # English-only synset lookups work without Open Multilingual WordNet.
        pass


def is_english_word(
    word: str,
    stopword_set: Set[str],
    synset_lookup: Callable[[str], Sequence[object]],
) -> bool:
    """Decide whether a single candidate word should be kept by curation.

    The lexical-resource dependencies are injected so the decision logic can
    be unit-tested without any NLTK data.

    Parameters
    ----------
    word : str
        Candidate word (expected already lowercased by the tokenizer).
    stopword_set : set[str]
        Set of English stopwords to keep even when absent from WordNet.
    synset_lookup : Callable[[str], Sequence[object]]
        Function returning the synsets for a word (e.g. ``wordnet.synsets``).
        A non-empty result means the word exists in WordNet.

    Returns
    -------
    bool
        ``True`` if the word passes curation, ``False`` if it is dropped.
    """
    if not _LOWER_ALPHA_PATTERN.match(word):
        return False
    if word in stopword_set:
        return True
    return bool(synset_lookup(word))


def curate_vocab(words: Iterable[str]) -> List[str]:
    """Apply the curation rules to a candidate vocabulary.

    Parameters
    ----------
    words : Iterable[str]
        Candidate words (the Brown ∪ Simple-Wikipedia union).

    Returns
    -------
    list[str]
        Sorted, de-duplicated vocabulary that passes curation.
    """
    _ensure_resources()
    stopword_set: Set[str] = set(stopwords.words("english"))
    kept = {w for w in words if is_english_word(w, stopword_set, wordnet.synsets)}
    return sorted(kept)


if __name__ == "__main__":
    # Smoke check on a tiny hand-picked sample (requires NLTK data).
    sample = ["dog", "running", "the", "aabach", "aad", "aachen", "happy", "xyzzq"]
    curated = curate_vocab(sample)
    print("Curation smoke check")
    print(f"- input : {sample}")
    print(f"- kept  : {curated}")
