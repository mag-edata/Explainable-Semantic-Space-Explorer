"""
test_curate.py
==============
Unit tests for the vocabulary curation logic (``data_pipeline/vocab/curate.py``).

Targets under test:
    - ``is_english_word()``: the per-word KEEP / DROP decision.

The lexical-resource dependencies (WordNet, stopwords) are injected as fakes,
so these tests verify the curation contract without requiring any NLTK data.

How to run:
    venv/bin/python3 -m unittest tests/test_curate.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.vocab.curate import is_english_word


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Words that our fake WordNet "knows" (i.e. return a non-empty synset list).
_FAKE_WORDNET: set[str] = {"dog", "running", "cat", "happy", "quickly"}

# A small stopword set (function words that WordNet does not cover).
_FAKE_STOPWORDS: set[str] = {"the", "of", "he", "she", "not"}


def _fake_synset_lookup(word: str) -> Sequence[object]:
    """Return a non-empty sequence iff the word is in the fake WordNet."""
    return [object()] if word in _FAKE_WORDNET else []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsEnglishWord(unittest.TestCase):
    """Verify the KEEP / DROP rules of ``is_english_word``."""

    def _decide(self, word: str) -> bool:
        return is_english_word(word, _FAKE_STOPWORDS, _fake_synset_lookup)

    def test_content_word_in_wordnet_is_kept(self) -> None:
        """A word present in WordNet is kept."""
        self.assertTrue(self._decide("dog"))

    def test_inflected_form_in_wordnet_is_kept(self) -> None:
        """An inflected form resolvable by WordNet is kept."""
        self.assertTrue(self._decide("running"))

    def test_stopword_absent_from_wordnet_is_kept(self) -> None:
        """A function word is kept via the stopword allowlist even when
        WordNet does not cover it."""
        self.assertFalse("the" in _FAKE_WORDNET)  # guard: really not in WordNet
        self.assertTrue(self._decide("the"))

    def test_non_word_noise_is_dropped(self) -> None:
        """Proper-noun fragments / abbreviation noise are dropped."""
        for junk in ("aabach", "aad", "aacta", "xyzzq"):
            with self.subTest(word=junk):
                self.assertFalse(self._decide(junk))

    def test_non_lowercase_alpha_form_is_dropped(self) -> None:
        """Anything that is not lowercase ASCII alphabetic is dropped,
        regardless of lexical membership."""
        for bad in ("Dog", "abc123", "e-mail", "über", ""):
            with self.subTest(word=bad):
                self.assertFalse(self._decide(bad))

    def test_form_rule_precedes_membership(self) -> None:
        """The form rule rejects malformed input before membership is
        consulted (a capitalized known word is still dropped)."""
        # "Cat" would match WordNet after lowercasing, but curation does not
        # lowercase; the tokenizer is responsible for that upstream.
        self.assertFalse(self._decide("Cat"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
