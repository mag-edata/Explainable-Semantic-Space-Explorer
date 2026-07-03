"""
test_vocab_pos.py
=================
Unit tests for the WordNet-based POS labelling (``data_pipeline/export/vocab_pos.py``).

Targets under test:
    - ``wordnet_pos_label()``: the coarse POS decision for a single word.

WordNet is injected as fake synset objects, so these tests verify the
labelling contract without requiring any NLTK data.

How to run:
    venv/bin/python3 -m unittest tests/test_vocab_pos.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.export.vocab_pos import wordnet_pos_label


class _FakeSynset:
    """Minimal stand-in for a WordNet synset exposing ``.pos()``."""

    def __init__(self, pos: str) -> None:
        self._pos = pos

    def pos(self) -> str:
        return self._pos


def _make_lookup(mapping: Dict[str, List[str]]):
    """Build a synset_lookup that returns fake synsets with the given POS tags."""

    def lookup(word: str) -> Sequence[object]:
        return [_FakeSynset(p) for p in mapping.get(word, [])]

    return lookup


class TestWordnetPosLabel(unittest.TestCase):
    """Verify the coarse POS decision rules."""

    def test_majority_pos_wins(self) -> None:
        """The coarse label with the most senses is chosen."""
        lookup = _make_lookup({"run": ["n", "v", "v"]})
        self.assertEqual(wordnet_pos_label("run", lookup), "verb")

    def test_single_pos(self) -> None:
        lookup = _make_lookup({"dog": ["n", "n", "n"]})
        self.assertEqual(wordnet_pos_label("dog", lookup), "noun")

    def test_adjective_satellite_folds_into_adjective(self) -> None:
        """The WordNet 's' (adjective satellite) tag maps to 'adjective'."""
        lookup = _make_lookup({"happy": ["a", "s"]})
        self.assertEqual(wordnet_pos_label("happy", lookup), "adjective")

    def test_adverb(self) -> None:
        lookup = _make_lookup({"quickly": ["r"]})
        self.assertEqual(wordnet_pos_label("quickly", lookup), "adverb")

    def test_tie_is_broken_by_priority(self) -> None:
        """Equal sense counts across POS resolve by fixed priority
        (noun > verb > adjective > adverb)."""
        lookup = _make_lookup({"tie": ["n", "v"]})
        self.assertEqual(wordnet_pos_label("tie", lookup), "noun")
        lookup2 = _make_lookup({"x": ["v", "r"]})
        self.assertEqual(wordnet_pos_label("x", lookup2), "verb")

    def test_no_synsets_is_any(self) -> None:
        """A word WordNet does not cover is labelled 'any'."""
        lookup = _make_lookup({})
        self.assertEqual(wordnet_pos_label("the", lookup), "any")

    def test_unmapped_pos_is_ignored(self) -> None:
        """A synset with an unrecognized POS tag contributes nothing;
        if nothing usable remains, the label is 'any'."""
        lookup = _make_lookup({"weird": ["z"]})
        self.assertEqual(wordnet_pos_label("weird", lookup), "any")


if __name__ == "__main__":
    unittest.main(verbosity=2)
