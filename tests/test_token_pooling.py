"""
test_token_pooling.py
=====================
Unit tests for the pure sentence-context pooling helpers
(``inference/token_pooling.py``).

These exercise the fiddly word-location and token-pooling logic with plain
data, so no transformer model is loaded.

How to run:
    venv/bin/python3 -m unittest tests/test_token_pooling.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.token_pooling import (
    NoTokenOverlapError,
    find_word_char_span,
    pool_token_vectors,
)


class TestFindWordCharSpan(unittest.TestCase):
    """Tests for ``find_word_char_span``."""

    def test_finds_simple_word(self) -> None:
        span = find_word_char_span("I sat by the river bank", "bank")
        self.assertEqual(span, (19, 23))

    def test_is_case_insensitive(self) -> None:
        span = find_word_char_span("The Bank raised rates", "bank")
        self.assertEqual(span, (4, 8))

    def test_whole_word_only(self) -> None:
        """'bank' must not match inside 'banker'."""
        span = find_word_char_span("She is a banker downtown", "bank")
        self.assertIsNone(span)

    def test_occurrence_selects_nth_match(self) -> None:
        sentence = "a bank near the bank"
        self.assertEqual(find_word_char_span(sentence, "bank", occurrence=0), (2, 6))
        self.assertEqual(find_word_char_span(sentence, "bank", occurrence=1), (16, 20))

    def test_missing_word_returns_none(self) -> None:
        self.assertIsNone(find_word_char_span("no match here", "bank"))

    def test_out_of_range_occurrence_returns_none(self) -> None:
        self.assertIsNone(find_word_char_span("one bank only", "bank", occurrence=2))

    def test_empty_word_returns_none(self) -> None:
        self.assertIsNone(find_word_char_span("anything", ""))


class TestPoolTokenVectors(unittest.TestCase):
    """Tests for ``pool_token_vectors``."""

    def setUp(self) -> None:
        # 4 tokens: [CLS](0,0), "river"(0,5), "bank"(6,10), [SEP](0,0)
        self.hidden = [
            [1.0, 1.0],   # [CLS]
            [2.0, 0.0],   # river
            [4.0, 8.0],   # bank
            [9.0, 9.0],   # [SEP]
        ]
        self.offsets = [(0, 0), (0, 5), (6, 10), (0, 0)]

    def test_pools_only_target_word_tokens(self) -> None:
        """Only the 'bank' token (span 6-10) is pooled; specials excluded."""
        vec = pool_token_vectors(self.hidden, self.offsets, (6, 10))
        np.testing.assert_allclose(vec, [4.0, 8.0])

    def test_pools_mean_of_multiple_subwords(self) -> None:
        """A word split into two subword tokens is averaged."""
        hidden = [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [0.0, 0.0]]
        # two subwords of the target word cover chars 6-9 and 9-12
        offsets = [(0, 0), (6, 9), (9, 12), (0, 0)]
        vec = pool_token_vectors(hidden, offsets, (6, 12))
        np.testing.assert_allclose(vec, [3.0, 6.0])  # mean of rows 1 and 2

    def test_special_tokens_excluded(self) -> None:
        """Zero-width special tokens never contribute even at span start 0."""
        vec = pool_token_vectors(self.hidden, self.offsets, (0, 5))
        np.testing.assert_allclose(vec, [2.0, 0.0])  # only 'river', not [CLS]

    def test_no_overlap_raises(self) -> None:
        with self.assertRaises(NoTokenOverlapError):
            pool_token_vectors(self.hidden, self.offsets, (50, 60))

    def test_returns_correct_shape(self) -> None:
        vec = pool_token_vectors(self.hidden, self.offsets, (6, 10))
        self.assertEqual(vec.shape, (2,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
