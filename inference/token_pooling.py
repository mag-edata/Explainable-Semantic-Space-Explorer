"""
inference/token_pooling.py
==========================
Pure helpers for turning a transformer's token-level output into a single
in-context vector for a target word.

These functions contain the fiddly, error-prone parts of sentence-context
encoding — locating the target word in the sentence and pooling the right
subword tokens — but they take plain data (character spans, offset mappings,
a hidden-state matrix), so they are deterministic and unit-testable **without
loading any model**. ``inference.context_encoder`` wires them to the real
transformer.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class TokenPoolingError(Exception):
    """Base exception for token-pooling failures."""


class WordNotInSentenceError(TokenPoolingError):
    """Raised when the target word cannot be found in the sentence."""


class NoTokenOverlapError(TokenPoolingError):
    """Raised when no transformer token overlaps the target word's characters."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def find_word_char_span(
    sentence: str,
    word: str,
    occurrence: int = 0,
) -> Optional[Tuple[int, int]]:
    """Locate the character span of a whole-word occurrence of ``word``.

    Matching is case-insensitive and restricted to whole words (so ``"bank"``
    does not match inside ``"banker"``).

    Args:
        sentence:   The sentence to search.
        word:       The target word (expected to be alphabetic).
        occurrence: Which match to return, 0-based (default: the first).

    Returns:
        ``(start, end)`` character offsets of the match, or ``None`` when the
        word does not occur (or ``occurrence`` is out of range).
    """
    if not word:
        return None

    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    matches = list(pattern.finditer(sentence))

    if occurrence < 0 or occurrence >= len(matches):
        return None

    match = matches[occurrence]
    return (match.start(), match.end())


def pool_token_vectors(
    hidden_states: Sequence[Sequence[float]],
    offsets: Sequence[Tuple[int, int]],
    char_span: Tuple[int, int],
) -> np.ndarray:
    """Mean-pool the hidden states of the tokens covering a character span.

    A token whose character offset ``(ts, te)`` overlaps the word span
    ``(ws, we)`` — i.e. ``ts < we and te > ws`` with ``te > ts`` — is included.
    The ``te > ts`` condition drops zero-width special tokens such as
    ``[CLS]`` / ``[SEP]`` (whose offset is ``(0, 0)``).

    Args:
        hidden_states: Per-token hidden vectors, shape ``(n_tokens, dim)``.
        offsets:       Per-token ``(start, end)`` character offsets, aligned
                       with ``hidden_states``.
        char_span:     The target word's ``(start, end)`` character span.

    Returns:
        np.ndarray: The mean of the overlapping token vectors, shape ``(dim,)``.

    Raises:
        NoTokenOverlapError: If no token overlaps ``char_span``.
    """
    word_start, word_end = char_span

    indices: List[int] = [
        i
        for i, (token_start, token_end) in enumerate(offsets)
        if token_start < word_end and token_end > word_start and token_end > token_start
    ]

    if not indices:
        raise NoTokenOverlapError(
            f"No token overlaps the character span {char_span}"
        )

    hidden = np.asarray(hidden_states, dtype=np.float32)
    return hidden[indices].mean(axis=0)
