"""
tokenizer.py

This module provides preprocessing functions for token normalization and
tokenization.

The goal is to keep the token definition consistent across the entire
project, guaranteeing consistency and reproducibility of preprocessing
throughout vocabulary generation, corpus generation, model training,
and inference.
"""

from typing import Iterable, List, Optional

from data_pipeline._common.token_definition import TOKEN_CONSTRAINT_PATTERN, TOKEN_EXTRACT_PATTERN


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    """
    Process an already-tokenized stream.
    Designed for text that is provided as a word-level sequence
    (for example, the NLTK Brown corpus).

    Parameters
    ----------
    tokens : Iterable[str]
        Pre-tokenized stream.

    Returns
    -------
    List[str]
        Token stream after normalization and filtering.
    """
    return [w.lower() for w in tokens if TOKEN_CONSTRAINT_PATTERN.match(w)]


def tokenize_text(text: str, vocab: Optional[set[str]] = None) -> List[str]:
    """
    Process raw text.
    Designed for text that is provided at the sentence / passage level
    (for example, Hugging Face Datasets' Simple Wikipedia).

    Parameters
    ----------
    text : str
        Raw text.
    vocab : set[str], optional
        Allowed vocabulary set.
        When provided, only tokens that belong to this set are returned.

    Returns
    -------
    List[str]
        Token stream after tokenization, normalization, filtering, and
        (when applicable) vocabulary restriction.
    """
    text = text.lower()
    candidates = TOKEN_EXTRACT_PATTERN.findall(text)
    tokens = [w for w in candidates if TOKEN_CONSTRAINT_PATTERN.match(w)]

    if vocab is not None:
        tokens = [w for w in tokens if w in vocab]

    return tokens
