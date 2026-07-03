"""
inference/context_encoder.py
============================
Runtime encoder that produces the in-context vector of a word inside a
sentence, using the locally deployed transformer (all-MiniLM-L6-v2).

This is the single component that runs a model at inference time
(requirements v2.0, CONST-03 as amended): it encodes the user's sentence
locally, with **no network access and no training**. Inference runs in eval
mode under ``torch.no_grad()``, so it is deterministic (satisfying the
reproducibility constraint, CONST-06).

Space consistency:
    The vector returned by :meth:`ContextEncoder.encode_in_context` lives in
    the same space as the isolated-word token vectors exported by
    ``data_pipeline/export/token_vectors.py`` — because that exporter builds
    each vocabulary vector by calling ``encode_in_context(word, word)``. Both
    are "the L2-normalized mean of the target word's subword-token hidden
    states", so nearest-word search between an in-context vector and the
    vocabulary is meaningful.
"""

from __future__ import annotations

import logging

import numpy as np

from inference.token_pooling import (
    WordNotInSentenceError,
    find_word_char_span,
    pool_token_vectors,
)

logger = logging.getLogger(__name__)

# HuggingFace repo id for the transformer backing all-MiniLM-L6-v2.
_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ContextEncoderError(Exception):
    """Base exception for :class:`ContextEncoder` failures."""


class ContextEncoder:
    """Encodes a word in the context of a sentence into a single vector.

    The tokenizer and transformer are loaded lazily on first use and cached
    on the instance, so a single ``ContextEncoder`` can be reused across many
    sentences (e.g. wrapped in ``@st.cache_resource`` by the UI).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        """Initialize the encoder without loading the model.

        Args:
            model_name: HuggingFace model id (default: the all-MiniLM-L6-v2
                backbone). Loaded from the local cache; no download occurs at
                inference time.
        """
        self.model_name: str = model_name
        self._tokenizer = None
        self._model = None
        self._torch = None

    def load(self) -> None:
        """Load the tokenizer and transformer (idempotent).

        Raises:
            ContextEncoderError: If ``torch`` / ``transformers`` are missing or
                the model cannot be loaded locally.
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ContextEncoderError(
                "torch/transformers are required for sentence-context mode"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ContextEncoderError(
                f"Failed to load model '{self.model_name}' locally: {exc}"
            ) from exc

        self._torch = torch
        self._model.eval()
        logger.info("ContextEncoder loaded model %s", self.model_name)

    def encode_in_context(
        self,
        sentence: str,
        word: str,
        occurrence: int = 0,
        normalize: bool = True,
    ) -> np.ndarray:
        """Return the in-context vector of ``word`` within ``sentence``.

        The sentence is tokenized (with character offsets), passed through the
        transformer, and the hidden states of the tokens covering the target
        word are mean-pooled into a single vector.

        Args:
            sentence:   The sentence containing the word.
            word:       The target word (matched as a whole word,
                        case-insensitively).
            occurrence: Which occurrence of the word to encode, 0-based.
            normalize:  If ``True`` (default), L2-normalize the result so that
                        cosine similarity equals a dot product.

        Returns:
            np.ndarray: The in-context vector, shape ``(hidden_dim,)``,
            dtype ``float32``.

        Raises:
            WordNotInSentenceError: If the word does not occur in the sentence.
            NoTokenOverlapError:    If no token overlaps the word's characters.
            ContextEncoderError:    If the model cannot be loaded.
        """
        self.load()

        span = find_word_char_span(sentence, word, occurrence=occurrence)
        if span is None:
            raise WordNotInSentenceError(
                f"'{word}' does not occur as a whole word in the sentence"
            )

        encoded = self._tokenizer(
            sentence,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()

        with self._torch.no_grad():
            output = self._model(**encoded)

        hidden = output.last_hidden_state[0].cpu().numpy()
        vector = pool_token_vectors(hidden, offsets, span)

        if normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                vector = vector / norm

        return vector.astype(np.float32)
