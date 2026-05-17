"""
distance_metrics.py
===================
Fully self-contained implementation of cosine similarity.

Uses numpy only. The ``cosine_similarity`` helpers from scipy / sklearn are
never used. With zero dependence on external libraries, the entire
computation path remains fully traceable.

Mathematical definitions:
    L2 norm:
        ||v||_2 = sqrt(v_1^2 + v_2^2 + ... + v_D^2)
                = sqrt(v · v)

    Cosine similarity:
        cos(θ) = (a · b) / (||a||_2 × ||b||_2)
               = Σ(a_i × b_i) / (sqrt(Σa_i^2) × sqrt(Σb_i^2))

        Range: [-1.0, 1.0]
            1.0  → same direction (perfectly close)
            0.0  → orthogonal (unrelated)
           -1.0  → opposite direction (perfectly far)
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Minimum norm value used to guard against division by zero
_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DistanceMetricsError(Exception):
    """Base exception class specific to DistanceMetrics."""


class VectorDimensionError(DistanceMetricsError):
    """Raised when a vector has an invalid dimension or shape.

    Examples: two vectors with mismatched dimensions, or an array that
    is not one-dimensional.
    """


# ---------------------------------------------------------------------------
# DistanceMetrics
# ---------------------------------------------------------------------------

class DistanceMetrics:
    """Fully self-contained implementation class for cosine similarity.

    All methods are staticmethods, so no instantiation is required to use them.
    The implementation relies on numpy only, keeping every computation step
    fully traceable.

    Explicit formulas:
        L2 norm:         ||v||_2 = sqrt(v · v)
        Cosine:          cos(θ)  = (a · b) / (||a|| × ||b||)
        Batched form:    sim_i   = (M_i · q) / (||M_i|| × ||q||)

    Example usage::

        metrics = DistanceMetrics()

        norm = DistanceMetrics.l2_norm(vec)
        sim  = DistanceMetrics.cosine_similarity(vec_a, vec_b)
        sims = DistanceMetrics.cosine_similarity_batch(query, matrix)
        info = DistanceMetrics.explain(vec_a, vec_b)
    """

    @staticmethod
    def l2_norm(vector: np.ndarray) -> float:
        """Compute the L2 (Euclidean) norm of a vector.

        Formula:
            ||v||_2 = sqrt(v_1^2 + v_2^2 + ... + v_D^2)
                    = sqrt(v · v)

        Implementation: ``np.linalg.norm`` is intentionally not used.
        The norm is computed by hand as ``np.sqrt(np.dot(v, v))``.

        Args:
            vector: 1-D ndarray with shape (D,).

        Returns:
            float: The L2 norm. Returns 0.0 for the zero vector.

        Raises:
            VectorDimensionError: If ``vector`` is not 1-D.
            TypeError:            If ``vector`` is not an ndarray.
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError(
                f"vector must be of type np.ndarray. "
                f"Received type: {type(vector)}"
            )
        if vector.ndim != 1:
            raise VectorDimensionError(
                f"vector must be a 1-D array. "
                f"Received shape: {vector.shape}"
            )

        # Self-rolled implementation that avoids np.linalg.norm.
        # sqrt(v · v) = sqrt(v_1^2 + v_2^2 + ... + v_D^2)
        return float(np.sqrt(np.dot(vector, vector)))

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute the cosine similarity between two vectors.

        Formula:
            cos(θ) = (a · b) / (||a||_2 × ||b||_2)
                   = Σ(a_i × b_i) / (sqrt(Σa_i^2) × sqrt(Σb_i^2))

        Zero-division guard:
            If ||a|| < ε or ||b|| < ε, returns 0.0.
            A zero vector has no defined position in semantic space, so
            similarity is treated as undefined.

        Args:
            vec_a: 1-D ndarray with shape (D,).
            vec_b: 1-D ndarray with shape (D,).

        Returns:
            float: Cosine similarity, in the range [-1.0, 1.0].

        Raises:
            VectorDimensionError: If either vector is not 1-D, or if the
                                  two vectors have mismatched dimensions.
            TypeError:            If either argument is not an ndarray.
        """
        _validate_vector_pair(vec_a, vec_b)

        dot: float = float(np.dot(vec_a, vec_b))
        norm_a: float = DistanceMetrics.l2_norm(vec_a)
        norm_b: float = DistanceMetrics.l2_norm(vec_b)

        # Zero-division guard: treat zero vectors as similarity 0.0
        if norm_a < _EPSILON or norm_b < _EPSILON:
            logger.debug(
                "cosine_similarity: zero vector detected (norm_a=%.2e, norm_b=%.2e) -> 0.0",
                norm_a, norm_b,
            )
            return 0.0

        return dot / (norm_a * norm_b)

    @staticmethod
    def cosine_similarity_batch(
        query: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarities between a single query and every row of a matrix.

        Formula (matrix form):
            dot_products = M @ q                  shape: (N,)
            row_norms    = sqrt(diag(M @ M^T))    shape: (N,)
                         = sqrt(Σ M_ij^2 for each row i)
            query_norm   = ||q||_2                scalar
            similarities = dot_products / (row_norms × query_norm)

        Zero-division guard:
            Entries where ``denominator_i < ε`` are set to 0.0.

        Complexity:
            O(N × D): a single batched operation over the entire vocabulary.
            Combined with ``np.argpartition``, Top-K retrieval becomes
            O(N + k log k).

        Args:
            query:  Query vector. 1-D ndarray with shape (D,).
            matrix: Embedding matrix. 2-D ndarray with shape (N, D).

        Returns:
            np.ndarray: Cosine similarity with each row, shape (N,).
                        The dtype is float64.

        Raises:
            VectorDimensionError: If ``query`` is not 1-D, ``matrix`` is not
                                  2-D, or the query dimension does not match
                                  the number of columns in ``matrix``.
            TypeError:            If ``query`` or ``matrix`` is not an ndarray.
        """
        _validate_query_matrix(query, matrix)

        # Dot product: M @ q → shape (N,)
        # Matrix-vector product computes dot products with every row at once.
        dot_products: np.ndarray = matrix @ query

        # Norm of the query (uses the hand-rolled implementation)
        query_norm: float = DistanceMetrics.l2_norm(query)

        # L2 norm of each row: sqrt(Σ M_ij^2) → shape (N,)
        # np.linalg.norm is avoided; sum of squares followed by sqrt is done manually.
        row_norms: np.ndarray = np.sqrt((matrix * matrix).sum(axis=1))

        # Denominator: ||M_i|| × ||q|| → shape (N,)
        denominators: np.ndarray = row_norms * query_norm

        # Zero-division guard: substitute 1.0 for denominators below ε before
        # dividing, then overwrite the corresponding results with 0.0.
        zero_mask: np.ndarray = denominators < _EPSILON
        safe_denominators: np.ndarray = np.where(zero_mask, 1.0, denominators)

        similarities: np.ndarray = dot_products / safe_denominators
        similarities[zero_mask] = 0.0

        if zero_mask.any():
            logger.debug(
                "cosine_similarity_batch: zero-division guard applied to %d entries",
                int(zero_mask.sum()),
            )

        return similarities

    @staticmethod
    def explain(vec_a: np.ndarray, vec_b: np.ndarray) -> Dict[str, Any]:
        """Return the cosine-similarity computation breakdown as a dictionary.

        Used by the UI to display "why this score" explanations. Including
        the formula as a string lets the output embed not just the number
        but the structure of the formula itself.

        Formula (expanded into the ``formula`` string):
            cos(θ) = (a · b) / (||a||_2 × ||b||_2)
                   = {dot_product:.6f} / ({norm_a:.6f} × {norm_b:.6f})
                   = {similarity:.6f}

        Args:
            vec_a: Query word vector. 1-D ndarray with shape (D,).
            vec_b: Target word vector. 1-D ndarray with shape (D,).

        Returns:
            Dict[str, Any]: Breakdown of the computation.

            - "dot_product"  (float): Dot product  a · b
            - "norm_a"       (float): L2 norm of the query vector ||a||_2
            - "norm_b"       (float): L2 norm of the target vector ||b||_2
            - "denominator"  (float): Denominator ||a||_2 × ||b||_2
            - "similarity"   (float): Cosine similarity cos(θ)
            - "formula"      (str):   The formula expanded with concrete values

        Raises:
            VectorDimensionError: Invalid input (same conditions as
                                  ``l2_norm`` / ``cosine_similarity``).
            TypeError:            If the input is not an ndarray.
        """
        _validate_vector_pair(vec_a, vec_b)

        dot_product: float = float(np.dot(vec_a, vec_b))
        norm_a: float = DistanceMetrics.l2_norm(vec_a)
        norm_b: float = DistanceMetrics.l2_norm(vec_b)
        denominator: float = norm_a * norm_b
        similarity: float = DistanceMetrics.cosine_similarity(vec_a, vec_b)

        formula: str = (
            f"cos(θ) = (a · b) / (||a|| × ||b||)"
            f" = {dot_product:.6f} / ({norm_a:.6f} × {norm_b:.6f})"
            f" = {dot_product:.6f} / {denominator:.6f}"
            f" = {similarity:.6f}"
        )

        return {
            "dot_product": dot_product,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "denominator": denominator,
            "similarity": similarity,
            "formula": formula,
        }


# ---------------------------------------------------------------------------
# Module-private helpers (shared validation logic)
# ---------------------------------------------------------------------------

def _validate_vector_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> None:
    """Validate the type, shape, and dimensionality of two vectors.

    Args:
        vec_a: Vector A to validate.
        vec_b: Vector B to validate.

    Raises:
        TypeError:            If either argument is not an ndarray.
        VectorDimensionError: If either vector is not 1-D, or if their
                              dimensions do not match.
    """
    for name, vec in (("vec_a", vec_a), ("vec_b", vec_b)):
        if not isinstance(vec, np.ndarray):
            raise TypeError(
                f"{name} must be of type np.ndarray. "
                f"Received type: {type(vec)}"
            )
        if vec.ndim != 1:
            raise VectorDimensionError(
                f"{name} must be a 1-D array. "
                f"Received shape: {vec.shape}"
            )

    if vec_a.shape != vec_b.shape:
        raise VectorDimensionError(
            f"vec_a and vec_b have mismatched dimensions. "
            f"vec_a: {vec_a.shape}, vec_b: {vec_b.shape}"
        )


def _validate_query_matrix(query: np.ndarray, matrix: np.ndarray) -> None:
    """Validate the type and shape of a query / matrix pair used in batch mode.

    Args:
        query:  Query vector.
        matrix: Embedding matrix.

    Raises:
        TypeError:            If either argument is not an ndarray.
        VectorDimensionError: If the shape or dimensionality is invalid.
    """
    if not isinstance(query, np.ndarray):
        raise TypeError(
            f"query must be of type np.ndarray. "
            f"Received type: {type(query)}"
        )
    if not isinstance(matrix, np.ndarray):
        raise TypeError(
            f"matrix must be of type np.ndarray. "
            f"Received type: {type(matrix)}"
        )
    if query.ndim != 1:
        raise VectorDimensionError(
            f"query must be a 1-D array. "
            f"Received shape: {query.shape}"
        )
    if matrix.ndim != 2:
        raise VectorDimensionError(
            f"matrix must be a 2-D array. "
            f"Received shape: {matrix.shape}"
        )
    if query.shape[0] != matrix.shape[1]:
        raise VectorDimensionError(
            f"query dimension does not match the number of matrix columns. "
            f"query: {query.shape[0]}, matrix columns: {matrix.shape[1]}"
        )
