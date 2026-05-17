"""
clustering.py
=============
Module for grouping word embeddings via KMeans clustering.

To realize cosine-distance clustering, vectors are L2-normalized first
and then sklearn's KMeans (Euclidean distance) is applied.

    cos(a, b) = dot(a, b) / (‖a‖ · ‖b‖)

Once normalized to unit vectors, ‖â‖ = ‖b̂‖ = 1, hence
    ‖â - b̂‖² = 2 - 2·cos(a, b)
which means minimizing Euclidean distance is equivalent to minimizing
cosine distance.

Constraints:
    - Norm computation is implemented from scratch (np.linalg.norm forbidden).
    - Cosine / normalization is implemented from scratch (sklearn cosine_similarity forbidden).
    - The random seed must always be fixed (reproducibility guarantee).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ---------- Constants ----------
DEFAULT_SEED: int = 42
DEFAULT_N_CLUSTERS: int = 8
DEFAULT_MAX_ITER: int = 300
EPSILON: float = 1e-10  # Zero-division guard (handles near-zero-norm vectors)


# ---------- Exception classes ----------


class ClusterError(Exception):
    """Base exception class specific to the cluster module."""


class NotFittedError(ClusterError):
    """Raised when ``get_labels()`` / ``get_result()`` is called before ``fit()``."""


class InvalidClusterCountError(ClusterError):
    """Raised when ``n_clusters`` is invalid (≤ 0 or larger than the vocabulary size)."""


class UnfitVectorError(ClusterError):
    """Raised when the input vectors have an invalid type or shape."""


# ---------- Data classes ----------


@dataclass
class ClusterResult:
    """Summary of a KMeans clustering result.

    Attributes:
        labels:     Cluster IDs for each word (0-based). Shape (N,).
        n_clusters: The cluster count that was requested.
        inertia:    Within-cluster sum of squared Euclidean distances
                    (computed in the normalized space). Smaller values
                    indicate denser clusters.
        seed:       The random seed used (for reproducibility checks).
        n_samples:  Vocabulary size N that was clustered.
    """

    labels: np.ndarray
    n_clusters: int
    inertia: float
    seed: int
    n_samples: int


# ---------- Main class ----------


class KMeansClusterer:
    """Cosine-distance KMeans clustering class.

    Internally, vectors are L2-normalized and then sklearn's KMeans
    (Euclidean distance) is applied. The result is equivalent to
    clustering by cosine distance.

    Equivalence formula (cosine ↔ Euclidean):
        ‖â - b̂‖² = 2 - 2·cos(a, b)   (â and b̂ are unit vectors)

    The random seed is fixed at initialization time, guaranteeing reproducibility.

    Attributes:
        _n_clusters: Number of clusters.
        _seed:       Random seed.
        _max_iter:   Maximum number of KMeans iterations.
        _result:     ``ClusterResult`` populated after ``fit()``. ``None`` = not fitted.
    """

    def __init__(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        seed: int = DEFAULT_SEED,
        max_iter: int = DEFAULT_MAX_ITER,
    ) -> None:
        """Initialize the KMeansClusterer.

        Args:
            n_clusters: Number of clusters (default: 8).
                        Consistency with vocabulary size N is validated at ``fit()`` time.
            seed:       Random seed (default: 42).
                        Guarantees full reproducibility for the same data and seed.
            max_iter:   Maximum number of KMeans iterations (default: 300).

        Raises:
            TypeError:                If ``n_clusters`` / ``seed`` / ``max_iter`` is not int.
            InvalidClusterCountError: If ``n_clusters`` is less than 1.
        """
        if not isinstance(n_clusters, int):
            raise TypeError(
                f"n_clusters は int 型である必要があります。"
                f"受け取った型: {type(n_clusters)}"
            )
        if not isinstance(seed, int):
            raise TypeError(
                f"seed は int 型である必要があります。"
                f"受け取った型: {type(seed)}"
            )
        if not isinstance(max_iter, int):
            raise TypeError(
                f"max_iter は int 型である必要があります。"
                f"受け取った型: {type(max_iter)}"
            )
        if n_clusters < 1:
            raise InvalidClusterCountError(
                f"n_clusters は 1 以上である必要があります。"
                f"指定値: {n_clusters}"
            )

        self._n_clusters: int = n_clusters
        self._seed: int = seed
        self._max_iter: int = max_iter
        self._result: ClusterResult | None = None

        logger.info(
            "KMeansClusterer 初期化完了: n_clusters=%d, seed=%d, max_iter=%d",
            self._n_clusters,
            self._seed,
            self._max_iter,
        )

    # ---------- Public API ----------

    def fit(self, vectors: np.ndarray) -> ClusterResult:
        """Apply KMeans clustering to a vector matrix.

        L2-normalizes the input vectors before running clustering.
        After normalization, minimizing Euclidean distance is equivalent
        to minimizing cosine distance.

        Pipeline:
            1. Validate inputs.
            2. Compute L2 norms in batch: ``‖v_i‖ = sqrt(Σ v_{ij}²)``.
            3. Row-wise L2 normalization: ``v̂_i = v_i / ‖v_i‖``.
            4. ``sklearn.KMeans.fit(unit_vectors)``.
            5. Build and return ``ClusterResult``.

        Args:
            vectors: Embedding matrix, shape (N, D). float32 dtype recommended.

        Returns:
            ClusterResult: Result object containing cluster labels, inertia, etc.

        Raises:
            UnfitVectorError:         If ``vectors`` is not an ndarray, or its ndim != 2.
            InvalidClusterCountError: If ``n_clusters`` exceeds the vocabulary size N.
        """
        self._validate_inputs(vectors)

        n_samples: int = vectors.shape[0]
        logger.debug(
            "fit 開始: n_samples=%d, dim=%d, n_clusters=%d",
            n_samples,
            vectors.shape[1],
            self._n_clusters,
        )

        unit_vectors: np.ndarray = self._normalize_rows(vectors)

        kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._seed,
            max_iter=self._max_iter,
        )
        kmeans.fit(unit_vectors)

        self._result = ClusterResult(
            labels=kmeans.labels_.copy(),
            n_clusters=self._n_clusters,
            inertia=float(kmeans.inertia_),
            seed=self._seed,
            n_samples=n_samples,
        )

        logger.info(
            "fit 完了: n_samples=%d, n_clusters=%d, inertia=%.6f",
            n_samples,
            self._n_clusters,
            self._result.inertia,
        )

        return self._result

    def get_labels(self) -> np.ndarray:
        """Return the cluster label array.

        Returns:
            np.ndarray: Cluster IDs for each word (0-based), shape (N,).

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        if self._result is None:
            raise NotFittedError(
                "get_labels() は fit() を呼び出した後にのみ使用できます。"
            )
        return self._result.labels

    def get_result(self) -> ClusterResult:
        """Return the full clustering result.

        Returns:
            ClusterResult: Contains ``labels`` / ``n_clusters`` / ``inertia`` /
            ``seed`` / ``n_samples``.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        if self._result is None:
            raise NotFittedError(
                "get_result() は fit() を呼び出した後にのみ使用できます。"
            )
        return self._result

    # ---------- Hand-rolled implementations: norm and normalization (staticmethod) ----------

    @staticmethod
    def _l2_norm_batch(matrix: np.ndarray) -> np.ndarray:
        """Compute the L2 norm for each row of a matrix in batch.

        Formula:
            ‖v‖ = sqrt(Σ_j v_j²)   for each row v

        ``np.linalg.norm`` is not used. The sum of squares followed by
        ``sqrt`` is computed by hand:
            ``norms = sqrt( sum(matrix * matrix, axis=1) )``
        ``EPSILON`` is not added here to clamp against division by zero;
        the normalization step guards against it separately.

        Args:
            matrix: Input matrix, shape (N, D).

        Returns:
            np.ndarray: L2 norm of each row, shape (N,).
        """
        row_sq_sum: np.ndarray = (matrix * matrix).sum(axis=1)
        norms: np.ndarray = np.sqrt(row_sq_sum)
        return norms

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row of the matrix and return the unit-vector matrix.

        Formula:
            â_i = a_i / ‖a_i‖   for each row a_i

        Zero vectors (where ``‖a_i‖ < EPSILON``) are kept as-is (division
        skipped). This prevents NaN values.

        Args:
            matrix: Input matrix, shape (N, D).

        Returns:
            np.ndarray: Normalized matrix, shape (N, D). Each row has L2 norm 1.0
            (except for skipped zero rows).
        """
        norms: np.ndarray = KMeansClusterer._l2_norm_batch(matrix)

        # Mask for zero vectors (avoids division)
        valid_mask: np.ndarray = norms >= EPSILON

        # Copy and normalize (do not mutate the original matrix)
        unit_matrix: np.ndarray = matrix.copy().astype(np.float64)
        unit_matrix[valid_mask] = (
            matrix[valid_mask] / norms[valid_mask, np.newaxis]
        )

        logger.debug(
            "_normalize_rows: %d / %d 行を正規化（ゼロベクトル %d 行をスキップ）",
            int(valid_mask.sum()),
            matrix.shape[0],
            int((~valid_mask).sum()),
        )

        return unit_matrix

    # ---------- Validation ----------

    def _validate_inputs(self, vectors: np.ndarray) -> None:
        """Validate the input to ``fit()``.

        Args:
            vectors: Embedding matrix to validate.

        Raises:
            UnfitVectorError:         If ``vectors`` is not an ndarray.
            UnfitVectorError:         If ``vectors.ndim`` is not 2.
            InvalidClusterCountError: If ``n_clusters`` exceeds the vocabulary size N.
        """
        if not isinstance(vectors, np.ndarray):
            raise UnfitVectorError(
                f"vectors は np.ndarray 型である必要があります。"
                f"受け取った型: {type(vectors)}"
            )
        if vectors.ndim != 2:
            raise UnfitVectorError(
                f"vectors は 2 次元配列（shape (N, D)）である必要があります。"
                f"受け取った次元数: {vectors.ndim}"
            )

        n_samples: int = vectors.shape[0]
        if self._n_clusters > n_samples:
            raise InvalidClusterCountError(
                f"n_clusters ({self._n_clusters}) が語彙数 N ({n_samples}) を超えています。"
                f"n_clusters ≤ N である必要があります。"
            )
