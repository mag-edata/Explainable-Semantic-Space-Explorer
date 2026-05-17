"""
projection.py
=============
Module that projects high-dimensional embedding vectors onto 2D coordinates.

Supported methods:
    - PCA  (Principal Component Analysis):  Linear dimensionality reduction
    - UMAP (Uniform Manifold Approximation): Non-linear dimensionality reduction

Fixing the seed guarantees full reproducibility for both methods.
UMAP preserves local neighborhood structure and tends to produce
visually clearer cluster separation than PCA. On the other hand, PCA
allows quantifying how much information the projection retains via the
explained variance ratio (``explained_variance_ratio_``).

Formula (PCA):
    Z = X · V^T      (V is the matrix of the top-k principal components; here k=2)
    Contribution rate per component = λ_i / Σλ_j

Constraints:
    - The random seed must always be fixed (reproducibility guarantee).
    - No dependency on Streamlit or external APIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

logger = logging.getLogger(__name__)

# ---------- Constants ----------
DEFAULT_SEED: int = 42
DEFAULT_METHOD: str = "pca"
SUPPORTED_METHODS: tuple[str, ...] = ("pca", "umap")


# ---------- Exception classes ----------


class ProjectionError(Exception):
    """Base exception class specific to the projection module."""


class NotFittedError(ProjectionError):
    """Raised when results are accessed before ``fit_transform()`` is called."""


class InvalidMethodError(ProjectionError):
    """Raised when an unsupported projection method is specified."""


class InvalidVectorError(ProjectionError):
    """Raised when the input vectors have an invalid type or shape."""


# ---------- Data classes ----------


@dataclass
class ProjectionResult:
    """Summary of a 2D projection result.

    Attributes:
        coords_2d:          2D coordinates for each word, shape (N, 2).
                            ``coords_2d[i] = (x, y)`` corresponds to
                            vocabulary index i.
        explained_variance: Per-component contribution rates (PCA only).
                            ``[ratio for PC1, ratio for PC2]`` in [0.0, 1.0].
                            For UMAP, this is an empty list.
        method:             Projection method name (``"pca"`` or ``"umap"``).
        cluster_labels:     Cluster ID array, shape (N,).
                            ``None`` until ``attach_clusters()`` populates it.
        n_samples:          Vocabulary size N that was projected.
        seed:               Random seed used (for reproducibility checks).
    """

    coords_2d: np.ndarray
    explained_variance: List[float]
    method: str
    cluster_labels: np.ndarray | None
    n_samples: int
    seed: int


# ---------- Main class ----------


class Projector:
    """Projects embedding vectors into 2D using PCA or UMAP.

    One instance handles one method. To switch methods, create a new
    instance.

    The random seed is fixed at initialization, guaranteeing full
    reproducibility for the same data and seed.

    Attributes:
        _method: Projection method name (``"pca"`` or ``"umap"``).
        _seed:   Random seed.
    """

    def __init__(
        self,
        method: str = DEFAULT_METHOD,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Initialize the Projector.

        Args:
            method: Projection method (``"pca"`` or ``"umap"``). Default: ``"pca"``.
            seed:   Random seed (default: 42). Fixed for reproducibility.

        Raises:
            InvalidMethodError: If ``method`` is anything other than ``"pca"`` / ``"umap"``.
            TypeError:          If ``method`` is not str, or ``seed`` is not int.
        """
        if not isinstance(method, str):
            raise TypeError(
                f"method は str 型である必要があります。"
                f"受け取った型: {type(method)}"
            )
        if not isinstance(seed, int):
            raise TypeError(
                f"seed は int 型である必要があります。"
                f"受け取った型: {type(seed)}"
            )
        if method not in SUPPORTED_METHODS:
            raise InvalidMethodError(
                f"method は {SUPPORTED_METHODS} のいずれかである必要があります。"
                f"指定値: '{method}'"
            )

        self._method: str = method
        self._seed: int = seed

        logger.info(
            "Projector 初期化完了: method=%s, seed=%d",
            self._method,
            self._seed,
        )

    # ---------- Public API ----------

    def fit_transform(self, vectors: np.ndarray) -> ProjectionResult:
        """Project an embedding matrix into 2D coordinates.

        For PCA:
            Z = X · V^T  (V is the top-2 principal components)
            Component contribution rates are stored in ``explained_variance``.

        For UMAP:
            Builds a local neighborhood graph and projects to low dimensions.
            ``explained_variance`` is ``[]`` (undefined for non-linear methods).

        Args:
            vectors: Embedding matrix, shape (N, D). float32 dtype recommended.

        Returns:
            ProjectionResult: Object containing 2D coordinates, contribution
            rates, the method name, and so on.

        Raises:
            InvalidVectorError: If ``vectors`` is not an ndarray, or ``ndim != 2``.
        """
        self._validate_inputs(vectors)

        n_samples: int = vectors.shape[0]
        logger.debug(
            "fit_transform 開始: method=%s, n_samples=%d, dim=%d",
            self._method,
            n_samples,
            vectors.shape[1],
        )

        if self._method == "pca":
            result = self._fit_pca(vectors)
        else:
            result = self._fit_umap(vectors)

        logger.info(
            "fit_transform 完了: method=%s, n_samples=%d, coords_2d.shape=%s",
            self._method,
            n_samples,
            result.coords_2d.shape,
        )

        return result

    def attach_clusters(
        self,
        result: ProjectionResult,
        cluster_labels: np.ndarray,
    ) -> ProjectionResult:
        """Return a new ProjectionResult with cluster labels attached.

        The original ``result`` is not mutated. A new ``ProjectionResult``
        with ``cluster_labels`` replaced is returned (immutable operation).

        Args:
            result:         The ``ProjectionResult`` returned by ``fit_transform()``.
            cluster_labels: Cluster ID array for each word, shape (N,).
                            Pass the output of ``KMeansClusterer.get_labels()``.

        Returns:
            ProjectionResult: A new result object with ``cluster_labels`` attached.

        Raises:
            TypeError:          If ``result`` is not a ``ProjectionResult``,
                                or ``cluster_labels`` is not an ndarray.
            InvalidVectorError: If ``cluster_labels`` length does not match
                                ``result.n_samples``.
        """
        if not isinstance(result, ProjectionResult):
            raise TypeError(
                f"result は ProjectionResult 型である必要があります。"
                f"受け取った型: {type(result)}"
            )
        if not isinstance(cluster_labels, np.ndarray):
            raise TypeError(
                f"cluster_labels は np.ndarray 型である必要があります。"
                f"受け取った型: {type(cluster_labels)}"
            )
        if cluster_labels.shape[0] != result.n_samples:
            raise InvalidVectorError(
                f"cluster_labels の長さ ({cluster_labels.shape[0]}) が "
                f"result.n_samples ({result.n_samples}) と一致しません。"
            )

        logger.debug(
            "attach_clusters: n_samples=%d, unique_clusters=%d",
            result.n_samples,
            int(len(set(cluster_labels.tolist()))),
        )

        return ProjectionResult(
            coords_2d=result.coords_2d,
            explained_variance=result.explained_variance,
            method=result.method,
            cluster_labels=cluster_labels.copy(),
            n_samples=result.n_samples,
            seed=result.seed,
        )

    # ---------- Per-method projection (private) ----------

    def _fit_pca(self, vectors: np.ndarray) -> ProjectionResult:
        """Run a 2D projection with PCA.

        Formula:
            Z = X · V^T
            contribution_i = λ_i / Σ_j λ_j   (λ are eigenvalues)

        Uses ``sklearn.decomposition.PCA``. Passing ``seed`` as
        ``random_state`` guarantees reproducibility.

        Args:
            vectors: Embedding matrix, shape (N, D).

        Returns:
            ProjectionResult: Result containing 2D coordinates and principal-component contribution rates.
        """
        pca = PCA(n_components=2, random_state=self._seed)
        coords: np.ndarray = pca.fit_transform(vectors)

        explained: List[float] = pca.explained_variance_ratio_.tolist()

        logger.debug(
            "PCA 完了: 第1主成分 寄与率=%.4f, 第2主成分 寄与率=%.4f",
            explained[0],
            explained[1],
        )

        return ProjectionResult(
            coords_2d=coords,
            explained_variance=explained,
            method="pca",
            cluster_labels=None,
            n_samples=vectors.shape[0],
            seed=self._seed,
        )

    def _fit_umap(self, vectors: np.ndarray) -> ProjectionResult:
        """Run a 2D projection with UMAP.

        UMAP is a non-linear method that preserves local neighborhood
        structure while projecting to lower dimensions. Unlike PCA, it
        has no notion of explained variance, so an empty list is stored.

        Passing ``seed`` as ``random_state`` guarantees reproducibility.

        Args:
            vectors: Embedding matrix, shape (N, D).

        Returns:
            ProjectionResult: Result containing 2D coordinates
            (``explained_variance`` is ``[]``).
        """
        reducer = UMAP(n_components=2, random_state=self._seed)
        coords: np.ndarray = reducer.fit_transform(vectors)

        logger.debug("UMAP 完了: coords_2d.shape=%s", coords.shape)

        return ProjectionResult(
            coords_2d=coords,
            explained_variance=[],
            method="umap",
            cluster_labels=None,
            n_samples=vectors.shape[0],
            seed=self._seed,
        )

    # ---------- Validation ----------

    def _validate_inputs(self, vectors: np.ndarray) -> None:
        """Validate the input to ``fit_transform()``.

        Args:
            vectors: Embedding matrix to validate.

        Raises:
            InvalidVectorError: If ``vectors`` is not an ndarray.
            InvalidVectorError: If ``vectors.ndim`` is not 2.
            InvalidVectorError: If ``vectors`` has fewer than 2 rows
                                (PCA's minimum requirement).
        """
        if not isinstance(vectors, np.ndarray):
            raise InvalidVectorError(
                f"vectors は np.ndarray 型である必要があります。"
                f"受け取った型: {type(vectors)}"
            )
        if vectors.ndim != 2:
            raise InvalidVectorError(
                f"vectors は 2 次元配列（shape (N, D)）である必要があります。"
                f"受け取った次元数: {vectors.ndim}"
            )
        if vectors.shape[0] < 2:
            raise InvalidVectorError(
                f"投影には 2 サンプル以上必要です。"
                f"受け取った行数: {vectors.shape[0]}"
            )
