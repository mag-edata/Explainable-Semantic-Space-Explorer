"""
embedding_loader.py
===================
Loads embedding assets and performs index alignment checks.

Reads every file under the ``data/`` directory and compares the actual
data with the expected values recorded in ``manifest.json``, so that any
index inconsistency is caught at startup.

The loaded data is held on the ``EmbeddingLoader`` instance. Passing it
into ``SimilarityEngine`` is the caller's responsibility (dependency
injection).

Directory layout:
    data/
    ├── embeddings/
    │   ├── static_vectors.npy   # [N, 300] Word2Vec float32
    │   └── contextual_vectors.npy    # [N, 384] SBERT float32
    ├── metadata/
    │   ├── vocab.json           # {"vocab": [...]} list form (the loader converts it to a dict)
    │   └── vocab_pos.npy        # [N] POS label array
    └── manifest.json            # Expected shape / dtype values
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class EmbeddingLoaderError(Exception):
    """Base exception class specific to EmbeddingLoader."""


class IndexAlignmentError(EmbeddingLoaderError):
    """Raised when the index count (N) does not match across files.

    Example: ``static_vectors.shape[0] == 50000`` but the vocab has 49999 entries.
    """


class ManifestViolationError(EmbeddingLoaderError):
    """Raised when actual shape / dtype does not match the manifest entry.

    Example: manifest says ``shape=(50000, 300)`` but the loaded array has
    ``shape=(49999, 300)``.
    """


# ---------------------------------------------------------------------------
# EmbeddingLoader
# ---------------------------------------------------------------------------

class EmbeddingLoader:
    """Loads embedding assets and performs index alignment checks.

    Calling ``load_all()`` reads every file and runs the alignment checks.
    Once the checks pass, each piece of data is held as an instance attribute.

    Example usage::

        from pathlib import Path
        from core.embedding_loader import EmbeddingLoader
        from core.distance_metrics import DistanceMetrics
        from core.similarity_engine import SimilarityEngine

        loader = EmbeddingLoader(Path("data"))
        loader.load_all()

        static_engine = SimilarityEngine(
            vectors=loader.static_vectors,
            vocab=loader.vocab,
            pos_tags=loader.pos,
            metrics=DistanceMetrics(),
        )

    Attributes:
        data_root:           Path to the ``data/`` directory
        emb_dir:             Path to the ``embeddings/`` subdirectory
        meta_dir:            Path to the ``metadata/`` subdirectory
        static_vectors:      Word2Vec static embedding matrix, shape (N, 300)
        contextual_vectors:  SBERT contextual embedding matrix, shape (N, 384)
        vocab:               Word → index dictionary {"word": index}
        pos:                 POS label array, shape (N,)
        manifest:            Contents of ``manifest.json``
    """

    def __init__(self, data_root: Path) -> None:
        """Initialize the EmbeddingLoader.

        Args:
            data_root: Path object pointing to the ``data/`` directory.

        Raises:
            FileNotFoundError: If ``data_root`` does not exist as a directory.
        """
        if not data_root.exists():
            raise FileNotFoundError(
                f"data ディレクトリが見つかりません: {data_root}"
            )

        self.data_root: Path = data_root
        self.emb_dir: Path = data_root / "embeddings"
        self.meta_dir: Path = data_root / "metadata"

        # Populated after load_all() is called
        self.static_vectors: np.ndarray | None = None
        self.contextual_vectors: np.ndarray | None = None
        self.vocab: Dict[str, int] | None = None
        self.pos: np.ndarray | None = None
        self.manifest: dict | None = None

        logger.info("EmbeddingLoader 初期化: data_root=%s", data_root)

    def load_all(self) -> None:
        """Load every asset file and run the index alignment checks.

        Execution order:
        1. Load ``manifest.json``
        2. Load embedding vectors (``.npy``)
        3. Load metadata (``vocab.json``, ``vocab_pos.npy``)
        4. Run the index alignment checks

        Raises:
            FileNotFoundError:      If a required file is missing.
            IndexAlignmentError:    If N (vocabulary size) does not match
                                    across files.
            ManifestViolationError: If actual data deviates from the manifest.
        """
        logger.info("資産の読み込みを開始します")
        self._load_manifest()
        self._load_embeddings()
        self._load_metadata()
        self._validate()
        logger.info(
            "全資産の読み込み完了: n_vocab=%d", len(self.vocab)
        )

    # -----------------------------------------------------------------------
    # Private: loading
    # -----------------------------------------------------------------------

    def _load_manifest(self) -> None:
        """Load ``manifest.json``.

        Raises:
            FileNotFoundError: If ``manifest.json`` does not exist.
        """
        manifest_path = self.data_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json が見つかりません: {manifest_path}"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        logger.debug("manifest.json 読み込み完了: %s", manifest_path)

    def _load_embeddings(self) -> None:
        """Load the embedding vector files (``.npy``).

        Raises:
            FileNotFoundError: If an embedding file does not exist.
        """
        static_path = self.emb_dir / "static_vectors.npy"
        contextual_path = self.emb_dir / "contextual_vectors.npy"

        for path in (static_path, contextual_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"埋め込みファイルが見つかりません: {path}"
                )

        self.static_vectors = np.load(static_path)
        self.contextual_vectors = np.load(contextual_path)

        logger.debug(
            "埋め込み読み込み完了: static=%s, contextual=%s",
            self.static_vectors.shape,
            self.contextual_vectors.shape,
        )

    def _load_metadata(self) -> None:
        """Load ``vocab.json`` and ``vocab_pos.npy``.

        Format of ``vocab.json``: ``{"vocab": ["word0", "word1", ...]}``
        (list form). After loading, it is automatically converted into a
        ``{"word": index, ...}`` dictionary.

        Raises:
            FileNotFoundError: If a metadata file does not exist.
        """
        vocab_path = self.meta_dir / "vocab.json"
        pos_path = self.meta_dir / "vocab_pos.npy"

        for path in (vocab_path, pos_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"メタデータファイルが見つかりません: {path}"
                )

        with open(vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # If vocab.json is in {"vocab": [word0, word1, ...]} list form,
        # convert that list to a dictionary. The expected form downstream is
        # {"word": index, ...} (Dict[str, int]).
        if isinstance(raw, dict) and "vocab" in raw and isinstance(raw["vocab"], list):
            word_list = raw["vocab"]
            self.vocab = {word: idx for idx, word in enumerate(word_list)}
            logger.debug("vocab.json をリスト形式から辞書形式に変換しました（件数=%d）", len(self.vocab))
        else:
            self.vocab = raw

        self.pos = np.load(pos_path)

        logger.debug(
            "メタデータ読み込み完了: vocab_size=%d, pos_shape=%s",
            len(self.vocab),
            self.pos.shape,
        )

    # -----------------------------------------------------------------------
    # Private: alignment checks
    # -----------------------------------------------------------------------

    def _validate(self) -> None:
        """Run index alignment checks across every file.

        Check order:
        1. Confirm all assets are loaded (i.e. ``load_all`` was called correctly).
        2. Confirm N (vocabulary size) is consistent across the 4 files.
        3. Compare against the shape / dtype recorded in ``manifest.json``.
        4. Confirm vocab index values form contiguous integers in [0, N-1].

        Raises:
            IndexAlignmentError:    N does not match, or index values are invalid.
            ManifestViolationError: Shape / dtype does not match the manifest.
        """
        # 1. Confirm everything is loaded
        if any(x is None for x in (
            self.static_vectors, self.contextual_vectors, self.vocab, self.pos
        )):
            raise EmbeddingLoaderError(
                "load_all() を実行する前に _validate() が呼ばれました"
            )

        vocab_size: int = len(self.vocab)

        # 2. Verify N (vocabulary size) is consistent
        checks: List[tuple[int, str]] = [
            (self.static_vectors.shape[0],     "static_vectors"),
            (self.contextual_vectors.shape[0], "contextual_vectors"),
            (self.pos.shape[0],                "vocab_pos"),
        ]
        for actual_n, name in checks:
            if actual_n != vocab_size:
                raise IndexAlignmentError(
                    f"インデックス不整合: {name} の行数={actual_n} に対し "
                    f"vocab の件数={vocab_size} が一致しません"
                )

        # 3. Compare shape / dtype against manifest.json
        self._validate_against_manifest(self.static_vectors,     "static_vectors")
        self._validate_against_manifest(self.contextual_vectors, "contextual_vectors")

        # 4. Confirm vocab indices form a contiguous range [0, N-1]
        index_values = list(self.vocab.values())
        unique_indices = set(index_values)

        if len(unique_indices) != vocab_size:
            raise IndexAlignmentError(
                f"vocab にインデックス値の重複があります: "
                f"ユニーク数={len(unique_indices)}, 語彙数={vocab_size}"
            )

        expected_indices = set(range(vocab_size))
        if unique_indices != expected_indices:
            missing = expected_indices - unique_indices
            raise IndexAlignmentError(
                f"vocab のインデックスが [0, N-1] の連続整数になっていません。"
                f"欠落インデックスの例: {sorted(missing)[:5]}"
            )

        logger.info("整合チェック: 全項目通過 (n_vocab=%d)", vocab_size)

    def _validate_against_manifest(
        self,
        array: np.ndarray,
        key: str,
    ) -> None:
        """Verify an array's shape and dtype against the manifest entry.

        Args:
            array: The ndarray to check.
            key:   The key in ``manifest.json`` (for example, ``"static_vectors"``).

        Raises:
            ManifestViolationError: If shape or dtype does not match the manifest.
        """
        if key not in self.manifest:
            logger.warning("manifest に '%s' キーが存在しません。スキップします", key)
            return

        spec = self.manifest[key]

        # shape check
        expected_shape: tuple[int, ...] = tuple(spec["shape"])
        if tuple(array.shape) != expected_shape:
            raise ManifestViolationError(
                f"{key}: shape 不一致。"
                f"manifest={expected_shape}, 実データ={tuple(array.shape)}"
            )

        # dtype check
        expected_dtype: str = spec["dtype"]
        if array.dtype != np.dtype(expected_dtype):
            raise ManifestViolationError(
                f"{key}: dtype 不一致。"
                f"manifest={expected_dtype}, 実データ={array.dtype}"
            )

        logger.debug(
            "manifest 照合通過: %s shape=%s dtype=%s",
            key, array.shape, array.dtype,
        )
