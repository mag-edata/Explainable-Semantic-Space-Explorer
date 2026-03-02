"""
embedding_loader.py
===================
埋め込み資産の読み込みとインデックス整合チェック。

assets/ ディレクトリ配下の全ファイルを読み込み、
manifest.json に記載された期待値と実データを照合することで
インデックス不整合を起動時に検出する。

読み込んだデータは EmbeddingLoader のインスタンス変数として保持する。
SimilarityEngine への受け渡しは呼び出し側で行う（依存注入）。

ディレクトリ構造:
    assets/
    ├── embeddings/
    │   ├── static_vectors.npy   # [N, 300] Word2Vec float32
    │   └── sbert_vectors.npy    # [N, 768] SBERT float32
    ├── metadata/
    │   ├── vocab.json           # {"word": index, ...}
    │   └── vocab_pos.npy        # [N] 品詞ラベル配列
    └── manifest.json            # shape / dtype の期待値
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# カスタム例外
# ---------------------------------------------------------------------------

class EmbeddingLoaderError(Exception):
    """EmbeddingLoader 固有の例外基底クラス。"""


class IndexAlignmentError(EmbeddingLoaderError):
    """ファイル間のインデックス数（N）が一致しない場合の例外。

    例: static_vectors.shape[0]=50000 に対し vocab の件数が 49999 だった場合。
    """


class ManifestViolationError(EmbeddingLoaderError):
    """実データの shape / dtype が manifest.json の記載と一致しない場合の例外。

    例: manifest では shape=(50000,300) と記載されているが、
        実際にロードした配列が shape=(49999,300) だった場合。
    """


# ---------------------------------------------------------------------------
# EmbeddingLoader
# ---------------------------------------------------------------------------

class EmbeddingLoader:
    """埋め込み資産の読み込みとインデックス整合チェック。

    load_all() を呼ぶと全ファイルを読み込み、整合チェックを実行する。
    チェック通過後、各データをインスタンス変数として保持する。

    使用例::

        from pathlib import Path
        from core.embedding_loader import EmbeddingLoader
        from core.distance_metrics import DistanceMetrics
        from core.similarity_engine import SimilarityEngine

        loader = EmbeddingLoader(Path("assets"))
        loader.load_all()

        static_engine = SimilarityEngine(
            vectors=loader.static_vectors,
            vocab=loader.vocab,
            pos_tags=loader.pos,
            metrics=DistanceMetrics(),
        )

    Attributes:
        asset_root:      assets/ ディレクトリのパス
        emb_dir:         embeddings/ サブディレクトリのパス
        meta_dir:        metadata/ サブディレクトリのパス
        static_vectors:  Word2Vec 埋め込み行列 shape (N, 300)
        sbert_vectors:   SBERT 埋め込み行列 shape (N, 768)
        vocab:           単語 → インデックスの辞書 {"word": index}
        pos:             品詞ラベル配列 shape (N,)
        manifest:        manifest.json の内容
    """

    def __init__(self, asset_root: Path) -> None:
        """EmbeddingLoader を初期化する。

        Args:
            asset_root: assets/ ディレクトリへの Path オブジェクト。

        Raises:
            FileNotFoundError: asset_root が存在しないディレクトリの場合。
        """
        if not asset_root.exists():
            raise FileNotFoundError(
                f"assets ディレクトリが見つかりません: {asset_root}"
            )

        self.asset_root: Path = asset_root
        self.emb_dir: Path = asset_root / "embeddings"
        self.meta_dir: Path = asset_root / "metadata"

        # load_all() 呼び出し後に設定される
        self.static_vectors: np.ndarray | None = None
        self.sbert_vectors: np.ndarray | None = None
        self.vocab: Dict[str, int] | None = None
        self.pos: np.ndarray | None = None
        self.manifest: dict | None = None

        logger.info("EmbeddingLoader 初期化: asset_root=%s", asset_root)

    def load_all(self) -> None:
        """全資産ファイルを読み込み、インデックス整合チェックを実行する。

        実行順序:
        1. manifest.json を読み込む
        2. 埋め込みベクトル (.npy) を読み込む
        3. メタデータ (vocab.json, vocab_pos.npy) を読み込む
        4. インデックス整合チェックを実行する

        Raises:
            FileNotFoundError:    必要なファイルが存在しない場合。
            IndexAlignmentError:  N（語彙数）がファイル間で一致しない場合。
            ManifestViolationError: 実データが manifest の記載と異なる場合。
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
    # Private: 読み込み
    # -----------------------------------------------------------------------

    def _load_manifest(self) -> None:
        """manifest.json を読み込む。

        Raises:
            FileNotFoundError: manifest.json が存在しない場合。
        """
        manifest_path = self.asset_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json が見つかりません: {manifest_path}"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        logger.debug("manifest.json 読み込み完了: %s", manifest_path)

    def _load_embeddings(self) -> None:
        """埋め込みベクトルファイル (.npy) を読み込む。

        Raises:
            FileNotFoundError: 埋め込みファイルが存在しない場合。
        """
        static_path = self.emb_dir / "static_vectors.npy"
        sbert_path = self.emb_dir / "sbert_vectors.npy"

        for path in (static_path, sbert_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"埋め込みファイルが見つかりません: {path}"
                )

        self.static_vectors = np.load(static_path)
        self.sbert_vectors = np.load(sbert_path)

        logger.debug(
            "埋め込み読み込み完了: static=%s, sbert=%s",
            self.static_vectors.shape,
            self.sbert_vectors.shape,
        )

    def _load_metadata(self) -> None:
        """vocab.json と vocab_pos.npy を読み込む。

        vocab.json の形式: {"word": index, ...}
        インデックスは 0 始まりの整数。

        Raises:
            FileNotFoundError: メタデータファイルが存在しない場合。
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

        # vocab.json が {"vocab": [word0, word1, ...]} 形式の場合はリストを辞書に変換する
        # 期待する形式は {"word": index, ...} (Dict[str, int])
        if isinstance(raw, dict) and "vocab" in raw and isinstance(raw["vocab"], list):
            word_list = raw["vocab"]
            self.vocab = {word: idx for idx, word in enumerate(word_list)}
            logger.debug("vocab.json をリスト形式から辞書形式に変換しました (件数=%d)", len(self.vocab))
        else:
            self.vocab = raw

        self.pos = np.load(pos_path)

        logger.debug(
            "メタデータ読み込み完了: vocab_size=%d, pos_shape=%s",
            len(self.vocab),
            self.pos.shape,
        )

    # -----------------------------------------------------------------------
    # Private: 整合チェック
    # -----------------------------------------------------------------------

    def _validate(self) -> None:
        """全ファイル間のインデックス整合チェックを実行する。

        チェック順序:
        1. 未ロード確認（load_all の順序が正しいか）
        2. N（語彙数）の一致確認（4ファイル間）
        3. manifest.json の shape / dtype との照合
        4. vocab のインデックス値が [0, N-1] の連続整数であるか確認

        Raises:
            IndexAlignmentError:    N が一致しない、またはインデックス値が不正。
            ManifestViolationError: shape / dtype が manifest と一致しない。
        """
        # 1. 未ロード確認
        if any(x is None for x in (
            self.static_vectors, self.sbert_vectors, self.vocab, self.pos
        )):
            raise EmbeddingLoaderError(
                "load_all() を実行する前に _validate() が呼ばれました"
            )

        vocab_size: int = len(self.vocab)

        # 2. N（語彙数）の一致確認
        checks: List[tuple[int, str]] = [
            (self.static_vectors.shape[0], "static_vectors"),
            (self.sbert_vectors.shape[0],  "sbert_vectors"),
            (self.pos.shape[0],            "vocab_pos"),
        ]
        for actual_n, name in checks:
            if actual_n != vocab_size:
                raise IndexAlignmentError(
                    f"インデックス不整合: {name} の行数={actual_n} に対し "
                    f"vocab の件数={vocab_size} が一致しません"
                )

        # 3. manifest.json との shape / dtype 照合
        self._validate_against_manifest(self.static_vectors, "static_vectors")
        self._validate_against_manifest(self.sbert_vectors,  "sbert_vectors")

        # 4. vocab のインデックス値が 0..N-1 の連続整数であるか確認
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
        """実データの shape と dtype を manifest.json の記載と照合する。

        Args:
            array: チェック対象の ndarray。
            key:   manifest.json 内のキー名（"static_vectors" など）。

        Raises:
            ManifestViolationError: shape または dtype が manifest と一致しない場合。
        """
        if key not in self.manifest:
            logger.warning("manifest に '%s' キーが存在しません。スキップします", key)
            return

        spec = self.manifest[key]

        # shape チェック
        expected_shape: tuple[int, ...] = tuple(spec["shape"])
        if tuple(array.shape) != expected_shape:
            raise ManifestViolationError(
                f"{key}: shape 不一致。"
                f"manifest={expected_shape}, 実データ={tuple(array.shape)}"
            )

        # dtype チェック
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
