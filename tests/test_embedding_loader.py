"""
test_embedding_loader.py
========================
EmbeddingLoader の単体テスト。

テスト対象:
    - load_all(): 正常系・各ファイルの読み込みと整合チェック
    - ManifestViolationError: shape / dtype 不一致
    - IndexAlignmentError: ファイル間の N 不一致・vocab インデックス不正
    - FileNotFoundError: ファイル欠落

テスト設計:
    実際の 83,823 語彙資産には依存しない。
    tempfile.TemporaryDirectory で小規模モックデータを生成してテストする。

実行方法:
    venv/bin/python3 -m unittest tests/test_embedding_loader.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embedding_loader import (
    EmbeddingLoader,
    EmbeddingLoaderError,
    IndexAlignmentError,
    ManifestViolationError,
)


# ---------------------------------------------------------------------------
# テスト用ヘルパー: モック資産ディレクトリを構築する
# ---------------------------------------------------------------------------

def _build_mock_assets(
    tmpdir: Path,
    n: int = 5,
    static_dim: int = 4,
    sbert_dim: int = 8,
    seed: int = 42,
    static_shape_override: tuple | None = None,
    sbert_shape_override: tuple | None = None,
    vocab_override: dict | None = None,
    pos_n_override: int | None = None,
    dtype_override: str | None = None,
) -> None:
    """モック資産ファイルを tmpdir に生成する。

    Args:
        tmpdir:               一時ディレクトリのパス
        n:                    語彙数
        static_dim:           static ベクトルの次元数
        sbert_dim:            sbert ベクトルの次元数
        seed:                 乱数シード
        static_shape_override: manifest の static_vectors.shape を上書き
        sbert_shape_override:  manifest の sbert_vectors.shape を上書き
        vocab_override:        vocab.json の内容を上書き
        pos_n_override:        vocab_pos.npy の長さを上書き
        dtype_override:        manifest の dtype を上書き
    """
    rng = np.random.default_rng(seed)

    emb_dir  = tmpdir / "embeddings"
    meta_dir = tmpdir / "metadata"
    emb_dir.mkdir()
    meta_dir.mkdir()

    static_vecs = rng.standard_normal((n, static_dim)).astype(np.float32)
    sbert_vecs  = rng.standard_normal((n, sbert_dim)).astype(np.float32)

    np.save(emb_dir / "static_vectors.npy", static_vecs)
    np.save(emb_dir / "sbert_vectors.npy",  sbert_vecs)

    vocab = vocab_override if vocab_override is not None else {
        f"word{i}": i for i in range(n)
    }
    with open(meta_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    pos_n = pos_n_override if pos_n_override is not None else n
    pos = np.array(["NOUN"] * pos_n)
    np.save(meta_dir / "vocab_pos.npy", pos)

    static_shape = list(static_shape_override) if static_shape_override else [n, static_dim]
    sbert_shape  = list(sbert_shape_override)  if sbert_shape_override  else [n, sbert_dim]
    dtype        = dtype_override if dtype_override else "float32"

    manifest = {
        "static_vectors": {"shape": static_shape, "dtype": dtype},
        "sbert_vectors":  {"shape": sbert_shape,  "dtype": dtype},
    }
    with open(tmpdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


# ---------------------------------------------------------------------------
# テストクラス
# ---------------------------------------------------------------------------

class TestLoadAllSuccess(unittest.TestCase):
    """EmbeddingLoader.load_all() の正常系テスト。"""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        _build_mock_assets(self.tmpdir, n=5, static_dim=4, sbert_dim=8)
        self.loader = EmbeddingLoader(self.tmpdir)
        self.loader.load_all()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_static_vectors_shape(self) -> None:
        """static_vectors の shape が (N, D) であることを確認。"""
        self.assertEqual(self.loader.static_vectors.shape, (5, 4))

    def test_sbert_vectors_shape(self) -> None:
        """sbert_vectors の shape が (N, D) であることを確認。"""
        self.assertEqual(self.loader.sbert_vectors.shape, (5, 8))

    def test_vocab_is_dict(self) -> None:
        """vocab が dict 型であることを確認。"""
        self.assertIsInstance(self.loader.vocab, dict)

    def test_vocab_size(self) -> None:
        """vocab の件数が N と一致することを確認。"""
        self.assertEqual(len(self.loader.vocab), 5)

    def test_pos_shape(self) -> None:
        """pos 配列の shape が (N,) であることを確認。"""
        self.assertEqual(self.loader.pos.shape, (5,))

    def test_manifest_loaded(self) -> None:
        """manifest が dict として読み込まれていることを確認。"""
        self.assertIsInstance(self.loader.manifest, dict)
        self.assertIn("static_vectors", self.loader.manifest)

    def test_vocab_indices_are_contiguous(self) -> None:
        """vocab のインデックス値が 0 から N-1 の連続整数であることを確認。"""
        indices = set(self.loader.vocab.values())
        expected = set(range(len(self.loader.vocab)))
        self.assertEqual(indices, expected)

    def test_static_dtype(self) -> None:
        """static_vectors の dtype が float32 であることを確認。"""
        self.assertEqual(self.loader.static_vectors.dtype, np.float32)


class TestManifestViolation(unittest.TestCase):
    """ManifestViolationError のテスト。"""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_shape_mismatch_static(self) -> None:
        """manifest の static shape が実データと異なる場合に ManifestViolationError が発生する。"""
        _build_mock_assets(
            self.tmpdir, n=5,
            static_shape_override=(99, 4),  # 実際は (5, 4)
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(ManifestViolationError):
            loader.load_all()

    def test_shape_mismatch_sbert(self) -> None:
        """manifest の sbert shape が実データと異なる場合に ManifestViolationError が発生する。"""
        _build_mock_assets(
            self.tmpdir, n=5,
            sbert_shape_override=(5, 999),  # 実際は (5, 8)
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(ManifestViolationError):
            loader.load_all()

    def test_dtype_mismatch(self) -> None:
        """manifest の dtype が実データと異なる場合に ManifestViolationError が発生する。"""
        _build_mock_assets(
            self.tmpdir, n=5,
            dtype_override="float64",  # 実データは float32
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(ManifestViolationError):
            loader.load_all()


class TestIndexAlignment(unittest.TestCase):
    """IndexAlignmentError のテスト。"""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_pos_n_mismatch(self) -> None:
        """vocab_pos.npy の長さが vocab と一致しない場合に IndexAlignmentError が発生する。"""
        _build_mock_assets(
            self.tmpdir, n=5,
            pos_n_override=3,  # vocab は 5 件だが pos は 3 件
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(IndexAlignmentError):
            loader.load_all()

    def test_vocab_index_not_contiguous(self) -> None:
        """vocab のインデックスが連続整数でない場合に IndexAlignmentError が発生する。"""
        bad_vocab = {"word0": 0, "word1": 1, "word2": 9}  # 9 は飛びインデックス
        _build_mock_assets(
            self.tmpdir, n=3,
            vocab_override=bad_vocab,
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(IndexAlignmentError):
            loader.load_all()

    def test_vocab_duplicate_index(self) -> None:
        """vocab にインデックスの重複がある場合に IndexAlignmentError が発生する。"""
        dup_vocab = {"word0": 0, "word1": 0, "word2": 2}  # 0 が重複
        _build_mock_assets(
            self.tmpdir, n=3,
            vocab_override=dup_vocab,
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(IndexAlignmentError):
            loader.load_all()


class TestFileNotFound(unittest.TestCase):
    """FileNotFoundError のテスト。"""

    def test_nonexistent_asset_root(self) -> None:
        """存在しないディレクトリを指定した場合に FileNotFoundError が発生する。"""
        with self.assertRaises(FileNotFoundError):
            EmbeddingLoader(Path("/nonexistent/path/assets"))

    def test_missing_manifest(self) -> None:
        """manifest.json が欠落している場合に FileNotFoundError が発生する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "embeddings").mkdir()
            (tmpdir_path / "metadata").mkdir()
            # manifest.json を作らない
            loader = EmbeddingLoader(tmpdir_path)
            with self.assertRaises(FileNotFoundError):
                loader.load_all()

    def test_missing_static_vectors(self) -> None:
        """static_vectors.npy が欠落している場合に FileNotFoundError が発生する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            _build_mock_assets(tmpdir_path, n=3)
            (tmpdir_path / "embeddings" / "static_vectors.npy").unlink()
            loader = EmbeddingLoader(tmpdir_path)
            with self.assertRaises(FileNotFoundError):
                loader.load_all()


if __name__ == "__main__":
    unittest.main()
