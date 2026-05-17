"""
test_embedding_loader.py
========================
Unit tests for ``EmbeddingLoader``.

Targets under test:
    - ``load_all()``: happy path; loads each file and runs alignment checks
    - ``ManifestViolationError``: shape / dtype mismatch
    - ``IndexAlignmentError``: N mismatch across files; invalid vocab indices
    - ``FileNotFoundError``: missing files

Test design:
    Does not depend on the real 83,823-word assets.
    Small mock data is built into a ``tempfile.TemporaryDirectory`` for each test.

How to run:
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
# Test helper: build a mock asset directory
# ---------------------------------------------------------------------------

def _build_mock_assets(
    tmpdir: Path,
    n: int = 5,
    static_dim: int = 4,
    contextual_dim: int = 8,
    seed: int = 42,
    static_shape_override: tuple | None = None,
    contextual_shape_override: tuple | None = None,
    vocab_override: dict | None = None,
    pos_n_override: int | None = None,
    dtype_override: str | None = None,
) -> None:
    """Create mock asset files inside ``tmpdir``.

    Args:
        tmpdir:                     Path to the temporary directory.
        n:                          Vocabulary size.
        static_dim:                 Dimensionality of the static vectors.
        contextual_dim:             Dimensionality of the contextual vectors (SBERT).
        seed:                       Random seed.
        static_shape_override:      Override ``static_vectors.shape`` in the manifest.
        contextual_shape_override:  Override ``contextual_vectors.shape`` in the manifest.
        vocab_override:             Override the contents of ``vocab.json``.
        pos_n_override:             Override the length of ``vocab_pos.npy``.
        dtype_override:             Override the dtype recorded in the manifest.
    """
    rng = np.random.default_rng(seed)

    emb_dir  = tmpdir / "embeddings"
    meta_dir = tmpdir / "metadata"
    emb_dir.mkdir()
    meta_dir.mkdir()

    static_vecs = rng.standard_normal((n, static_dim)).astype(np.float32)
    contextual_vecs  = rng.standard_normal((n, contextual_dim)).astype(np.float32)

    np.save(emb_dir / "static_vectors.npy", static_vecs)
    np.save(emb_dir / "contextual_vectors.npy",  contextual_vecs)

    vocab = vocab_override if vocab_override is not None else {
        f"word{i}": i for i in range(n)
    }
    with open(meta_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    pos_n = pos_n_override if pos_n_override is not None else n
    pos = np.array(["NOUN"] * pos_n)
    np.save(meta_dir / "vocab_pos.npy", pos)

    static_shape = list(static_shape_override) if static_shape_override else [n, static_dim]
    contextual_shape  = list(contextual_shape_override)  if contextual_shape_override  else [n, contextual_dim]
    dtype        = dtype_override if dtype_override else "float32"

    manifest = {
        "static_vectors": {"shape": static_shape, "dtype": dtype},
        "contextual_vectors":  {"shape": contextual_shape,  "dtype": dtype},
    }
    with open(tmpdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestLoadAllSuccess(unittest.TestCase):
    """Happy-path tests for ``EmbeddingLoader.load_all()``."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        _build_mock_assets(self.tmpdir, n=5, static_dim=4, contextual_dim=8)
        self.loader = EmbeddingLoader(self.tmpdir)
        self.loader.load_all()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_static_vectors_shape(self) -> None:
        """``static_vectors.shape`` is (N, D)."""
        self.assertEqual(self.loader.static_vectors.shape, (5, 4))

    def test_contextual_vectors_shape(self) -> None:
        """``contextual_vectors.shape`` is (N, D)."""
        self.assertEqual(self.loader.contextual_vectors.shape, (5, 8))

    def test_vocab_is_dict(self) -> None:
        """``vocab`` is a dictionary."""
        self.assertIsInstance(self.loader.vocab, dict)

    def test_vocab_size(self) -> None:
        """``vocab`` size matches N."""
        self.assertEqual(len(self.loader.vocab), 5)

    def test_pos_shape(self) -> None:
        """``pos.shape`` is (N,)."""
        self.assertEqual(self.loader.pos.shape, (5,))

    def test_manifest_loaded(self) -> None:
        """``manifest`` is loaded as a dictionary."""
        self.assertIsInstance(self.loader.manifest, dict)
        self.assertIn("static_vectors", self.loader.manifest)

    def test_vocab_indices_are_contiguous(self) -> None:
        """``vocab`` indices form contiguous integers in [0, N-1]."""
        indices = set(self.loader.vocab.values())
        expected = set(range(len(self.loader.vocab)))
        self.assertEqual(indices, expected)

    def test_static_dtype(self) -> None:
        """``static_vectors.dtype`` is float32."""
        self.assertEqual(self.loader.static_vectors.dtype, np.float32)


class TestManifestViolation(unittest.TestCase):
    """Tests for ``ManifestViolationError``."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_shape_mismatch_static(self) -> None:
        """Mismatched static shape in the manifest raises ``ManifestViolationError``."""
        _build_mock_assets(
            self.tmpdir, n=5,
            static_shape_override=(99, 4),  # actual is (5, 4)
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(ManifestViolationError):
            loader.load_all()

    def test_shape_mismatch_contextual(self) -> None:
        """Mismatched contextual shape in the manifest raises ``ManifestViolationError``."""
        _build_mock_assets(
            self.tmpdir, n=5,
            contextual_shape_override=(5, 999),  # actual is (5, 8)
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(ManifestViolationError):
            loader.load_all()

    def test_dtype_mismatch(self) -> None:
        """Mismatched dtype in the manifest raises ``ManifestViolationError``."""
        _build_mock_assets(
            self.tmpdir, n=5,
            dtype_override="float64",  # actual data is float32
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(ManifestViolationError):
            loader.load_all()


class TestIndexAlignment(unittest.TestCase):
    """Tests for ``IndexAlignmentError``."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_pos_n_mismatch(self) -> None:
        """``vocab_pos.npy`` length mismatch raises ``IndexAlignmentError``."""
        _build_mock_assets(
            self.tmpdir, n=5,
            pos_n_override=3,  # vocab has 5 entries but pos has 3
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(IndexAlignmentError):
            loader.load_all()

    def test_vocab_index_not_contiguous(self) -> None:
        """Non-contiguous vocab indices raise ``IndexAlignmentError``."""
        bad_vocab = {"word0": 0, "word1": 1, "word2": 9}  # 9 is a gap
        _build_mock_assets(
            self.tmpdir, n=3,
            vocab_override=bad_vocab,
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(IndexAlignmentError):
            loader.load_all()

    def test_vocab_duplicate_index(self) -> None:
        """Duplicate vocab indices raise ``IndexAlignmentError``."""
        dup_vocab = {"word0": 0, "word1": 0, "word2": 2}  # 0 is duplicated
        _build_mock_assets(
            self.tmpdir, n=3,
            vocab_override=dup_vocab,
        )
        loader = EmbeddingLoader(self.tmpdir)
        with self.assertRaises(IndexAlignmentError):
            loader.load_all()


class TestFileNotFound(unittest.TestCase):
    """Tests for ``FileNotFoundError``."""

    def test_nonexistent_data_root(self) -> None:
        """A non-existent directory raises ``FileNotFoundError``."""
        with self.assertRaises(FileNotFoundError):
            EmbeddingLoader(Path("/nonexistent/path/data"))

    def test_missing_manifest(self) -> None:
        """A missing ``manifest.json`` raises ``FileNotFoundError``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "embeddings").mkdir()
            (tmpdir_path / "metadata").mkdir()
            # Do not create manifest.json
            loader = EmbeddingLoader(tmpdir_path)
            with self.assertRaises(FileNotFoundError):
                loader.load_all()

    def test_missing_static_vectors(self) -> None:
        """A missing ``static_vectors.npy`` raises ``FileNotFoundError``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            _build_mock_assets(tmpdir_path, n=3)
            (tmpdir_path / "embeddings" / "static_vectors.npy").unlink()
            loader = EmbeddingLoader(tmpdir_path)
            with self.assertRaises(FileNotFoundError):
                loader.load_all()


if __name__ == "__main__":
    unittest.main()
