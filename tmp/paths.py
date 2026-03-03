"""
paths.py

本モジュールは、プロジェクト全体で使用するパス定義する。

目的は、パス定義を単一箇所に固定することであり、
ディレクトリ構成変更時の修正箇所を最小化し、
各モジュールにおけるパス参照の一貫性と保守性を保証する。

ディレクトリ構成を変更する場合は、本ファイルのみを修正する設計とする。
"""

from pathlib import Path

# ============================================================
# Project root directory
# ============================================================

# PROJECT_ROOT/src/utils/paths.py -> PROJECT_ROOT/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ============================================================
# Base directories
# ============================================================

# データ関連ディレクトリ
DATA_DIR: Path = PROJECT_ROOT / "data"

# 前処理済みデータを格納するディレクトリ
PROCESSED_DIR: Path = DATA_DIR / "processed"

# 学習済みモデルを格納するディレクトリ
MODELS_DIR: Path = PROJECT_ROOT / "models"

# ============================================================
# Data files
# ============================================================

# 語彙情報(JSON)
VOCAB_JSON: Path = PROCESSED_DIR / "vocab.json"

# 単語配列(静的埋め込み)
STATIC_WORDS: Path = PROCESSED_DIR / "static_words.npy"

# 単語ベクトル(静的埋め込み)
STATIC_VECTORS: Path = PROCESSED_DIR / "static_vectors.npy"

# 単語配列(SBERT)
SBERT_WORDS: Path = PROCESSED_DIR / "sbert_words.npy"

# 単語ベクトル(SBERT)
SBERT_VECTORS: Path = PROCESSED_DIR / "sbert_vectors.npy"

# 語彙POSラベル(numpy array)
VOCAB_POS: Path = PROCESSED_DIR / "vocab_pos.npy"

# ============================================================
# Model files
# ============================================================

# 学習済みWord2Vecモデル
W2V_MODEL = MODELS_DIR / "w2v_brown10_simplewiki10_sg_300d_w5.model"

# ============================================================
# ディレクトリ補完
# ============================================================

for p in [DATA_DIR, PROCESSED_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)