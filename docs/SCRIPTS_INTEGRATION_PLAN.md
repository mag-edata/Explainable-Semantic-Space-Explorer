# tmp/ → scripts/ 組み込み計画

## ステータス

| 項目 | 内容 |
|---|---|
| 作成日 | 2026-03-03 |
| 現在の状態 | **PENDING** |
| ブロッカー | `HuggingFace オンラインダウンロード禁止` 制約の解除待ち |
| 解除後の作業 | 本計画に従い `scripts/` を実装・統合する |

> **注記（2026-03-03）:** `gen_wiki_vocab.py`（`datasets` 経由 Simple Wikipedia）および `export_sbert_vectors.py`（`all-MiniLM-L6-v2` モデル）はセットアップ時に HuggingFace へのネットワークアクセスが必要。CLAUDE.md の制約解除後に着手する。

---

## 目的

`tmp/` に散在している資産生成スクリプト群を、プロジェクト正規ディレクトリ `scripts/` として再配置・修正する。

---

## 1. 現状の問題点

| 問題 | 詳細 |
|---|---|
| パス設計の不整合 | `tmp/paths.py` は `src/utils/paths.py` 前提（`data/processed/` 参照）。現PJは `assets/` 構成 |
| インポートパスの不整合 | `from src.text.tokenizer import ...` 等、現PJに存在しないパス |
| 不要ファイル参照 | `STATIC_WORDS` / `SBERT_WORDS`（単語配列）を生成しているが、現PJは `vocab.json` で代替済み |
| `manifest.json` 未生成 | `EmbeddingLoader` が必須とする `manifest.json` を生成するスクリプトが存在しない |
| `models/` 配置未定義 | Word2Vec モデルファイルの置き場所が未定義 |

---

## 2. 新ディレクトリ構成

```
Explainable-Semantic-Space-Explorer/
├── scripts/                        ← 新規（資産生成スクリプト群）
│   ├── __init__.py
│   ├── paths.py                    ← 書き直し（assets/ 構成に合わせる）
│   ├── token_definition.py         ← ほぼそのまま（インポートのみ修正）
│   ├── tokenizer.py                ← ほぼそのまま（インポートのみ修正）
│   ├── gen_brown_vocab.py          ← インポートパス修正
│   ├── gen_wiki_vocab.py           ← インポートパス修正
│   ├── merge_vocab.py              ← インポートパス修正
│   ├── export_static_vectors.py    ← パス・出力修正（words.npy 廃止）
│   ├── export_sbert_vectors.py     ← パス・出力修正（words.npy 廃止）
│   ├── export_vocab_pos.py         ← インポートパス修正
│   └── gen_manifest.py             ← 新規作成（manifest.json 生成）
├── models/                         ← 新規（Word2Vec モデル配置場所）
│   └── .gitkeep
│   └── （w2v_brown10_simplewiki10_sg_300d_w5.model をここに置く）
└── ...
```

---

## 3. 各ファイルの変更内容

### scripts/paths.py（書き直し）

現PJの `assets/` 構成に合わせて全パスを再定義する。

```python
# 変更前（tmp/paths.py）
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # PROJECT_ROOT/src/utils/paths.py 前提
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VOCAB_JSON   = PROCESSED_DIR / "vocab.json"
STATIC_WORDS = PROCESSED_DIR / "static_words.npy"   # 不要
SBERT_WORDS  = PROCESSED_DIR / "sbert_words.npy"    # 不要

# 変更後（scripts/paths.py）
PROJECT_ROOT  = Path(__file__).resolve().parents[1]  # PROJECT_ROOT/scripts/paths.py 前提
ASSETS_DIR    = PROJECT_ROOT / "assets"
EMBEDDINGS_DIR = ASSETS_DIR / "embeddings"
METADATA_DIR   = ASSETS_DIR / "metadata"
MODELS_DIR     = PROJECT_ROOT / "models"

VOCAB_JSON        = METADATA_DIR / "vocab.json"
STATIC_VECTORS    = EMBEDDINGS_DIR / "static_vectors.npy"
SBERT_VECTORS     = EMBEDDINGS_DIR / "sbert_vectors.npy"
VOCAB_POS         = METADATA_DIR / "vocab_pos.npy"
MANIFEST_JSON     = ASSETS_DIR / "manifest.json"
W2V_MODEL         = MODELS_DIR / "w2v_brown10_simplewiki10_sg_300d_w5.model"
```

`STATIC_WORDS` / `SBERT_WORDS` は廃止（`vocab.json` で代替）。

---

### scripts/token_definition.py（変更なし）

インポート不要のため変更なし。

---

### scripts/tokenizer.py（インポート修正のみ）

```python
# 変更前
from .token_definition import TOKEN_EXTRACT_PATTERN, TOKEN_CONSTRAINT_PATTERN

# 変更後（同じ相対インポートでOK。scripts パッケージ内として動作）
from .token_definition import TOKEN_EXTRACT_PATTERN, TOKEN_CONSTRAINT_PATTERN
```

実質変更なし（`__init__.py` の追加で解決）。

---

### scripts/gen_brown_vocab.py（インポート修正）

```python
# 変更前
from src.text.tokenizer import normalize_tokens

# 変更後
from scripts.tokenizer import normalize_tokens
```

---

### scripts/gen_wiki_vocab.py（インポート修正）

```python
# 変更前
from src.text.tokenizer import tokenize_text

# 変更後
from scripts.tokenizer import tokenize_text
```

---

### scripts/merge_vocab.py（インポート修正）

```python
# 変更前
from src.vocab_generator.gen_brown_vocab import gen_brown_vocab
from src.vocab_generator.gen_wiki_vocab import gen_wiki_vocab
from src.utils.paths import VOCAB_JSON

# 変更後
from scripts.gen_brown_vocab import gen_brown_vocab
from scripts.gen_wiki_vocab import gen_wiki_vocab
from scripts.paths import VOCAB_JSON
```

---

### scripts/export_static_vectors.py（パス・出力修正）

```python
# 変更前（インポート）
from src.utils.paths import VOCAB_JSON, W2V_MODEL, STATIC_WORDS, STATIC_VECTORS

# 変更後（インポート）
from scripts.paths import VOCAB_JSON, W2V_MODEL, STATIC_VECTORS
# STATIC_WORDS は廃止（vocab.json があるため不要）

# 変更前（保存）
np.save(STATIC_WORDS, np.array(valid_words))   # ← 廃止
np.save(STATIC_VECTORS, vectors_array)

# 変更後（保存）
np.save(STATIC_VECTORS, vectors_array)          # ← vectors のみ保存
```

---

### scripts/export_sbert_vectors.py（パス・出力修正）

```python
# 変更前
from src.utils.paths import VOCAB_JSON, SBERT_WORDS, SBERT_VECTORS

# 変更後
from scripts.paths import VOCAB_JSON, SBERT_VECTORS

# 変更前（保存）
np.save(SBERT_WORDS, np.array(vocab))    # ← 廃止
np.save(SBERT_VECTORS, vectors)

# 変更後（保存）
np.save(SBERT_VECTORS, vectors)          # ← vectors のみ保存
```

---

### scripts/export_vocab_pos.py（インポート修正）

```python
# 変更前
from src.utils.paths import VOCAB_JSON, PROCESSED_DIR, VOCAB_POS

# 変更後
from scripts.paths import VOCAB_JSON, VOCAB_POS
```

---

### scripts/gen_manifest.py（新規作成）

`EmbeddingLoader._validate_against_manifest()` が必要とする `manifest.json` を生成する。

```python
# 生成する manifest.json の構造（EmbeddingLoader の期待値に合わせる）
{
    "static_vectors": {"shape": [83823, 300], "dtype": "float32"},
    "sbert_vectors":  {"shape": [83823, 384], "dtype": "float32"},
    "vocab_pos":      {"shape": [83823],      "dtype": "<U9"}
}
```

生成ロジック: 各 `.npy` ファイルを読み込み、実際の shape / dtype を記録して書き出す。

---

## 4. 制約との整合チェック

| 制約 | スクリプトへの影響 |
|---|---|
| 外部 API 禁止 | 生成スクリプトは推論時に使用しない。セットアップ専用のため問題なし |
| HuggingFace オンラインダウンロード禁止 | `gen_wiki_vocab.py` は `datasets.load_dataset` を使用 → **初回のみネットワーク必要**。キャッシュ後はオフライン動作。要注記 |
| HuggingFace モデルダウンロード | `export_sbert_vectors.py` は `SentenceTransformer("all-MiniLM-L6-v2")` を使用 → **初回のみネットワーク必要**。キャッシュ後はオフライン動作。要注記 |
| コサイン類似度自前実装 | スクリプト群に類似度計算なし。問題なし |
| manifest.json 整合チェック | `gen_manifest.py` で生成後、`EmbeddingLoader` が自動チェック |

---

## 5. 実行順序（セットアップ手順）

```bash
# 0. NLTK データ（既存手順）
python -c "import nltk; nltk.download('brown')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

# 1. Word2Vec モデルを models/ に配置（手動）
#    → models/w2v_brown10_simplewiki10_sg_300d_w5.model

# 2. 統合語彙を生成（Brown + Simple Wikipedia）
python -m scripts.merge_vocab
#    → assets/metadata/vocab.json

# 3. 静的ベクトルを生成（Word2Vec）
python -m scripts.export_static_vectors
#    → assets/embeddings/static_vectors.npy

# 4. SBERT ベクトルを生成
python -m scripts.export_sbert_vectors
#    → assets/embeddings/sbert_vectors.npy

# 5. 品詞ラベルを生成
python -m scripts.export_vocab_pos
#    → assets/metadata/vocab_pos.npy

# 6. manifest.json を生成
python -m scripts.gen_manifest
#    → assets/manifest.json

# 7. アプリ起動
venv/bin/python3 -m streamlit run ui/app.py
```

---

## 6. .gitignore 追加項目

```gitignore
# Word2Vec モデル（大容量）
models/*.model
models/*.model.syn1neg.npy
models/*.model.wv.vectors.npy

# HuggingFace キャッシュ（任意）
# ~/.cache/huggingface/
```

---

## 7. ドキュメント更新箇所

| ファイル | 更新内容 |
|---|---|
| `README.md` | セットアップ手順に `scripts/` の実行手順を追加 |
| `CLAUDE.md` | ディレクトリ構造に `scripts/` を追加。セットアップ手順を更新 |
| `docs/ARCHITECTURE_SPEC.md` | Section 3（Data Assets）にスクリプト層の説明を追加 |

---

## 8. tmp/ の扱い

`tmp/` は組み込み完了後に削除推奨。
削除は人間が実行すること（`rm -rf tmp/` または Git 管理から除外）。

---

## 9. 作業ファイル一覧（新規作成・編集対象）

| 操作 | ファイル |
|---|---|
| 新規作成 | `scripts/__init__.py` |
| 新規作成（書き直し） | `scripts/paths.py` |
| 新規作成（移植） | `scripts/token_definition.py` |
| 新規作成（移植） | `scripts/tokenizer.py` |
| 新規作成（修正） | `scripts/gen_brown_vocab.py` |
| 新規作成（修正） | `scripts/gen_wiki_vocab.py` |
| 新規作成（修正） | `scripts/merge_vocab.py` |
| 新規作成（修正） | `scripts/export_static_vectors.py` |
| 新規作成（修正） | `scripts/export_sbert_vectors.py` |
| 新規作成（修正） | `scripts/export_vocab_pos.py` |
| 新規作成 | `scripts/gen_manifest.py` |
| 新規作成 | `models/.gitkeep` |
| 編集 | `.gitignore` |
| 編集 | `README.md` |
| 編集 | `CLAUDE.md` |
