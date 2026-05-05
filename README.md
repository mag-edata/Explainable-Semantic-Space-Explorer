# Explainable Semantic Space Explorer

単語埋め込み空間を可視化・分析し、**「なぜその単語が近いのか」を数値で説明する**ツール。

- 静的埋め込み（Word2Vec）と文脈埋め込み（SBERT）の差異を定量的に比較
- コサイン類似度の計算式を UI 上に明示（ブラックボックスにしない）
- 品詞・クラスタ・Z-score など多角的な視点で「近さの理由」を説明
- 外部 API 不使用・ローカル CPU のみで完全動作

---

## 動作環境

| 項目 | バージョン |
|---|---|
| Python | 3.12 |
| numpy | 2.4.2 |
| scikit-learn | 1.8.0 |
| umap-learn | 0.5.11 |
| streamlit | 1.54.0 |
| altair | 6.0.0 |

---

## セットアップ

```bash
# 仮想環境の作成・有効化
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt

# NLTK データのダウンロード（資産ファイル生成に必要）
python -c "import nltk; nltk.download('brown')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

---

## 起動方法

```bash
# Streamlit アプリを起動
venv/bin/python3 -m streamlit run ui/app.py
# → http://localhost:8501 がブラウザで開く
```

---

## プロジェクト構造

```
Explainable-Semantic-Space-Explorer/
├── assets/                        # 埋め込みベクトル・メタデータ（変更禁止）
│   ├── embeddings/
│   │   ├── static_vectors.npy     # Word2Vec  shape (83823, 300)
│   │   └── sbert_vectors.npy      # SBERT     shape (83823, 384)
│   ├── metadata/
│   │   ├── vocab.json             # {"vocab": ["word0", ...]}  ← リスト形式（ローダーが辞書に変換）
│   │   └── vocab_pos.npy          # 品詞ラベル配列 shape (83823,)
│   └── manifest.json              # shape / dtype の整合チェック用
│
├── core/                          # 純粋ロジック層（Streamlit 禁止）
│   ├── embedding_loader.py        # ベクトル読み込み・整合チェック
│   ├── similarity_engine.py       # 類似度検索・比較
│   ├── distance_metrics.py        # コサイン類似度の自前実装
│   ├── pos_filter.py              # 品詞フィルタリング
│   └── analyzer.py                # 距離分布の統計分析
│
├── analysis/                      # 可視化前処理層
│   ├── cluster.py                 # KMeans クラスタリング（コサイン距離）
│   └── projection.py              # PCA / UMAP による 2D 投影
│
├── ui/
│   └── app.py                     # Streamlit UI（4タブ）
│
├── tests/                         # 単体テスト群（71テスト）
│   ├── test_distance_metrics.py   # DistanceMetrics の検証（29テスト）
│   ├── test_similarity_engine.py  # SimilarityEngine の検証（26テスト）
│   └── test_embedding_loader.py   # EmbeddingLoader の検証（16テスト）
├── DOCS/
│   └── ARCHITECTURE_SPEC.md       # 詳細アーキテクチャ仕様
├── requirements.txt
└── README.md
```

---

## アーキテクチャ概要

依存方向は一方向のみ。逆流禁止。

```
assets/ ──► core/ ──► analysis/ ──► ui/app.py
```

| 層 | 役割 |
|---|---|
| `assets/` | 埋め込みデータ（読み取り専用） |
| `core/` | 距離計算・検索・統計など純粋ロジック |
| `analysis/` | PCA・UMAP・KMeans など可視化前処理 |
| `ui/` | Streamlit で表示するだけ。ロジック記述禁止 |

---

## 主要機能（実装済み）

### 類似度検索

```python
from pathlib import Path
from core.embedding_loader import EmbeddingLoader
from core.similarity_engine import SimilarityEngine
from core.distance_metrics import DistanceMetrics

loader = EmbeddingLoader(Path("assets"))
loader.load_all()
metrics = DistanceMetrics()

engine = SimilarityEngine(
    vectors=loader.static_vectors,
    vocab=loader.vocab,
    pos_tags=loader.pos,
    metrics=metrics,
)

# Top-10 類似語を検索
results = engine.search("king", top_k=10)
for r in results:
    print(f"{r.rank}位: {r.word}  similarity={r.similarity:.4f}  ({r.pos_tag})")
```

### static vs SBERT 比較

```python
sbert_engine = SimilarityEngine(
    vectors=loader.sbert_vectors,
    vocab=loader.vocab,
    pos_tags=loader.pos,
    metrics=metrics,
)

comparison = engine.compare("king", other=sbert_engine, top_k=10)
print("共通語:", comparison.common_words)
print("static 固有:", comparison.static_only)
print("SBERT 固有:", comparison.sbert_only)
```

### 距離分布 + Z-score

```python
dist = engine.get_distance_distribution("king")
print(f"平均類似度: {dist['mean']:.4f}")
print(f"標準偏差:   {dist['std']:.4f}")
print(f"Top-1:      {dist['top1_similarity']:.4f}")
print(f"Z-score:    {dist['z_score']:.4f}")
```

### クラスタリング（コサイン KMeans）

```python
from analysis.cluster import KMeansClusterer

clusterer = KMeansClusterer(n_clusters=8, seed=42)
result = clusterer.fit(loader.static_vectors)

print(f"クラスタ数: {result.n_clusters}")
print(f"inertia:   {result.inertia:.4f}")
print(f"単語 'king' のクラスタID: {result.labels[loader.vocab['king']]}")
```

---

## 説明可能性の設計方針

各検索結果には計算の内訳が付属する。

```python
result = engine.search("king", top_k=1)[0]
exp = result.explanation
print(f"内積:       {exp['dot_product']:.4f}")
print(f"|query|:    {exp['norm_a']:.4f}")
print(f"|target|:   {exp['norm_b']:.4f}")
print(f"コサイン:   {exp['similarity']:.4f}")
print(f"計算式:     {exp['formula']}")
# → "dot(a,b) / (‖a‖·‖b‖) = 12.34 / (5.00·3.00) = 0.8227"
```

---

## テストの実行

Python 標準の `unittest` を使用。pytest 不要。

```bash
# 全テストを実行
venv/bin/python3 -m unittest discover tests/ -v

# 個別ファイルを実行
venv/bin/python3 -m unittest tests/test_distance_metrics.py -v
venv/bin/python3 -m unittest tests/test_similarity_engine.py -v
venv/bin/python3 -m unittest tests/test_embedding_loader.py -v
```

期待される出力:
```
Ran 71 tests in 0.063s
OK
```

---

## 制約

| 制約 | 理由 |
|---|---|
| 外部 API 禁止（OpenAI 等） | ローカル完結・再現性保証 |
| HuggingFace オンラインダウンロード禁止 | オフライン動作 |
| コサイン類似度は自前実装 | 計算過程の透明性 |
| 乱数 seed 固定（42） | 完全再現性 |
| 推論時の学習禁止 | CPU 環境での速度保証 |

---

## 実装状況

| コンポーネント | 状態 | 備考 |
|---|---|---|
| `core/` 全ファイル | ✅ 完成 | 5ファイル |
| `analysis/cluster.py` | ✅ 完成 | コサイン KMeans |
| `analysis/projection.py` | ✅ 完成 | PCA / UMAP |
| `ui/app.py` | ✅ 完成 | Streamlit 4タブ |
| `tests/` | ✅ 完成 | 71テスト全通過 |

---

## 既知の課題

### [UI] 投影・クラスタタブ: クエリ語マーカーの凡例と図の不一致

- **場所**: `ui/app.py` — 投影・クラスタタブの散布図
- **現象**: Altair の `shape` エンコーディング（`alt.Shape` + `alt.Scale(domain, range)`）でクエリ語に `"cross"` を指定しているが、凡例の表示と図上のマーカー形状が一致しないことがある
- **試みた対策**:
  - `range=["star", "circle"]` → `"star"` は Vega-Lite の無効値のため凡例が消滅
  - チャートをレイヤー分割（近傍語・クエリ語を別 `alt.Chart`）→ 描画自体が消えた
- **現在の状態**: `range=["cross", "circle"]` に戻して保留中
- **候補解決策**:
  - Altair の `mark_rule` / `mark_point` を組み合わせてクエリ点を別途オーバーレイする
  - `shape` エンコーディングを廃止し、`size` と `color` だけでクエリを目立たせる
  - Vega-Lite の SVG パス文字列を `range` に直接渡す
