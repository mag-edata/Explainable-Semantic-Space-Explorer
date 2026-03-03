# CLAUDE.md — Explainable Semantic Space Explorer

## プロジェクト概要

単語埋め込み空間を可視化・分析し、「なぜその単語が近いのか」を説明可能にするツール。

- static 埋め込み（Word2Vec）と文脈埋め込み（SBERT）の差異を数値で説明する
- UIの美しさは目的外。**説明可能性・再現性が最優先**
- 詳細仕様: `docs/ARCHITECTURE_SPEC.md`

---

## 絶対制約（違反禁止）

1. 外部 API 禁止（OpenAI 等）
2. HuggingFace オンラインダウンロード禁止
3. **コサイン類似度は自前実装**（scipy / sklearn の `cosine_similarity` 使用禁止）
4. ローカル CPU 環境のみで動作保証
5. 推論時の学習処理禁止
6. 乱数使用時は seed 固定
7. manifest.json によるインデックス整合チェック必須

---

## 資産ファイルの生成前提（セットアップ）

`assets/` の `.npy` / `.json` は Git 管理外（大容量のため）。
別途以下を実行してから資産を生成・配置すること。

```bash
# NLTK データ（コーパス・品詞タガー）
python -c "import nltk; nltk.download('brown')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

---

## 資産ファイル（既存・変更禁止）

```
assets/
├── embeddings/
│   ├── static_vectors.npy    # [83823, 300]  Word2Vec  float32
│   └── sbert_vectors.npy     # [83823, 384]  SBERT     float32
├── metadata/
│   ├── vocab.json            # {"vocab": ["word0", ...]}  ← リスト形式（ローダーが Dict[str,int] に変換）
│   └── vocab_pos.npy         # [83823]  品詞ラベル配列  ← ファイル名注意（pos.npy ではない）
└── manifest.json             # shape / dtype の期待値
```

---

## アーキテクチャ（3層・依存方向厳守）

```
assets/ ──► core/ ──► analysis/ ──► ui/app.py
```

| 層 | ディレクトリ | ルール |
|---|---|---|
| データ | `assets/` | 読み取り専用 |
| ロジック | `core/` | Streamlit 禁止、外部 API 禁止 |
| 前処理 | `analysis/` | core に依存可、ui に依存禁止 |
| UI | `ui/` | core / analysis を呼ぶだけ。ロジック記述禁止 |

---

## コーディング規約

- **クラス設計優先**（関数型より）
- 型ヒント必須（全引数・戻り値）
- Docstring 必須（全クラス・全メソッド）
- ログ出力実装（`logging` モジュール）
- 例外処理明示（カスタム例外クラスを定義して使う）

---

## ディレクトリ構造（確定版）

```
Explainable-Semantic-Space-Explorer/
├── assets/                      （上記参照）
├── core/
│   ├── __init__.py              ✅ 完成
│   ├── embedding_loader.py      ✅ 完成
│   ├── similarity_engine.py     ✅ 完成
│   ├── distance_metrics.py      ✅ 完成
│   ├── pos_filter.py            ✅ 完成
│   └── analyzer.py              ✅ 完成
├── analysis/
│   ├── __init__.py              ✅ 完成
│   ├── projection.py            ✅ 完成（PCA / UMAP）
│   └── cluster.py               ✅ 完成（コサイン KMeans）
├── ui/
│   └── app.py                   ✅ 完成（Streamlit 4タブ）
├── tests/
│   ├── __init__.py              ✅ 完成
│   ├── test_distance_metrics.py ✅ 完成（29テスト）
│   ├── test_similarity_engine.py✅ 完成（26テスト）
│   └── test_embedding_loader.py ✅ 完成（16テスト）
├── docs/
│   └── ARCHITECTURE_SPEC.md     ✅ 既存
├── CLAUDE.md                    ✅ 本ファイル
└── requirements.txt             ✅ 作成済み
```

---

## 実装済みファイルの設計

### core/embedding_loader.py

```
EmbeddingLoaderError          # 基底例外
IndexAlignmentError           # N（語彙数）不一致
ManifestViolationError        # shape/dtype が manifest と乖離

EmbeddingLoader(asset_root: Path)
  load_all()                  # 唯一の公開 API。以下を順番に呼ぶ
  _load_manifest()
  _load_embeddings()
  _load_metadata()
  _validate()                 # N一致 + manifest照合 + vocab index連続性チェック
  _validate_against_manifest(array, key)
```

**インスタンス変数（load_all() 後に参照可能）:**

| 変数 | 型 | 内容 |
|---|---|---|
| `static_vectors` | `np.ndarray` | shape (83823, 300) |
| `sbert_vectors` | `np.ndarray` | shape (83823, 384) |
| `vocab` | `Dict[str, int]` | `{"word": index}`（ローダーがリスト形式から変換） |
| `pos` | `np.ndarray` | shape (83823,) |
| `manifest` | `dict` | manifest.json の中身 |

---

### core/similarity_engine.py

**設計方針:** 1インスタンス = 1モデル。依存注入方式。

```
SimilarityEngineError / UnknownWordError / InvalidTopKError

SearchResult (dataclass)
  word, index, similarity, rank, pos_tag, pos_rank, explanation

ComparisonResult (dataclass)
  query_word, static_results, sbert_results
  common_words, static_only, sbert_only
  rank_diff, similarity_diff

SimilarityEngine(vectors, vocab, pos_tags, metrics)
  search(query, top_k=10, pos_filter=None) → list[SearchResult]
  compare(query, other, top_k=10)          → ComparisonResult
  get_distance_distribution(query)         → dict
  word_to_index(word)                      → int
  # private:
  _build_results(query_vec, query_idx, top_k)
  _search_single(query_vec, query_idx, top_k)  # argpartition で O(N+k log k)
  _assign_pos_ranks(results)
  _build_explanation(query_vec, target_vec)
  _validate_top_k(top_k)
```

**標準的な呼び出しパターン:**

```python
loader = EmbeddingLoader(Path("assets"))
loader.load_all()
metrics = DistanceMetrics()

static_engine = SimilarityEngine(
    vectors=loader.static_vectors,
    vocab=loader.vocab,
    pos_tags=loader.pos,
    metrics=metrics,
)
sbert_engine = SimilarityEngine(
    vectors=loader.sbert_vectors,
    vocab=loader.vocab,
    pos_tags=loader.pos,
    metrics=metrics,
)

results    = static_engine.search("king", top_k=10)
comparison = static_engine.compare("king", other=sbert_engine, top_k=10)
dist       = static_engine.get_distance_distribution("king")
```

---

## 実装済み analysis 層の設計

### analysis/cluster.py

```
ClusterError / NotFittedError / InvalidClusterCountError / UnfitVectorError

ClusterResult (dataclass)
  labels: np.ndarray  # shape (N,)  クラスタID (0始まり)
  n_clusters, inertia, seed, n_samples

KMeansClusterer(n_clusters=8, seed=42, max_iter=300)
  fit(vectors) → ClusterResult     # L2 正規化 → sklearn KMeans
  get_labels() → np.ndarray        # NotFittedError ガード
  get_result() → ClusterResult     # NotFittedError ガード
  # private:
  _l2_norm_batch(matrix)           # 自前ノルム: sqrt(sum(v²))
  _normalize_rows(matrix)          # 自前正規化: â = a / ‖a‖
  _validate_inputs(vectors)
```

**設計原則:** ‖â - b̂‖² = 2 - 2·cos(a,b) によりコサイン距離とユークリッド距離が等価

---

## 実装済み analysis 層の設計（続き）

### analysis/projection.py

```
ProjectionError / NotFittedError / InvalidMethodError / InvalidVectorError

ProjectionResult (dataclass)
  coords_2d: np.ndarray           # shape (N, 2)  2D 座標
  explained_variance: list[float] # 主成分寄与率（PCA のみ。UMAP は []）
  method: str                     # "pca" or "umap"
  cluster_labels: np.ndarray | None  # shape (N,)  クラスタID（付与時）
  n_samples: int
  seed: int

Projector(method="pca", seed=42)
  fit_transform(vectors) → ProjectionResult
  attach_clusters(result, cluster_labels) → ProjectionResult  # イミュータブル操作
  # private:
  _fit_pca(vectors)    # sklearn PCA、寄与率付き
  _fit_umap(vectors)   # umap.UMAP、explained_variance=[]
  _validate_inputs(vectors)
```

**数式（PCA）:** Z = X·V^T　寄与率_i = λ_i / Σλ_j

---

## get_distance_distribution の返り値仕様

```python
{
    "query_word":      str,
    "mean":            float,   # 全語彙との平均コサイン類似度
    "std":             float,   # 標準偏差
    "top1_similarity": float,   # Top-1 スコア
    "z_score":         float,   # (top1 - mean) / std
    "histogram_data":  list[float],  # 全 N-1 件（可視化用生データ）
}
```

---

## 進捗状況

```
core 層     ██████  5/5 ファイル完成（100%）
analysis 層 ██████  2/2 完成（100%）
ui 層       ██████  1/1 完成（100%）
tests/      ██████  3/3 完成（100%）  ← 71テスト全通過
全体        ██████████████████  100%
```

**テスト実行コマンド:**
```bash
venv/bin/python3 -m unittest discover tests/ -v
# → 71 tests in 0.063s  OK
```

---

## 既知の課題（TODO）

### [UI] 投影・クラスタタブ: クエリ語マーカーの凡例と図の不一致

- **ファイル**: `ui/app.py`（投影・クラスタタブの散布図）
- **現象**: `alt.Shape` + `alt.Scale(domain, range)` でクエリ語を `"cross"` にしているが、凡例と図上マーカーが一致しない
- **試みた失敗策**:
  - `"star"` → Vega-Lite の無効値、凡例が消える
  - チャートをレイヤー分割 → 描画自体が消える
- **現在**: `range=["cross", "circle"]` で保留
- **次の候補**:
  - `shape` エンコーディングを廃止し `size` + `color` でクエリを強調
  - Vega-Lite SVG パス文字列を `range` に直接渡す
