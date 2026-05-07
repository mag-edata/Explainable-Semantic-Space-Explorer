# CLAUDE.md — Explainable Semantic Space Explorer

## 目的

本プロジェクトは以下を満たすための転職用プロダクトである。
- IT企業（NLPエンジニア / AI導入コンサル）への市場接触
- 「説明可能な NLP 設計」の意思決定能力の提示
- GitHub によるコード・ドキュメントの即時公開状態の構築

重要：本プロジェクトは静的・文脈埋め込みの比較分析ツールとして、技術的深度と設計能力を同時に示す。

---

## 戦略（最重要ルール）

### 実装完了・デプロイ待ち（現在フェーズ）

フェーズ1（実装）は完了済み。現在の最優先事項は Streamlit Cloud デプロイである。

**ブロッカー:** `data/` の埋め込みファイルが大容量のため、Git 管理・デプロイ手段を検討中。
解決策（Git LFS 等）確定後、即デプロイする。

### デプロイ後に改善する

開発順序は以下に固定する：
1. Streamlit Cloud デプロイ
2. 転職エージェント・企業へ送付
3. フィードバック取得
4. 改善

---

## ドキュメントの役割分離

README.md
- 外部公開用
- 日本語で記述
- 採用担当が読む前提

CLAUDE.md
- 内部意思決定用（本ファイル）
- 絶対制約・設計方針・実装仕様・進捗管理を記録
- Claude Code 向け

DOCS/
- 設計書群（要件定義書・基本設計書・詳細設計書・テスト設計書・テスト項目書）
- 内部仕様の詳細記録

---

## プロジェクト概要

単語埋め込み空間を可視化・分析し、「なぜその単語が近いのか」を説明可能にするツール。

- static 埋め込み（Word2Vec）と文脈埋め込み（SBERT）の差異を数値で説明する
- UIの美しさは目的外。**説明可能性・再現性が最優先**
- 詳細仕様: `DOCS/要件定義書.md`

---

## 絶対制約（違反禁止）

1. 外部 API 禁止（OpenAI 等）
2. HuggingFace オンラインダウンロード禁止（セットアップ時 data_pipeline/ 実行時のみ許可）
3. **コサイン類似度は自前実装**（scipy / sklearn の `cosine_similarity` 使用禁止）
4. ローカル CPU 環境のみで動作保証
5. 推論時の学習処理禁止
6. 乱数使用時は seed 固定
7. manifest.json によるインデックス整合チェック必須

---

## 資産ファイルの生成前提（セットアップ）

`data/` の `.npy` / `.json` は Git 管理外（大容量のため）。
別途以下を実行してから資産を生成・配置すること。

```bash
# NLTK データ（コーパス・品詞タガー）
python -c "import nltk; nltk.download('brown')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

---

## 資産ファイル（既存・変更禁止）

```
data/
├── embeddings/
│   ├── static_vectors.npy    # [83823, 300]  Word2Vec  float32
│   └── contextual_vectors.npy     # [83823, 384]  SBERT     float32
├── metadata/
│   ├── vocab.json            # {"vocab": ["word0", ...]}  ← リスト形式（ローダーが Dict[str,int] に変換）
│   └── vocab_pos.npy         # [83823]  品詞ラベル配列  ← ファイル名注意（pos.npy ではない）
└── manifest.json             # shape / dtype の期待値
```

---

## アーキテクチャ（3層・依存方向厳守）

```
data/ ──► core/ ──► transforms/ ──► ui/app.py
```

| 層 | ディレクトリ | ルール |
|---|---|---|
| データ | `data/` | 読み取り専用 |
| ロジック | `core/` | Streamlit 禁止、外部 API 禁止 |
| 変換 | `transforms/` | core に依存可、ui に依存禁止 |
| UI | `ui/` | core / transforms を呼ぶだけ。ロジック記述禁止 |

---

## 技術スタック

| 役割 | 技術 |
|---|---|
| 静的埋め込み | Word2Vec（学習済みモデル、ローカル配置） |
| 文脈埋め込み | SBERT（all-MiniLM-L6-v2、ローカル配置） |
| 次元削減 | PCA / UMAP |
| クラスタリング | KMeans（コサイン距離） |
| UI | Streamlit + Altair |
| 言語 | Python 3.12 |
| テスト | unittest（192テスト全通過） |

---

## コーディング規約

- **クラス設計優先**（関数型より）
- 型ヒント必須（全引数・戻り値）
- Docstring 必須（全クラス・全メソッド）
- ログ出力実装（`logging` モジュール）
- 例外処理明示（カスタム例外クラスを定義して使う）

---

## ディレクトリ構造（現在の実態）

凡例：✅ 完成　⚠ 保留

```
Explainable-Semantic-Space-Explorer/
├── data/                        （上記参照・Git管理外）
├── core/
│   ├── __init__.py              ✅ 完成
│   ├── embedding_loader.py      ✅ 完成
│   ├── similarity_engine.py     ✅ 完成
│   ├── distance_metrics.py      ✅ 完成
│   ├── pos_filter.py            ✅ 完成
│   └── analyzer.py              ✅ 完成
├── transforms/
│   ├── __init__.py              ✅ 完成
│   ├── projection.py            ✅ 完成（PCA / UMAP）
│   └── clustering.py            ✅ 完成（コサイン KMeans）
├── ui/
│   └── app.py                   ✅ 完成（Streamlit 4タブ）
├── tests/
│   ├── __init__.py              ✅ 完成
│   ├── test_distance_metrics.py ✅ 完成（29テスト）
│   ├── test_similarity_engine.py ✅ 完成（26テスト）
│   ├── test_embedding_loader.py ✅ 完成（16テスト）
│   ├── test_pos_filter.py       ✅ 完成（29テスト）
│   ├── test_analyzer.py         ✅ 完成（31テスト）
│   ├── test_clustering.py       ✅ 完成（29テスト）
│   └── test_projection.py       ✅ 完成（32テスト）
├── data_pipeline/               ✅ サブパッケージ化完了
│   ├── __init__.py
│   ├── manifest.py
│   ├── _common/
│   │   ├── __init__.py
│   │   ├── token_definition.py
│   │   └── tokenizer.py
│   ├── vocab/
│   │   ├── __init__.py
│   │   ├── gen_brown.py
│   │   ├── gen_wiki.py
│   │   └── merge.py
│   └── export/
│       ├── __init__.py
│       ├── static_vectors.py
│       ├── contextual_vectors.py
│       └── vocab_pos.py
├── models/                      ✅ 配置先のみ作成（モデル本体は Git 管理外）
│   └── .gitkeep
├── DOCS/
│   ├── 要件定義書.md
│   ├── 基本設計書.md
│   ├── 詳細設計書.md
│   ├── テスト設計書.md
│   ├── テスト項目書.md
│   └── DATA_PIPELINE_REFACTOR_PLAN.md
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

EmbeddingLoader(data_root: Path)
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
| `contextual_vectors` | `np.ndarray` | shape (83823, 384) |
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
  query_word, static_results, contextual_results
  common_words, static_only, contextual_only
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
loader = EmbeddingLoader(Path("data"))
loader.load_all()
metrics = DistanceMetrics()

static_engine = SimilarityEngine(
    vectors=loader.static_vectors,
    vocab=loader.vocab,
    pos_tags=loader.pos,
    metrics=metrics,
)
contextual_engine = SimilarityEngine(
    vectors=loader.contextual_vectors,
    vocab=loader.vocab,
    pos_tags=loader.pos,
    metrics=metrics,
)

results    = static_engine.search("king", top_k=10)
comparison = static_engine.compare("king", other=contextual_engine, top_k=10)
dist       = static_engine.get_distance_distribution("king")
```

---

### transforms/clustering.py

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

### transforms/projection.py

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

## 実装優先順位

最終更新：2026-05-05

### フェーズ1（実装完了）

| タスク | 状態 | 備考 |
|--------|------|------|
| core/ 全ファイル | ✅ 完了 | 5ファイル |
| transforms/clustering.py | ✅ 完了 | コサイン KMeans |
| transforms/projection.py | ✅ 完了 | PCA / UMAP |
| ui/app.py | ✅ 完了 | Streamlit 4タブ |
| tests/ | ✅ 完了 | 192テスト全通過 |
| GitHub 公開 | ✅ 完了 | |

→ フェーズ1 全タスク完了。実装完了ライン達成済み

### フェーズ2（デプロイ）

| 項目 | 状態 | 備考 |
|------|------|------|
| Streamlit Cloud デプロイ | ⬜ 未着手 | data/ 大容量問題（Git LFS 等の解決策を検討中） |

### フェーズ3（改善）

| 項目 | 状態 | 備考 |
|------|------|------|
| [UI] クエリ語マーカー凡例修正 | ⬜ 未着手 | Altair `shape` 問題（既知の課題参照） |
| データ拡張 | ⬜ 未着手 | 語彙数・コーパス追加 |

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

---

## 転職での活用方針

使い方
- GitHub リンク送付
- Streamlit デモ URL 添付（デプロイ後）
- 「設計意図」を口頭説明

強調ポイント
- 説明可能性の設計（コサイン類似度の内訳・Z-score の意味）
- 外部 API 不使用のローカル完結 NLP パイプライン
- 静的 vs 文脈埋め込みの差異を定量的に比較するアーキテクチャ
- 192テストによるコアロジックの品質担保（core 層 + analysis 層を網羅）
