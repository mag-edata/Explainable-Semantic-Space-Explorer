# ARCHITECTURE_SPEC.md

# Project Name
Explainable Semantic Space Explorer

---

# 1. Objective（目的）

本プロジェクトの目的は、
単語埋め込み空間を可視化・分析し、
「なぜその単語が近いのか」を説明可能にすることである。

本プロジェクトは以下を証明するためのものである：

- 静的埋め込み（Word2Vec）と文脈埋め込み（SBERT）の差異理解
- 距離計算の構造理解
- 品詞の影響分析
- コーパス依存性の明示
- 説明可能なNLP設計能力

UIの美しさは目的ではない。
説明可能性と再現性が最優先である。

---

# 2. Constraints（制約条件）

1. 外部API禁止（OpenAI API等使用不可）
2. HuggingFaceオンラインダウンロード禁止
3. 事前保存済みベクトルのみ使用
4. ローカルCPU環境で動作保証
5. 推論時学習処理は禁止
6. 乱数使用時はseed固定
7. コサイン類似度は自前実装（ブラックボックス禁止）

---

# 3. Data Assets（既存資産）

- static_vectors.npy
- sbert_vectors.npy
- vocab.json
- pos.npy

すべてインデックス整合済である前提。

インデックス不整合を起こさない設計を必須とする。

---

# 4. System Architecture

## 4.1 Layer Separation

### core/
純粋ロジック層。Streamlit依存禁止。

- embedding_loader.py
- similarity_engine.py
- distance_metrics.py
- pos_filter.py
- analyzer.py

### analysis/
可視化前処理層。

- projection.py（PCA / UMAP）
- cluster.py（KMeans等）

### ui/
Streamlit専用。

- app.py

UIからcoreを直接改変しない。

---

# 5. Functional Requirements

## 5.1 Similarity Search

入力単語に対し：

- Top-K類似語（static）
- Top-K類似語（SBERT）
- 類似度スコア
- 順位
- 同品詞内順位
- 異品詞率

を出力する。

---

## 5.2 Static vs Contextual 差分分析

出力する情報：

- 順位差分
- 距離差分
- 共通語数
- 固有語数
- 差分ランキング

---

## 5.3 Distance Distribution Analysis

- 類似語距離ヒストグラム
- 全語彙平均との差
- Z-score

---

## 5.4 Projection View

- PCA 2D投影
- 主成分寄与率表示
- クラスタID表示

---

# 6. Non-Functional Requirements

- 再現可能性100%
- 実行時間は5秒以内（語彙10万以下想定）
- メモリ使用量を明示
- ロジックはクラス設計

---

# 7. Explainability Policy

各出力に対して、
以下の説明要素を必ず付与する：

- 距離計算式
- 距離平均との差
- 品詞影響の有無
- クラスタ所属
- コーパス頻度

「なぜ？」に数値で答えること。

---

# 8. Evaluation Metrics

- staticとSBERTの差分一貫性
- 同品詞凝集率
- 近傍安定性（Top-K変動）

---

# 9. Coding Standards

- 関数型よりクラス設計を優先
- 型ヒント必須
- Docstring必須
- ログ出力実装
- 例外処理明示

---

# 10. Out of Scope

- LLM生成
- クラウド接続
- API通信
- リアルタイム学習

---

# 11. Deliverables

- 動作するStreamlitアプリ
- 完全なREADME
- 設計図説明ドキュメント
- 処理フロー図
- 距離計算説明書

---

# 12. Data Loading Policy
- ベクトルは assets/embeddings から読み込む
- メタデータは assets/metadata から読み込む
- manifest.json による整合性チェックを必須とする
- shape不一致時は例外を投げる

---

# 13. Final Goal

本プロジェクトは
「意味空間を説明できるNLP設計能力」を証明するものである。

コード量ではなく、
構造理解を示すことが最終目的である。