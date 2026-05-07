# Explainable Semantic Space Explorer

単語埋め込み空間を可視化・分析し、**「なぜその単語が近いのか」を数値で説明する**ツール。

- 静的埋め込み（Word2Vec）と文脈埋め込み（SBERT）の差異を定量的に比較
- コサイン類似度の計算を内積・ノルム・式の形式で分解し、ブラックボックスにしない
- 外部 API 不使用・ローカル CPU のみで完全動作

---

## 課題

NLP の実務では単語埋め込みが広く使われているが、その挙動は説明しにくい。

### 典型的なブラックボックス問題

- **結果は出るが理由がわからない**  
  「"king" の近傍に "queen" が出る」は知っていても、なぜ近いのかを数値で説明できない

- **モデル間の差異が把握しにくい**  
  Word2Vec（静的）と SBERT（文脈）は同じ単語を異なる位置に置く。その差を定量的に比較する手段が乏しい

- **類似度スコアの意味が不明瞭**  
  0.82 は「高い」のか「普通」なのか、語彙全体の分布を見なければ判断できない

その結果：
- モデル選定の根拠が感覚的になる
- 埋め込みの挙動をチームで共有しにくい
- デバッグ・改善の試行錯誤にコストがかかる

---

## 解決策

このツールは、埋め込みを**理解するための対象**として扱う。

1. **類似度の内訳を表示**  
   コサイン類似度を `dot(a,b) / (‖a‖·‖b‖)` の形式で分解し、内積・ノルムを個別に確認できる

2. **Z-score で相対的な位置を把握**  
   語彙全体の分布における外れ値度を数値化し、0.82 が本当に高いかを判断できる

3. **静的 vs 文脈埋め込みを並列比較**  
   Word2Vec と SBERT の近傍語・ランク差・共通語・固有語を対称的に比較できる

4. **2D 投影 + クラスタリングで構造を可視化**  
   PCA / UMAP で埋め込み空間を俯瞰し、KMeans で語彙の意味的グループを確認できる

5. **品詞フィルタリングで絞り込み**  
   名詞・動詞・形容詞などで近傍語を絞り込み、統語的なバイアスを分離できる

---

## アーキテクチャ

依存方向は一方向のみ。逆流禁止。

```
data/ ──► core/ ──► transforms/ ──► ui/app.py
```

| 層 | 役割 |
|---|---|
| `data/` | 埋め込みデータ（読み取り専用） |
| `core/` | 距離計算・検索・統計など純粋ロジック |
| `transforms/` | PCA・UMAP・KMeans などベクトル変換 |
| `ui/` | Streamlit で表示するだけ。ロジック記述禁止 |

例：

入力（クエリ語）
- "bank"

出力（静的 vs 文脈 比較）

**Word2Vec（静的）の近傍**
- river, shore, creek, lake ...
  → 文脈なしのため、地理的な意味に引っ張られる

**SBERT（文脈）の近傍**
- financial, credit, loan, fund ...
  → コーパス中の文脈から、金融的な意味が優勢になる

→ 同じ "bank" でも埋め込みモデルによって**意味空間の位置が変わる**ことが数値で確認できる

---

## 設計方針

### なぜコサイン類似度を自前実装するか

scipy / sklearn の `cosine_similarity` を使えば 1 行で済む。
あえて自前実装することで、計算過程（内積・ノルム・除算）を UI 上に明示し、
「類似度とは何か」を説明可能にする。

### なぜ静的と文脈の両方を扱うか

Word2Vec は単語に固定ベクトルを割り当てる（多義語を区別しない）。
SBERT は文脈を考慮して動的にベクトルを生成する。
この差異を並列表示することで、「どちらが優れているか」ではなく「何が違うのか」を理解できる。

### なぜ外部 API を使わないか

再現性と透明性を最優先にするため、外部サービスへの依存を排除した。
ローカル CPU 環境で完全に動作し、コードを読めば挙動がすべて追える。

---

## 既知の課題

### [UI] 投影・クラスタタブ: クエリ語マーカーの凡例と図の不一致

- **現象**: Altair の `shape` エンコーディングでクエリ語に `"cross"` を指定しているが、凡例の表示と図上のマーカー形状が一致しないことがある
- **試みた対策**:
  - `range=["star", "circle"]` → Vega-Lite の無効値のため凡例が消滅
  - チャートをレイヤー分割（近傍語・クエリ語を別 `alt.Chart`）→ 描画自体が消えた
- **現在の状態**: `range=["cross", "circle"]` に戻して保留中
- **候補解決策**:
  - `mark_rule` / `mark_point` を組み合わせてクエリ点を別途オーバーレイする
  - `shape` を廃止し `size` + `color` でクエリを目立たせる

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

## このプロジェクトが示すもの

- 埋め込み空間の「説明可能性」を設計としてどう組み込むか
- 外部 API に依存しないローカル完結 NLP パイプラインの構築
- 静的 vs 文脈埋め込みの差異を定量的に可視化・比較するアーキテクチャ
- 192テストによるコアロジックの品質担保（core 層 + transforms 層を網羅）

---

## Demo

> *(Pending: `data/` の埋め込みファイルが大容量のため Streamlit Cloud へのデプロイを検討中)*

ローカルでの動作確認は「セットアップ」を参照。

---

## セットアップ

```bash
# 1. リポジトリのクローン
git clone https://github.com/mag-edata/Explainable-Semantic-Space-Explorer.git
cd Explainable-Semantic-Space-Explorer

# 2. 仮想環境の作成・有効化
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. 依存パッケージのインストール
pip install -r requirements.txt

# 4. NLTK データのダウンロード
python -c "import nltk; nltk.download('brown')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

# 5. 資産ファイルの生成（初回のみ。HuggingFace へのアクセスが必要）
python -m data_pipeline.vocab.merge                    # → data/metadata/vocab.json
python -m data_pipeline.train.train_w2v               # → models/w2v_brown10_simplewiki10_sg_300d_w5.model
python -m data_pipeline.export.static_vectors         # → data/embeddings/static_vectors.npy
python -m data_pipeline.export.contextual_vectors     # → data/embeddings/contextual_vectors.npy
python -m data_pipeline.export.vocab_pos              # → data/metadata/vocab_pos.npy
python -m data_pipeline.manifest                      # → data/manifest.json

# 7. アプリ起動
venv/bin/python3 -m streamlit run ui/app.py
# → http://localhost:8501 がブラウザで開く
```

### 動作確認済み環境

| 項目 | バージョン |
|---|---|
| Python | 3.12 |
| numpy | 2.4.2 |
| scikit-learn | 1.8.0 |
| umap-learn | 0.5.11 |
| streamlit | 1.54.0 |
| altair | 6.0.0 |

---

## プロジェクト構造

```
Explainable-Semantic-Space-Explorer/
├── data/                        # 埋め込みベクトル・メタデータ（Git管理外）
│   ├── embeddings/
│   │   ├── static_vectors.npy     # Word2Vec  shape (83823, 300)
│   │   └── contextual_vectors.npy      # SBERT     shape (83823, 384)
│   ├── metadata/
│   │   ├── vocab.json             # 語彙リスト
│   │   └── vocab_pos.npy          # 品詞ラベル配列 shape (83823,)
│   └── manifest.json              # shape / dtype 整合チェック用
│
├── core/                          # 純粋ロジック層
│   ├── embedding_loader.py        # ベクトル読み込み・整合チェック
│   ├── similarity_engine.py       # 類似度検索・比較
│   ├── distance_metrics.py        # コサイン類似度の自前実装
│   ├── pos_filter.py              # 品詞フィルタリング
│   └── analyzer.py                # 距離分布の統計分析
│
├── transforms/                    # ベクトル変換層
│   ├── clustering.py              # KMeans クラスタリング
│   └── projection.py              # PCA / UMAP による 2D 投影
│
├── ui/
│   └── app.py                     # Streamlit UI（4タブ）
│
├── tests/                         # 単体テスト群（192テスト全通過）
│   ├── test_distance_metrics.py
│   ├── test_similarity_engine.py
│   ├── test_embedding_loader.py
│   ├── test_pos_filter.py
│   ├── test_analyzer.py
│   ├── test_clustering.py
│   └── test_projection.py
│
├── data_pipeline/                 # 資産生成パイプライン（セットアップ時のみ実行）
│   ├── manifest.py
│   ├── _common/
│   │   ├── token_definition.py
│   │   └── tokenizer.py
│   ├── vocab/
│   │   ├── gen_brown.py
│   │   ├── gen_wiki.py
│   │   └── merge.py
│   ├── export/
│   │   ├── static_vectors.py
│   │   ├── contextual_vectors.py
│   │   └── vocab_pos.py
│   └── train/
│       └── train_w2v.py           # Word2Vec モデル学習
│
├── models/                        # Word2Vec モデル配置場所（Git管理外）
│
├── DOCS/                          # 設計ドキュメント
│   ├── 要件定義書.md
│   ├── 基本設計書.md
│   ├── 詳細設計書.md
│   ├── テスト設計書.md
│   └── テスト項目書.md
│
├── requirements.txt
└── README.md
```

---

## 背景

大学で英語学・英語コーパス・統計的著者推定を学んだことをきっかけに、Python によるデータ分析と AI に興味を持った。

単語の「意味の近さ」を統計的に扱う研究に触れる中で、埋め込みモデルが**なぜそう判断するのか**を説明する手段がないことに課題を感じた。

このプロジェクトは、その課題を出発点として設計した。ツールとして「動く」だけでなく、**埋め込み空間の挙動を理解・説明できること**を最優先に置いている。
