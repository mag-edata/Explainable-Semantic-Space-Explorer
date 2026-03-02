# SKILL.md — 実装スキルサマリー

## このプロジェクトで証明するスキル

### 1. NLP 設計理解

| 項目 | 実装箇所 |
|---|---|
| Word2Vec と SBERT の構造的差異の理解 | `compare()` の rank_diff / similarity_diff |
| コサイン類似度の数学的構造の把握 | `distance_metrics.py` の完全自前実装 |
| 品詞が意味空間に与える影響の分析 | `pos_filter.py` の heterogeneity_rate |
| 距離分布の統計的解釈（Z-score） | `analyzer.py` の enrich_distribution |

---

### 2. ソフトウェア設計

| 設計原則 | 実装箇所 |
|---|---|
| 依存注入（DI） | `SimilarityEngine(vectors, vocab, pos_tags, metrics)` |
| 単一責任原則 | 各クラスが1つの役割のみ担う |
| 層の分離（core / analysis / ui） | Streamlit を core に持ち込まない |
| カスタム例外による異常系の明示 | 各モジュールに `XxxError` クラスを定義 |

---

### 3. Python 実装スキル

| 技術 | 実装内容 |
|---|---|
| numpy バッチ演算 | `cosine_similarity_batch`: `matrix @ query` で O(N) 一括計算 |
| 効率的 Top-K | `np.argpartition` で O(N+k log k)（全ソートより高速） |
| dataclass 設計 | `SearchResult`, `ComparisonResult`, `DistributionStats` |
| 型ヒント完備 | 全メソッド・引数・戻り値に型アノテーション |
| Docstring 完備 | 数式・引数・戻り値・例外を全メソッドに記述 |

---

### 4. データ整合性保証

| チェック項目 | 実装箇所 |
|---|---|
| manifest.json との shape/dtype 照合 | `_validate_against_manifest()` |
| 4ファイル間の N（語彙数）一致確認 | `_validate()` |
| vocab index の連続性検証 | `set(values) == set(range(N))` |

---

## 実装済みモジュール一覧

```
core/
├── embedding_loader.py   資産読み込み + インデックス整合チェック
├── distance_metrics.py   コサイン類似度完全自前実装（numpy のみ）
├── similarity_engine.py  Top-K 検索 / static vs SBERT 比較
├── pos_filter.py         品詞フィルタリング / 異品詞率
└── analyzer.py           Z-score 付与 / ヒストグラム / 近傍安定性
```
