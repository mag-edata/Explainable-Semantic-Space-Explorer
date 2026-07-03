# CLAUDE.md — Explainable Semantic Space Explorer

## 目的

本プロジェクトは、**NLP学習者が「単語埋め込み」の教科書的主張を実データで検証・腹落ちできる、説明付きの実験室**である。静的（Word2Vec）と SBERT の単語ベクトル空間を比較分析する。

---

## プロダクト方針（2026-07-03 確定）

- 対象ユーザー：**NLP学習者**（学生・若手エンジニア・LLMから基礎に遡る人）。実務のモデル選定用途は対象外と明記
- 中核価値：**類似度スコアの相対化**（Z-score・分布。「0.82は語彙全体の上位何%か」を平易な判定文で示す）
- 式分解（内積・ノルム）は「なぜ近いかの説明」ではなく「**どう計算されるかの透明化・教材**」として扱う。意味的因果の過大主張は禁止
- 単語単独encodeのSBERTベクトルを「文脈埋め込み」と表示しない（正直なラベリング。文脈の本物化はPhase Bで実現）

### 段階実施計画

| 段階 | 内容 |
|------|------|
| **Phase A**（データ品質・UX是正） | 語彙クレンジング（8.4万語→良質な2〜3万語目標）＋資産再生成、POSタグ生成是正、Word2Vec再学習（`seed=42, workers=1`）、正規化統一、計算キャッシュ＋form化、例示ボタン・OOV候補提示・判定文。コーパス拡張（WikiText-103）はクレンジング後の品質を見て**条件付き**実施 |
| **Phase B**（文脈の本物化） | 文入力モード（文中トークンベクトル抽出・2文比較で多義性を観察）、比較ビューを主役に画面再構成 |
| **Phase C**（公開） | Streamlit Cloud デプロイ（Phase A でデータ縮小後）、README 全面改訂 |

- 保留タスク（Streamlit Cloud デプロイ・GIF貼り付け）は **Phase A 完了まで着手しない**

---

## ドキュメントの役割分離

CLAUDE.md
- 内部意思決定用（本ファイル）
- 絶対制約・設計方針・進捗管理を記録

README.md
- 外部公開用（Phase C で全面改訂予定）

DOCS/
- 設計書群（要件定義書・基本設計書・詳細設計書・テスト設計書・テスト項目書）
- `phase3_roadmap/`：改善バックログ
- 内部仕様の詳細記録

---

## 絶対制約（違反禁止）

1. 外部 API 禁止（OpenAI 等）
2. **コサイン類似度は自前実装**（scipy / sklearn の `cosine_similarity` 使用禁止）
3. ローカル CPU 環境のみで動作保証
4. 推論時の学習処理禁止（※文入力モードでのローカルモデルによる encode は許可。通信・学習は不可 — CONST-03 v2.0 改訂）
5. 乱数使用時は seed 固定（**学習パイプラインを含む**）
6. manifest.json によるインデックス整合チェック必須

---

## コーディング規約

- **クラス設計優先**（関数型より）
- 型ヒント必須（全引数・戻り値）
- Docstring 必須（全クラス・全メソッド）
- ログ出力実装（`logging` モジュール）
- 例外処理明示（カスタム例外クラスを定義して使う）

---

## セットアップ

`data/` の `.npy` / `.json` は Git 管理外（大容量のため）。
セットアップ手順（NLTK データ DL・資産生成コマンド）は README.md の「セットアップ」セクションを参照。

---

## リポジトリ状態

最終更新：2026-07-03

### フェーズ1（実装）

→ 全タスク完了 ✅

### フェーズ2（デプロイ）

→ **Phase C に統合**（Phase A のデータ縮小後に再開）。GitHub 公開は完了済み ✅

### フェーズ3（改善）

| 項目 | 状態 | 備考 |
|------|------|------|
| 方針確定・要件定義 v2.0 改訂 | ✅ 完了（2026-07-03） | ユーザー像を「NLP学習者」に再定義 |
| Phase A-1：語彙curation＋整合修正＋seed固定 | ✅ 完了（2026-07-03） | `curate.py`（WordNet+stopword membership・最小長3・短縮形ノイズ除外）、`merge.py`配線、`static_vectors.py`で1.1整合修正、`train_w2v.py` seed=42/workers=1、`manifest.py` training_date自動記入。再生成・検証済（83,823→**40,032語**、全ファイル行整合、資産約230MB→約110MB）。commit `9d5cab4`＋`4209881` |
| Phase A-2：POS再生成是正(1.2)・正規化統一(1.4) | ✅ 完了（2026-07-03、要コミット） | 1.2=`vocab_pos.py`をWordNet由来の品詞導出に変更（辞書順`pos_tag`廃止。king誤判定verb→noun等を修正、any 625→50）。1.4=`static_vectors.py`でL2正規化しcontextualと対称化（cos順位不変を実証）。ローカル再生成・検証済。再生成データはgitignore、manifest差分なし＝コミット対象はコードのみ |
| Phase A-3：UX是正（例示・OOV候補・判定文・form化） | ✅ 完了（2026-07-03） | FR-25判定文（`Analyzer.interpret_similarity`＋Tab1/3表示、既commit）／FR-23例示ボタン／FR-24 OOV候補（difflib）／FR-26 form＋`@st.cache_data`でタブ横断の重複計算を解消。AppTestで実描画・ボタン挙動・OOV候補を検証、208+9テストpass |
| Phase B：文入力モード | ⬜ 未着手 | Phase A 完了後 |
| Phase C：デプロイ・README改訂 | ⬜ 未着手 | Phase A/B 完了後 |

> **Phase A-1 の再生成手順**：README「Full Data Pipeline Setup」の順序どおり再実行（`merge → train_w2v → static_vectors → contextual_vectors → vocab_pos → manifest`）。NLTK は brown/tagger に加え **wordnet・stopwords** の取得が必要。`static_vectors` が `vocab.json` を確定するので順序厳守。
