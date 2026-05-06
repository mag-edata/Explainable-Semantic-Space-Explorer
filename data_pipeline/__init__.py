"""data_pipeline パッケージ。

資産生成パイプライン。`python -m data_pipeline.<name>` で実行する。
推論時には使用せず、セットアップ時のみ実行する。

実行順序:
  1. 語彙生成: gen_brown_vocab → gen_wiki_vocab → merge_vocab
  2. ベクトル/メタデータ出力: export_static_vectors / export_contextual_vectors / export_vocab_pos
  3. 整合性: gen_manifest
"""
