"""data_pipeline パッケージ。

資産生成パイプライン。`python -m data_pipeline.<name>` で実行する。
推論時には使用せず、セットアップ時のみ実行する。

実行順序:
  1. 語彙生成: vocab.merge（内部で vocab.gen_brown / vocab.gen_wiki を呼ぶ）
  2. ベクトル/メタデータ出力: export.static_vectors / export.contextual_vectors / export.vocab_pos
  3. 整合性: manifest

実行コマンド例:
  python -m data_pipeline.vocab.merge
  python -m data_pipeline.export.static_vectors
  python -m data_pipeline.export.contextual_vectors
  python -m data_pipeline.export.vocab_pos
  python -m data_pipeline.manifest
"""
