"""data_pipeline package.

Asset generation pipeline. Run with ``python -m data_pipeline.<name>``.
Not used at inference time; executed only during setup.

Execution order:
  1. Vocabulary generation: ``vocab.merge`` (internally calls
     ``vocab.gen_brown`` / ``vocab.gen_wiki``).
  2. Model training: ``train.train_w2v`` (saves the Word2Vec model into
     ``models/``).
  3. Vector / metadata export: ``export.static_vectors`` /
     ``export.contextual_vectors`` / ``export.vocab_pos``.
  4. Integrity: ``manifest``.

Example commands:
  python -m data_pipeline.vocab.merge
  python -m data_pipeline.train.train_w2v
  python -m data_pipeline.export.static_vectors
  python -m data_pipeline.export.contextual_vectors
  python -m data_pipeline.export.vocab_pos
  python -m data_pipeline.manifest
"""
