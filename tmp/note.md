# static_words.npy および sbert_words.npy は今回のPJには不要。残すかは要検討。

# 今回のPJと現在の以下.pyファイルの命名相性については要検討。

# 以下の.pyファイルが準拠するpath理論は今回のPJに適合しない。

## 他PJでのPATH管理

- paths.py

## vocab.jsonの生成

- merge_vocab.py
  - gen_brown_vocab.py
  - gen_wiki_vocab.py

## static_words.npyの生成 および
## static_vectors.npyの生成

- export_static_vectors.py

## sbert_words.npyの生成 および
## sbert_vectors.npyの生成

- export_sbert_vectors.py

## vocab_pos.npyの生成

- export_vocab_pos.py
  - tokenizer.py
  - token_definition.py