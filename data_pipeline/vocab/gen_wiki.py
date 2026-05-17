"""
gen_wiki.py

Inputs:
    HuggingFace ``pszemraj/simple_wikipedia`` (network or cache).

Outputs:
    ``set[str]`` (no file output; intermediate data passed to ``merge.py``).

This module provides the preprocessing function for building a vocabulary
suitable for NLP tasks from Hugging Face Datasets' Simple Wikipedia
corpus.

Excluding low-frequency words suppresses noise and stabilizes training
for word embeddings, classifiers, and language models.

Assumes HuggingFace network access is available at setup time.
"""

from collections import Counter

from datasets import load_dataset

from data_pipeline._common.tokenizer import tokenize_text


def gen_wiki(min_freq: int = 10) -> set[str]:
    """
    Build a vocabulary set containing only words above a minimum frequency.

    Parameters
    ----------
    min_freq : int, optional
        Minimum occurrence count required for inclusion in the vocabulary.

    Returns
    -------
    set[str]
        Set of words that satisfy the frequency condition.
    """
    # Load the Simple Wikipedia corpus.
    # Network access is required on first execution.
    try:
        dataset = load_dataset("pszemraj/simple_wikipedia", split="train")
    except Exception as e:
        raise RuntimeError(
            "Simple-Wikipediaデータセットのロードに失敗しました。"
            "datasetsキャッシュが利用可能であることを確認するか、"
            "ネットワークアクセス環境で load_dataset を一度実行してください。"
        ) from e

    # Count word occurrences
    counter: Counter[str] = Counter()

    # Iterate over each article and apply the tokenizer
    for item in dataset:
        text = item.get("text", "")
        tokens = tokenize_text(text)
        counter.update(tokens)

    # Return words occurring at least min_freq times
    return {w for w, freq in counter.items() if freq >= min_freq}


if __name__ == "__main__":
    vocab = gen_wiki()
    print("Simple Wikipediaから語彙を生成しました")
    print(f"- 件数: {len(vocab)}")
