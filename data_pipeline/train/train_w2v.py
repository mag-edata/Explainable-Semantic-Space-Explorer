"""
train_w2v.py

Inputs:
    ``data/nltk_data/corpora/brown`` (the NLTK Brown corpus).
    HuggingFace ``pszemraj/simple_wikipedia`` (network or cache).

Outputs:
    ``models/w2v_brown10_simplewiki10_sg_300d_w5.model``

This module trains a Word2Vec model from the combined corpus generated
from the Brown corpus and the Simple Wikipedia corpus and saves it under
``models/``.

The goal is to produce a general-purpose, stable word embedding space
that reflects the vocabulary distribution of multiple corpora, so that
downstream synonym search and vector arithmetic can consistently reuse
a pre-determined model.

Assumes HuggingFace network access is available at setup time
(see CONST-02 in the requirements document).
"""

from pathlib import Path
from typing import List

from datasets import load_dataset
from gensim.models import Word2Vec
from nltk.corpus import brown

from data_pipeline._common.nltk_setup import ensure_nltk_resource
from data_pipeline._common.tokenizer import normalize_tokens, tokenize_text
from data_pipeline.vocab.merge import merge

# ---------- Output path ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
W2V_MODEL: Path = PROJECT_ROOT / "models" / "w2v_brown10_simplewiki10_sg_300d_w5.model"


def load_brown_sentences(vocab: set) -> List[List[str]]:
    """
    Generate a sentence list from the Brown corpus for Word2Vec training.
    Extremely short sentences are dropped as noise.

    Parameters
    ----------
    vocab : set[str]
        Allowed (merged) vocabulary set.

    Returns
    -------
    List[List[str]]
        List of per-sentence token streams used for Word2Vec training.
    """
    ensure_nltk_resource("corpora/brown", "brown")
    sents = brown.sents()

    sentences: List[List[str]] = []

    for sent in sents:
        tokens = normalize_tokens(sent)
        tokens = [w for w in tokens if w in vocab]
        if len(tokens) >= 3:
            sentences.append(tokens)

    return sentences


def load_wiki_sentences(vocab: set) -> List[List[str]]:
    """
    Generate a sentence list from the Simple Wikipedia corpus for Word2Vec training.
    Extremely short token streams are dropped as noise.

    Parameters
    ----------
    vocab : set[str]
        Allowed (merged) vocabulary set.

    Returns
    -------
    List[List[str]]
        List of token streams used for Word2Vec training.
    """
    try:
        dataset = load_dataset("pszemraj/simple_wikipedia", split="train")
    except Exception as e:
        raise RuntimeError(
            "Simple Wikipediaデータセットのロードに失敗しました。"
        ) from e

    sentences: List[List[str]] = []

    for item in dataset:
        text = item.get("text", "")
        tokens = tokenize_text(text, vocab=vocab)
        if len(tokens) >= 5:
            sentences.append(tokens)

    return sentences


def load_corpus() -> List[List[str]]:
    """
    Generate a unified Word2Vec training corpus that combines the Brown
    corpus and the Simple Wikipedia corpus.

    Returns
    -------
    List[List[str]]
        The unified corpus (list of per-sentence token streams).
    """
    vocab = set(merge())

    sentences: List[List[str]] = []
    sentences.extend(load_brown_sentences(vocab))
    sentences.extend(load_wiki_sentences(vocab))

    return sentences


if __name__ == "__main__":
    corpus = load_corpus()

    model = Word2Vec(
        sentences=corpus,
        vector_size=300,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
    )

    W2V_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(W2V_MODEL))

    print("Word2Vec モデルの学習が完了しました")
    print(f"- 出力先: {W2V_MODEL}")
    print(f"- 学習文数: {len(corpus)}")
