"""
train_w2v.py

本モジュールは、
BrownコーパスおよびSimple Wikipediaコーパスから生成した
統合コーパスを用いてWord2Vecモデルを学習し、
models/ 配下へ保存する。

目的は、
複数コーパス由来の語彙分布を反映した
汎用的かつ安定した単語埋め込み空間を生成し、
下流の類義語検索やベクトル演算等において
一貫して再利用可能なモデルを事前に確定することである。

セットアップ時の HuggingFace ネットワークアクセスを前提とする
（要件定義書 CONST-02 を参照）。
"""

from pathlib import Path
from typing import List

from datasets import load_dataset
from gensim.models import Word2Vec
from nltk.corpus import brown

from data_pipeline._common.nltk_setup import ensure_nltk_resource
from data_pipeline._common.tokenizer import normalize_tokens, tokenize_text
from data_pipeline.vocab.merge import merge

# ---------- 出力パス ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
W2V_MODEL: Path = PROJECT_ROOT / "models" / "w2v_brown10_simplewiki10_sg_300d_w5.model"


def load_brown_sentences(vocab: set) -> List[List[str]]:
    """
    BrownコーパスからWord2Vec学習用のsentence listを生成する。
    極端に短い文はノイズとして除外する。

    Parameters
    ----------
    vocab : set[str]
        使用を許可する統合語彙集合。

    Returns
    -------
    List[List[str]]
        Word2Vec学習に用いる文単位トークン列のリスト。
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
    Simple WikipediaコーパスからWord2Vec学習用のsentence listを生成する。
    極端に短いトークン列はノイズとして除外する。

    Parameters
    ----------
    vocab : set[str]
        使用を許可する統合語彙集合。

    Returns
    -------
    List[List[str]]
        Word2Vec学習に用いるトークン列のリスト。
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
    BrownコーパスおよびSimple Wikipediaコーパスを統合した
    Word2Vec学習用コーパスを生成する。

    Returns
    -------
    List[List[str]]
        統合コーパス(文単位トークン列のリスト)。
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
