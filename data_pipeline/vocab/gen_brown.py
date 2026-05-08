"""
gen_brown.py

入力:
    data/nltk_data/corpora/brown  （NLTK Brownコーパス）

出力:
    set[str]  （ファイル出力なし。merge.py へ渡す中間データ）

本モジュールは、
NLTKのBrownコーパスを用いて、
自然言語処理タスク向けの語彙(vocabulary)を構築するための前処理関数を提供する。

低頻度語を除外することでノイズを抑え、
単語分散表現、分類モデル、言語モデル等の学習を安定させる設計とする。
"""

from collections import Counter

from nltk.corpus import brown

from data_pipeline._common.nltk_setup import ensure_nltk_resource
from data_pipeline._common.tokenizer import normalize_tokens


def gen_brown(min_freq: int = 10) -> set[str]:
    """
    指定出現回数以上の単語のみを集めた語彙集合(vocabulary)を構築する。

    Parameters
    ----------
    min_freq : int, optional
        語彙に含める最小出現回数。

    Returns
    -------
    set[str]
        出現回数条件を満たした単語のset。
    """
    # Brownコーパスの全単語を取得
    ensure_nltk_resource("corpora/brown", "brown")
    words_raw = brown.words()

    # tokenizerを適用
    words = normalize_tokens(words_raw)

    # 単語の出現回数をカウント
    counter = Counter(words)
    # 出現回数がmin_freq以上の単語を返却
    return {w for w, f in counter.items() if f >= min_freq}


if __name__ == "__main__":
    vocab = gen_brown()
    print("Brownコーパスから語彙を生成しました")
    print(f"- 件数: {len(vocab)}")
