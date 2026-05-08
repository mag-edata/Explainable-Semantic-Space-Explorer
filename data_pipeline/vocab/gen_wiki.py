"""
gen_wiki.py

入力:
    HuggingFace pszemraj/simple_wikipedia（ネットワークまたはキャッシュ）

出力:
    set[str]（ファイル出力なし。merge.py へ渡す中間データ）

本モジュールは、
Hugging Face DatasetsのSimple Wikipediaコーパスを用いて、
自然言語処理タスク向けの語彙(vocabulary)を構築するための前処理関数を提供する。

低頻度語を除外することでノイズを抑え、
単語分散表現、分類モデル、言語モデル等の学習を安定させる設計とする。

セットアップ時の HuggingFace ネットワークアクセスを前提とする
"""

from collections import Counter

from datasets import load_dataset

from data_pipeline._common.tokenizer import tokenize_text


def gen_wiki(min_freq: int = 10) -> set[str]:
    """
    指定出現回数以上の単語のみを集めた語彙集合(vocabulary)を構築する。

    Parameters
    ----------
    min_freq : int, optional
        語彙に含める最小出現頻度。

    Returns
    -------
    set[str]
        出現回数条件を満たした単語のset。
    """
    # Simple Wikipediaコーパスをロード
    # 初回実行時はネットワークアクセスが必要
    try:
        dataset = load_dataset("pszemraj/simple_wikipedia", split="train")
    except Exception as e:
        raise RuntimeError(
            "Simple-Wikipediaデータセットのロードに失敗しました。"
            "datasetsキャッシュが利用可能であることを確認するか、"
            "ネットワークアクセス環境で load_dataset を一度実行してください。"
        ) from e

    # 単語の出現回数をカウント
    counter: Counter[str] = Counter()

    # 各記事を走査し、tokenizerを適用
    for item in dataset:
        text = item.get("text", "")
        tokens = tokenize_text(text)
        counter.update(tokens)

    # 出現回数がmin_freq以上の単語を返却
    return {w for w, freq in counter.items() if freq >= min_freq}


if __name__ == "__main__":
    vocab = gen_wiki()
    print("Simple Wikipediaから語彙を生成しました")
    print(f"- 件数: {len(vocab)}")
