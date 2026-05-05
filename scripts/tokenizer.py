"""
tokenizer.py

本モジュールは、トークン正規化およびトークン化するための前処理関数を提供する。

目的は、プロジェクト全体のトークン定義を一致させることであり、
語彙生成・コーパス生成・モデル学習・推論等の段階において
前処理の一貫性と再現性を保証する。
"""

from typing import Iterable, List, Optional

from scripts.token_definition import TOKEN_CONSTRAINT_PATTERN, TOKEN_EXTRACT_PATTERN


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    """
    既に分割済のトークン列を処理対象とする。
    主に単語単位で提供されるテキストを想定する。
    (例：NLTK Brown corpus)

    Parameters
    ----------
    tokens : Iterable[str]
        分割済トークン列

    Returns
    -------
    List[str]
        正規化・フィルタリング適用後のトークン列
    """
    return [w.lower() for w in tokens if TOKEN_CONSTRAINT_PATTERN.match(w)]


def tokenize_text(text: str, vocab: Optional[set[str]] = None) -> List[str]:
    """
    生テキストを処理対象とする。
    主に文章単位で提供されるテキストを想定する。
    (例：Hugging Face Datasets Simple Wikipedia)

    Parameters
    ----------
    text : str
        生テキスト
    vocab : set[str], optional
        使用可能とする語彙集合。
        指定された場合、この集合に含まれるトークンのみを返す。

    Returns
    -------
    List[str]
        トークン化・正規化・フィルタリング・(必要に応じて)語彙制約適用後のトークン列
    """
    text = text.lower()
    candidates = TOKEN_EXTRACT_PATTERN.findall(text)
    tokens = [w for w in candidates if TOKEN_CONSTRAINT_PATTERN.match(w)]

    if vocab is not None:
        tokens = [w for w in tokens if w in vocab]

    return tokens
