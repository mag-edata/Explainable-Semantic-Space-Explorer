"""
nltk_setup.py

NLTK リソースをリポジトリ内 data/nltk_data/ に集約するためのユーティリティ。

デフォルトの ~/nltk_data への分散を防ぎ、
プロジェクト内で再現性のあるセットアップを保証する。
"""

from pathlib import Path

import nltk

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
NLTK_DATA_DIR: Path = PROJECT_ROOT / "data" / "nltk_data"


def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    """
    NLTK リソースを data/nltk_data/ に確保する。
    未取得の場合は自動ダウンロードする。

    Parameters
    ----------
    resource_path : str
        nltk.data.find() に渡すパス（例: "corpora/brown"）。
    package_name : str
        nltk.download() に渡すパッケージ名（例: "brown"）。
    """
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name, download_dir=str(NLTK_DATA_DIR))
