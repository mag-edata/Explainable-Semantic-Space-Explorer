"""
nltk_setup.py

Utility for consolidating NLTK resources under the repository-local
``data/nltk_data/`` directory.

Prevents resources from being scattered into the default
``~/nltk_data`` location and guarantees a reproducible per-project setup.
"""

from pathlib import Path

import nltk

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
NLTK_DATA_DIR: Path = PROJECT_ROOT / "data" / "nltk_data"


def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    """
    Ensure an NLTK resource is present under ``data/nltk_data/``.
    Automatically downloads it if missing.

    Parameters
    ----------
    resource_path : str
        Path passed to ``nltk.data.find()`` (for example, ``"corpora/brown"``).
    package_name : str
        Package name passed to ``nltk.download()`` (for example, ``"brown"``).
    """
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name, download_dir=str(NLTK_DATA_DIR))
