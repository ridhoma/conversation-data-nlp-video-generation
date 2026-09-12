"""
Project paths, anchored to the repo root — independent of the notebook's CWD.

    from nlp.paths import RAW, PROCESSED
    messages = pd.read_csv(RAW / "messages.csv")
    clean.to_csv(PROCESSED / "messages_clean.csv", index=False)

Resolves from this file's location (nlp/paths.py -> repo root is one level up),
so it works whether the notebook runs from notebooks/, the repo root, or anywhere.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../conversation-data-nlp-video-generation
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

__all__ = ["ROOT", "DATA", "RAW", "PROCESSED"]
