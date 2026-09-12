"""
Reusable NLP helpers.

These are deliberately separated from the pipeline classes so any sub-step of a
pipeline can be imported and reproduced independently — e.g. in a notebook you
can `from nlp.utils import cosine_similarity` and recompute exactly what
RepetitionDetector does internally, to sanity-check it.

    from nlp.utils import cosine_similarity, load_embedder, pick_device
    model = load_embedder(device=pick_device())
    emb = model.encode(["make it blue", "the background should be blue"],
                       normalize_embeddings=True)
    cosine_similarity(emb[0], emb[1])     # -> the same number the pipeline uses
"""

from __future__ import annotations
from functools import lru_cache
import numpy as np
import html, re

__all__ = [
    "clean_html_escape_characters",
    "pick_device",
    "set_seed",
    "load_zero_shot",
    "load_embedder",
    "cosine_similarity",
    "zero_shot_top",
]

# ── Text Cleansing ───────────────────────────────────────────────────────────────

def clean_html_escape_characters(s):
    if not isinstance(s, str):
        return s                                     
    s = html.unescape(s)                             
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()               
    return s

# ── Environment ───────────────────────────────────────────────────────────────
def pick_device(device: str | None = None) -> str:
    """Return the torch device string. Auto: MPS if available, else CPU."""
    if device is not None:
        return device
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    """Seed everything for reproducibility (call once at notebook top)."""
    import torch
    import transformers
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


# ── Model loaders (cached: load once, reuse everywhere) ─────────────────────────
@lru_cache(maxsize=2)
def load_zero_shot(model_name: str = "facebook/bart-large-mnli", device: str | None = None):
    """Load (once, cached) a zero-shot-classification pipeline."""
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=model_name, device=pick_device(device))


@lru_cache(maxsize=2)
def load_embedder(model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
    """Load (once, cached) a SentenceTransformer embedder."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=pick_device(device))


# ── Reusable computations ───────────────────────────────────────────────────────
def cosine_similarity(a, b) -> np.ndarray:
    """
    Cosine similarity between vectors/matrices, returned as a 2-D array of
    shape (n_a, n_b). Accepts 1-D vectors (treated as a single row).

        cosine_similarity(vec, vec)        -> shape (1, 1)
        cosine_similarity(vec, matrix)     -> shape (1, k)
        cosine_similarity(matrix, matrix)  -> shape (n, m)
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def zero_shot_top(clf, texts: list[str], labels: list[str], batch_size: int = 32,
                  hypothesis_template: str | None = None,
                  progress: bool = False, desc: str = "zero-shot"):
    """
    Run a zero-shot pipeline over a list of texts and return the top
    (label, score) per item. Blank strings are skipped (-> (None, nan)) so the
    model never receives empty input.

    `hypothesis_template` is the sentence MNLI tests entailment against
    (default: "This example is {}."). It's a major accuracy lever — e.g. a
    verb-phrase template like "The user is {}." pairs well with verb-phrase
    labels. None uses the pipeline's default.

    `progress=True` shows a tqdm bar over batches (for bulk runs on large corpora).
    """
    top_label: list = [None] * len(texts)
    top_score: list = [float("nan")] * len(texts)

    idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not idx:
        return top_label, top_score

    non_blank = [texts[i] for i in idx]
    hyp = {} if hypothesis_template is None else {"hypothesis_template": hypothesis_template}

    if progress:
        from tqdm.auto import tqdm
        results = []
        for start in tqdm(range(0, len(non_blank), batch_size), desc=desc, unit="batch"):
            chunk = non_blank[start:start + batch_size]
            r = clf(chunk, candidate_labels=labels, batch_size=batch_size, **hyp)
            results.extend([r] if isinstance(r, dict) else r)
    else:
        results = clf(non_blank, candidate_labels=labels, batch_size=batch_size, **hyp)
        if isinstance(results, dict):          # single-item inputs return a dict
            results = [results]

    for pos, r in zip(idx, results):
        top_label[pos] = r["labels"][0]
        top_score[pos] = float(r["scores"][0])
    return top_label, top_score
