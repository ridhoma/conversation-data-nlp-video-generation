"""
Conversation NLP — NLP package.

Layout:
    nlp.pipelines  — the pipeline classes (MessageIntent, FullBriefChecker, Repetition)
    nlp.rubric     — config objects (MessageIntentRubric, FullBriefCheckerKeywords)
    nlp.utils      — reusable helpers (cosine_similarity, model loaders, seeding)
    nlp.paths      — project paths (ROOT, RAW, PROCESSED)

Convenience re-exports below let you `from nlp import MessageIntentPipeline`, but the
explicit `from nlp.pipelines import ...` / `from nlp.utils import ...` forms are
equally supported.
"""

from .pipelines import (
    MessageIntentPipeline,
    FullBriefCheckerPipeline,
    RepetitionDetector,
)
from .rubric import (
    MessageIntentRubric, DEFAULT_MESSAGE_INTENT_RUBRIC,
    FullBriefCheckerKeywords, DEFAULT_FULL_BRIEF_CHECKER_KEYWORDS,
)
from .utils import cosine_similarity, load_embedder, load_zero_shot, pick_device, set_seed

__all__ = [
    "MessageIntentPipeline",
    "FullBriefCheckerPipeline",
    "RepetitionDetector",
    "MessageIntentRubric",
    "DEFAULT_MESSAGE_INTENT_RUBRIC",
    "FullBriefCheckerKeywords",
    "DEFAULT_FULL_BRIEF_CHECKER_KEYWORDS",
    "cosine_similarity",
    "load_embedder",
    "load_zero_shot",
    "pick_device",
    "set_seed",
]
