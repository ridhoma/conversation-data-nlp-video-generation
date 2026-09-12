from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .rubric import (
    MessageIntentRubric,
    FullBriefCheckerKeywords,
    DEFAULT_MESSAGE_INTENT_RUBRIC,
    DEFAULT_FULL_BRIEF_CHECKER_KEYWORDS,
)
from .utils import cosine_similarity, load_embedder, load_zero_shot, pick_device, zero_shot_top


def _as_series(texts) -> pd.Series:
    """Coerce input to a Series of strings, preserving index if it's a Series."""
    if isinstance(texts, pd.Series):
        return texts.astype("string").fillna("")
    return pd.Series(list(texts), dtype="string").fillna("")


class MessageIntentPipeline:
    """
    Two-stage zero-shot intent classification.

    Stage 1 assigns a coarse bucket; Stage 2 refines only the "Edit" bucket into
    content vs visual. Set collapse_edit_split=True to keep a single "Edit" label
    (the agreed fallback if Stage 2 proves unreliable in validation).

    Configured by a `MessageIntentRubric` (hypothesis template + stage-1/stage-2
    label maps); see nlp/rubric.py. `Other` is applied by a rule (blank/nonsensical
    messages), not by the model.

    predict(texts, progress=True) -> DataFrame[intent, intent_confidence,
    edit_intent_split_confidence]. `progress=True` shows a tqdm bar — use it for
    bulk runs over large corpora (e.g. all user messages).
    """

    def __init__(self, rubric: MessageIntentRubric = DEFAULT_MESSAGE_INTENT_RUBRIC, 
                batch_size: int = 32,
                collapse_edit_split: bool = False,
                model_name: str = "facebook/bart-large-mnli", 
                device: str | None = None):
        self.rubric = rubric
        self.batch_size = batch_size
        self.collapse_edit_split = collapse_edit_split
        self.model_name = model_name
        self.device = pick_device(device)

    def predict(self, texts, batch_size: int | None = None, progress: bool = False) -> pd.DataFrame:
        s = _as_series(texts)
        bs = batch_size or self.batch_size
        r = self.rubric
        clf = load_zero_shot(self.model_name, self.device)
        items = s.tolist()

        s1_hyp, s1_conf = zero_shot_top(clf, items, list(r.intent_labels), bs,
                                        hypothesis_template=r.hypothesis_template,
                                        progress=progress, desc="intent · stage 1")
        stage1_bucket = [r.intent_labels.get(h) if h else None for h in s1_hyp]

        intent = list(stage1_bucket)
        stage1_confidence = list(s1_conf)
        stage2_confidence = [float("nan")] * len(items)

        if not self.collapse_edit_split:
            edit_pos = [i for i, b in enumerate(stage1_bucket) if b == "Edit"]
            if edit_pos:
                edit_texts = [items[i] for i in edit_pos]
                s2_hyp, s2_conf = zero_shot_top(clf, edit_texts, list(r.edit_intent_labels), bs,
                                                hypothesis_template=r.hypothesis_template,
                                                progress=progress, desc="intent · stage 2 (edits)")
                for j, i in enumerate(edit_pos):
                    if s2_hyp[j] is not None:
                        intent[i] = r.edit_intent_labels[s2_hyp[j]]
                        stage2_confidence[i] = s2_conf[j]

        # Rule: blank / nonsensical messages are `Other` (never sent to the model).
        for i, t in enumerate(items):
            if not (isinstance(t, str) and t.strip()):
                intent[i] = r.other_label
                stage1_confidence[i] = 1.0
                stage2_confidence[i] = float("nan")

        return pd.DataFrame({
            "intent": intent,
            "intent_confidence": stage1_confidence,
            "edit_intent_split_confidence": stage2_confidence,
        }, index=s.index)

class FullBriefCheckerPipeline:
    """
    Regex detection of brief elements + a 0-5 completeness score. Intended to run
    on messages Stage 1 labelled "Create". Detection vocabulary is configured by a
    `FullBriefCheckerKeywords` (see nlp/rubric.py) so it's transparent and tunable.

    extract(texts) -> DataFrame[has_script, has_avatar, has_visual, has_tone,
                                structured, completeness_score]
    """

    def __init__(self, keywords: FullBriefCheckerKeywords = DEFAULT_FULL_BRIEF_CHECKER_KEYWORDS):
        self.keywords = keywords
        names = "|".join(re.escape(n.lower()) for n in keywords.avatar_names)
        cues = "|".join(re.escape(c.lower()) for c in keywords.avatar_label_cues)
        # avatar := a NAMED presenter (known name) OR an explicit "<cue>:" label.
        # (Bare "presenter at desk" must NOT match — it's generic, not an avatar choice.)
        parts = ([rf"\b(?:{cues})\s*:"] if cues else []) + ([rf"\b(?:{names})\b"] if names else [])
        self._avatar_re = re.compile("|".join(parts) or r"(?!x)x")
        self._visual_re = re.compile(r"\b(?:" + "|".join(re.escape(t.lower()) for t in keywords.visual_terms) + r")\b")
        self._tone_re = re.compile(r"\b(?:" + "|".join(re.escape(t.lower()) for t in keywords.tone_terms) + r")\b")
        self._script_re = re.compile("|".join(re.escape(c.lower()) for c in keywords.script_cues))
        # structured := N+ "Label:"-style sections (any 1-5 word label ending in a colon)
        self._label_re = re.compile(r"[a-z][a-z0-9]*(?: [a-z0-9]+){0,4}:")

    def _row(self, text: str) -> dict:
        t = (text or "").lower()
        has_script = bool(self._script_re.search(t))
        has_avatar = bool(self._avatar_re.search(t))
        has_visual = bool(self._visual_re.search(t))
        has_tone = bool(self._tone_re.search(t))
        structured = len(self._label_re.findall(t)) >= self.keywords.min_labels_for_structured
        return {
            "has_script": has_script, "has_avatar": has_avatar, "has_visual": has_visual,
            "has_tone": has_tone, "structured": structured,
            "completeness_score": int(has_script + has_avatar + has_visual + has_tone + structured),
        }

    def extract(self, texts) -> pd.DataFrame:
        s = _as_series(texts)
        return pd.DataFrame([self._row(t) for t in s.tolist()], index=s.index)


class RepetitionDetector:
    """
    Flags a user message that restates an earlier one in the SAME conversation
    (max cosine similarity vs all prior user messages > threshold).

    Texts are embedded once (batched); similarities use nlp.utils.cosine_similarity
    so the computation can be reproduced independently in a notebook.

    detect(df, text_col, conversation_col, order_col)
        -> DataFrame[is_repetition, repetition_similarity, matched_prior_id]
           indexed like `df`.
    """

    def __init__(self, threshold: float = 0.80, batch_size: int = 32,
                 model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.threshold = threshold
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = pick_device(device)

    def detect(self, df: pd.DataFrame, text_col: str = "content",
               conversation_col: str = "conversation_id", order_col: str = "timestamp",
               threshold: float | None = None) -> pd.DataFrame:
        thr = self.threshold if threshold is None else threshold
        model = load_embedder(self.model_name, self.device)

        work = df[[text_col, conversation_col, order_col]].copy()
        texts = work[text_col].astype("string").fillna("").tolist()
        emb = model.encode(texts, batch_size=self.batch_size,
                           convert_to_numpy=True, normalize_embeddings=True)

        is_rep = pd.Series(False, index=df.index)
        sim = pd.Series(float("nan"), index=df.index)
        matched = pd.Series(pd.NA, index=df.index, dtype="object")

        # positional map so we can index the embedding matrix per conversation
        pos = {label: i for i, label in enumerate(df.index)}
        for _, grp in work.groupby(conversation_col, sort=False):
            order = grp.sort_values(order_col).index.tolist()
            for k in range(1, len(order)):
                cur = pos[order[k]]
                priors = [pos[order[j]] for j in range(k)]
                sims = cosine_similarity(emb[cur], emb[priors]).ravel()
                best = int(np.argmax(sims))          # index into priors (0..k-1)
                sim.loc[order[k]] = float(sims[best])
                if sims[best] > thr:
                    is_rep.loc[order[k]] = True
                    matched.loc[order[k]] = order[best]   # the prior message it restates

        return pd.DataFrame(
            {"is_repetition": is_rep, "repetition_similarity": sim, "matched_prior_id": matched},
            index=df.index,
        )
