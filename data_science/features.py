"""
Conversation-level feature engineering for the "what drives a published video?"
analysis (notebook 03).

The unit of analysis is a **conversation** (1:1 with a video). We assemble one row
per conversation from the cleaned/processed tables and attach the binary target
`published`. All inputs are the cleaned files under data/processed/.

    from data_science.features import build_conversation_feature_table
    feat = build_conversation_feature_table()      # -> DataFrame, one row per conversation

Sources
-------
conversations.csv                     conversation shape (turns, duration)
messages.csv                          raw text (user-message length)
user_messages_intent.csv              per-message intent (Component 1)
create_intent_messages_anatomy.csv    brief completeness for Create messages (Component 2)
videos.csv                            outcomes -> target `published`
users.csv                             plan segment
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlp.paths import PROCESSED

INTENTS = ["Create", "Edit content", "Edit visual", "Approval", "Dissatisfaction", "Other"]
ANATOMY_FLAGS = ["has_script", "has_avatar", "has_visual", "has_tone", "structured"]


def _intent_features(user_msgs: pd.DataFrame) -> pd.DataFrame:
    """Per-conversation intent composition from labelled user messages."""
    counts = (user_msgs.pivot_table(index="conversation_id", columns="intent",
                                    values="message_id", aggfunc="count", fill_value=0))
    for c in INTENTS:                                   # ensure all intent columns exist
        if c not in counts:
            counts[c] = 0
    counts = counts[INTENTS].add_prefix("n_")

    order = user_msgs.sort_values("timestamp")
    opening = order.groupby("conversation_id")["intent"].first().rename("opening_intent")
    diversity = order.groupby("conversation_id")["intent"].nunique().rename("intent_diversity")

    out = counts.join(opening).join(diversity)
    # split edits into content vs visual (mirrors the Component 1 two-stage intent split)
    out["n_edits_content"] = out["n_Edit content"]
    out["n_edits_visual"] = out["n_Edit visual"]
    out["n_edits"] = out["n_edits_content"] + out["n_edits_visual"]
    out["has_approval"] = out["n_Approval"] > 0
    out["has_dissatisfaction"] = out["n_Dissatisfaction"] > 0
    return out.reset_index()


def _anatomy_features(anatomy: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    """Per-conversation brief completeness (Create messages only). A conversation
    with no Create message gets completeness 0 and has_create_brief=False."""
    a = anatomy.merge(messages[["message_id", "conversation_id"]], on="message_id", how="left")
    agg = {f: "max" for f in ANATOMY_FLAGS}             # OR the flags across a conversation's Create msgs
    agg["completeness_score"] = "max"                   # best brief in the conversation
    out = a.groupby("conversation_id").agg(agg).reset_index()
    out = out.rename(columns={"completeness_score": "brief_completeness"})
    out["has_create_brief"] = True
    return out


def _text_features(messages: pd.DataFrame) -> pd.DataFrame:
    """Per-conversation average user-message length (characters)."""
    u = messages[messages["role"] == "user"].copy()
    u["msg_len"] = u["content"].fillna("").astype(str).str.len()
    return (u.groupby("conversation_id")["msg_len"].mean()
            .rename("avg_user_msg_len").reset_index())

