#!/usr/bin/env python3
"""
Component 1 — Intent classification (TWO-STAGE zero-shot, bart-large-mnli).

Stage 1: coarse bucket {Create, Edit, Correction, Approval, Other}.
Stage 2: only if Stage 1 == "Edit", split into content vs visual.

Usage:
    python examples/ex1_intent.py "rewrite the second paragraph"
    python examples/ex1_intent.py "change the background to blue"

Reference demo — the production wrapper will live in src/nlp.py.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

STAGE1_LABELS = ["Create", "Edit", "Correction", "Approval", "Other"]
STAGE2_LABELS = ["Edit content (script/wording/text)",
                 "Edit visual (avatar/background/font/transition)"]


def classify_intent(zs, text):
    s1 = zs(text, candidate_labels=STAGE1_LABELS)
    bucket, conf = s1["labels"][0], s1["scores"][0]
    if bucket == "Edit":                       # only refine the Edit bucket
        s2 = zs(text, candidate_labels=STAGE2_LABELS)
        return s2["labels"][0], s2["scores"][0]
    return bucket, conf


def main():
    ap = argparse.ArgumentParser(description="Two-stage zero-shot intent classification.")
    ap.add_argument("text", help="the user message to classify")
    args = ap.parse_args()

    from transformers import pipeline
    zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    label, score = classify_intent(zs, args.text)
    print(f"intent={label}  confidence={score:.2f}")


if __name__ == "__main__":
    main()
