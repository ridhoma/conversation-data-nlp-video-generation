#!/usr/bin/env python3
"""
Component 3 — Repetition flag (embeddings + cosine, all-MiniLM-L6-v2).

Given the ordered user messages of ONE conversation, flags each message that
restates an earlier one (max cosine similarity vs all priors > --threshold).

Usage:
    python examples/ex3b_repetition.py "make the background blue" "add a title card" "the background should be blue like I said"
    python examples/ex3b_repetition.py msg1 msg2 msg3 --threshold 0.75

Reference demo — the production wrapper will live in src/nlp.py.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")


def repetition_flags(embed, user_msgs, threshold):
    from sentence_transformers.util import cos_sim
    embs = embed.encode(user_msgs)
    flags = [False]                    # first message can't be a repeat
    for i in range(1, len(user_msgs)):
        sims = [float(cos_sim(embs[i], embs[j])) for j in range(i)]
        flags.append(max(sims) > threshold)
    return flags


def main():
    ap = argparse.ArgumentParser(description="Embedding-based repetition detection within a conversation.")
    ap.add_argument("messages", nargs="+", help="ordered user messages of one conversation")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="repetition if max cosine vs priors exceeds this (default 0.80)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    embed = SentenceTransformer("all-MiniLM-L6-v2")

    flags = repetition_flags(embed, args.messages, args.threshold)
    print(f"(threshold={args.threshold})")
    for flag, msg in zip(flags, args.messages):
        mark = "REPEAT" if flag else "  ok  "
        print(f"  [{mark}] {msg}")


if __name__ == "__main__":
    main()
