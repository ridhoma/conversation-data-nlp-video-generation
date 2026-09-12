#!/usr/bin/env python3
"""
Component 2 — Full-brief deep-dive (REGEX, no model).

Detects which brief elements are present and returns a completeness score (0-5).
In the real pipeline this runs only on messages Stage 1 (Component 1) labelled "Create".

Usage:
    python examples/ex2_full_brief.py "Script: Welcome... Avatar: Sam. Tone: professional."

Reference demo — the production wrapper will live in src/nlp.py.
"""
import argparse
import re

AVATAR_NAMES = ["Sam", "Casey", "Riley", "Drew"]   # real list: mine from the data


def brief_elements(text):
    t = text.lower()
    has_script = bool(re.search(r"script\s*:", t)) or len(text) > 300  # label OR prose heuristic
    has_avatar = bool(re.search(r"(avatar|presenter|speaker)\s*:?\s*\w+", t)) \
        or any(n.lower() in t for n in AVATAR_NAMES)
    has_visual = bool(re.search(r"\b(scene|background|transition|font|layout|colou?r)\b", t))
    has_tone = bool(re.search(r"\b(professional|casual|formal|friendly|corporate|serious|fun)\b", t))
    structured = bool(re.search(r"(script:|scene guide:|avatar:|tone:)", t))
    flags = dict(has_script=has_script, has_avatar=has_avatar,
                 has_visual=has_visual, has_tone=has_tone, structured=structured)
    return flags, sum(flags.values())


def main():
    ap = argparse.ArgumentParser(description="Regex full-brief element detection.")
    ap.add_argument("text", help="the brief message to inspect")
    args = ap.parse_args()

    flags, score = brief_elements(args.text)
    print(f"completeness={score}/5")
    for k, v in flags.items():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
