from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MessageIntentRubric:
    hypothesis_template: str                 # e.g. "The user is {}."
    intent_labels: dict[str, str]            # hypothesis -> coarse category
    edit_intent_labels: dict[str, str]            # hypothesis -> edit sub-type
    other_label: str = "Other"               # rule-assigned to blank/nonsensical messages


DEFAULT_MESSAGE_INTENT_RUBRIC = MessageIntentRubric(
    hypothesis_template="The user is {}.",
    intent_labels={
        "asking to create or make a new video, or providing a script, brief, or topic for one": "Create",
        "asking to change, revise, or adjust an existing video, including specifying what its script should say or how it should look": "Edit",
        "confirming the video looks good, appreciating the video, or approving it as final": "Approval",
        "pointing out that the assistant did something wrong or made an unwanted change": "Dissatisfaction",
    },
    edit_intent_labels={
        # content = the spoken words; visual vocab widened (layout/template/captions/…)
        # to fix Edit-visual recall (0.57 -> 0.79 on the 63-row gold set).
        "editing the spoken script, narration, or wording": "Edit content",
        "editing the visuals: avatar, background, colour, font, layout, template, "
        "captions, overlay, slides, transition, or animation": "Edit visual",
    },
)


@dataclass(frozen=True)
class FullBriefCheckerKeywords:
    avatar_names: tuple[str, ...]          # known avatar pool; a bare name ⇒ avatar specified
    avatar_label_cues: tuple[str, ...]     # "<cue>:" (e.g. "avatar:") ⇒ explicit avatar assignment
    visual_terms: tuple[str, ...]          # words that signal visual direction
    tone_terms: tuple[str, ...]            # words that signal tone / style
    script_cues: tuple[str, ...]           # substrings that signal a script is provided
    min_labels_for_structured: int = 2     # "structured" ⇔ ≥ this many "Label:" sections


DEFAULT_FULL_BRIEF_CHECKER_KEYWORDS = FullBriefCheckerKeywords(
    avatar_names=("Sam", "Casey", "Riley", "Drew", "Alex", "Jordan", "Morgan", "Taylor"),
    avatar_label_cues=("avatar", "presenter", "speaker"),
    visual_terms=(
        "scene", "background", "transition", "font", "layout", "colour", "color",
        "logo", "overlay", "shot", "frame", "montage", "split screen", "split-screen",
        "screen", "animation", "graphic", "infographic", "visual",
    ),
    tone_terms=("professional", "casual", "formal", "friendly", "corporate", "serious", "fun"),
    script_cues=("script:",),
    min_labels_for_structured=2,
)
