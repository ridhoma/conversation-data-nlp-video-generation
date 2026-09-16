# Conversation NLP for a Video-Generation Assistant

A data-science project analysing conversations between users and an AI assistant embedded in a video-creation workflow. Users talk to the assistant in natural language to build and iterate on videos — describing what they want, refining the script, adjusting visuals, and correcting it when it gets things wrong. This repo turns that messy conversational text into signal about **what makes an interaction succeed**.

## Question

Given several months of assistant conversations and the videos they produced, what separates successful assistant interactions from unsuccessful ones — and what would you change about the product as a result?

**Success metric:** a video reaching **published** state (chosen over `generated`, which is too weak a bar, and `downloaded`, which is too sparse/noisy — see `ANALYSIS_PLAN.md`).

## Data

The datasets are **not included in this repository** — download them from Google Drive and place them under `data/`. See **[`data/README.md`](data/README.md)** for the link and the expected folder layout.

Three tables under `data/raw/` (cleaned versions in `data/processed/`):

| Table | Rows | What it captures |
|-------|------|------------------|
| `messages.csv` | ~15K | One row per message (user ↔ assistant), with `content`, `model`, `conversation_id`, `timestamp` |
| `videos.csv` | ~2K | Video lifecycle: `created_at → generated_at → published_at → downloaded_at` |
| `users.csv` | ~225 | User metadata: `plan`, `workspace_id`, `country` |

The message data is synthetic (heavily templated, fictional entities), so the analysis treats it *as if real* but carries the non-independence caveats — see the notes in `ANALYSIS_PLAN.md`.

## Approach

1. **Data preparation** — clean casing/date inconsistencies, resolve referential integrity, treat informative nulls correctly (a video never generated ≠ generated-but-not-published). Every quality issue is documented, not silently dropped.
2. **NLP layer** (`nlp/`):
   - **Intent classification** — two-stage zero-shot (`bart-large-mnli`): a coarse bucket (Create / Edit / Approval / Dissatisfaction), then Edit split into content vs visual.
   - **Brief completeness** — regex extraction of brief elements (script, avatar, visual, tone, structure) into a 0–5 completeness score.
   - **Repetition detection** — sentence-embedding cosine similarity to flag when a user restates an earlier request in the same conversation.
3. **Modelling** (`data_science/`) — assemble a per-conversation feature table, attach the `published` target, and fit an interpretable logistic regression (odds ratios) to see what moves the outcome.

## Layout

```
conversation-data-nlp-video-generation/
├── nlp/                    NLP engine
│   ├── pipelines.py        MessageIntentPipeline, FullBriefCheckerPipeline, RepetitionDetector
│   ├── rubric.py           Tunable label / keyword configs
│   ├── utils.py            Model loaders, cosine similarity, zero-shot helper, text cleaning
│   └── paths.py            Repo-root-anchored data paths
├── data_science/           Analysis layer
│   ├── features.py         Per-conversation feature engineering
│   ├── modeling.py         scikit-learn wrappers (metrics, odds-ratio table)
│   └── visualisation.py    Matplotlib/seaborn plot helpers
├── notebooks/              01 cleaning · 02 NLP · 03 regression
├── examples/               Standalone CLI demos of each NLP component
├── data/                   raw/ and processed/ CSVs
├── ANALYSIS_PLAN.md        Analytical strategy and findings
└── IMPLEMENTATION_PLAN.md  Build plan and engineering decisions
```

## Running it

First, **download the data** (see [`data/README.md`](data/README.md)) and place it under `data/` — it's not bundled with the repo.

Dependencies come from the personal workspace's local `ds-core[nlp,viz,model]` copy (see
`pyproject.toml` / `uv.lock`). This keeps the project independent from the company
workspace while preserving the same reproducible stack.

```bash
uv sync
jupyter lab   # pick the "conversation-data-nlp-video-generation" kernel
```

Then run the notebooks in order (`01` → `02` → `03`). `nlp/` and `data_science/` are installable packages, so notebooks import them directly (`from nlp.pipelines import MessageIntentPipeline`). Model weights download on first run.

Try the NLP components standalone:

```bash
python examples/demo_intent.py "change the background to blue"
python examples/demo_full_brief.py
python examples/demo_repetition.py
```

## License

MIT
