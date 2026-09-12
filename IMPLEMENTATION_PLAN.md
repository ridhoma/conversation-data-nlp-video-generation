# Implementation Plan

This document outlines the technical execution strategy for the conversation-analysis project, grounded in the `ANALYSIS_PLAN.md`.

## 1. Environment & Dependencies

We will use **Python 3.11.8** (available via your local `pyenv`) as it offers maximum stability and compatibility with standard ML/NLP libraries. We will use `uv` for lightning-fast dependency management.

**Core Packages:**
- **Data Manipulation:** `pandas`, `numpy`
- **NLP / ML:** 
  - `transformers` (for Hugging Face zero-shot classification pipelines)
  - `torch` (backend for transformers)
  - `sentence-transformers` (for repetition detection embeddings)
- **Visualization:** `matplotlib`, `seaborn` (for generating charts to back product recommendations)
- **Environment:** `jupyter`, `ipykernel` (for the analysis notebooks)

**Setup Commands:**
```bash
pyenv local 3.11.8
uv venv
source .venv/bin/activate
uv pip install pandas numpy transformers torch sentence-transformers matplotlib seaborn jupyter
```

## 2. Project Directory Structure

We will adopt a modular structure to separate raw data, reusable source code, and exploratory notebooks. This keeps the notebooks clean and makes the complex logic (like NLP wrappers and data cleaning functions) testable and importable.

```text
conversation-data-nlp-video-generation/
├── data/
│   ├── raw/                  # Original CSV files (messages.csv, users.csv, videos.csv)
│   └── processed/            # Cleaned and NLP-enriched datasets
├── notebooks/
│   ├── 01_data_prep.ipynb         # Data cleaning execution & Success Metric validation
│   ├── 02_nlp_pipeline.ipynb      # Running the heavy NLP models to enrich messages
│   └── 03_analysis.ipynb          # Conversation anatomy, charts, & Product insights
├── nlp/                      # importable package (editable install: `uv pip install -e .`)
│   ├── __init__.py           # convenience re-exports
│   ├── pipelines.py          # Pipeline classes: Zero-shot, Regex, Embeddings
│   ├── utils.py              # Reusable helpers (cosine_similarity, model loaders, seeding)
│   └── paths.py              # Project paths (ROOT, RAW, PROCESSED)
├── ANALYSIS_PLAN.md
├── IMPLEMENTATION_PLAN.md
└── README.md
```

## 3. NLP Pipeline Design

The NLP processing will be handled in a batch-oriented manner to optimize the inference speed of local transformer models.

### Component 1: Intent Classification (Two-Stage Zero-Shot)

**Method: hierarchical (two-stage) zero-shot** with `facebook/bart-large-mnli`. Flat zero-shot over 7 overlapping labels lets similar classes (edit-content / edit-visual / correction) bleed into each other, since MNLI scores each label independently and takes the argmax of near-ties. Classifying in two stages attacks this structurally, needs **no training data to run**, and keeps cost at zero.

- **Model:** `facebook/bart-large-mnli` (via `transformers.pipeline("zero-shot-classification")`), same model for both stages.
- **Inputs:** Raw `content` of all `role == 'user'` messages.
- **Stage 1 — coarse bucket:** the model classifies into the four real intents `{Create, Edit, Approval, Dissatisfaction}`; blank/nonsensical messages are routed to `Other` by an explicit rule (not the model — this corpus has no off-topic messages, only empty ones).
- **Stage 2 — refine:** only messages landing in `Edit` are re-classified into `{Edit content, Edit visual}`. Other buckets pass through unchanged. `Create` messages feed Component 2 (full-brief deep-dive), which resolves the old dependency where Component 2 relied on a flat 7-way argmax.
- **Execution:** batched inference (`batch_size=32`). Top label mapped back to short category names and stored in an `intent` column.
- **Baseline for free:** also run the original flat 7-label zero-shot once, so we can quantify the hierarchy's lift (flat X% → two-stage Y%) against the validation gold set rather than asserting it.
- **Validation gate (before the real run):** build a small **validation-only** gold set (~80–100 hand-labelled user messages) to measure accuracy and inspect the confusion between classes. This is validation, not training — the pipeline runs without it.
- **Weak-split fallback (no overengineering):** if Stage 2 can't reliably separate edit-content from edit-visual, we simply **drop the split and keep a single `Edit` label**. No rules, no embeddings, no rebuild — this is an exploratory analysis, not production.

### Component 2: Full Brief Deep-Dive (Regex)
- **Filter:** Only run on rows where Stage 1 labelled the message `Create`.
- **Method:** Define a dictionary of regex patterns in `nlp/pipelines.py`.
  - `has_script`: `(?i)(script:|here is the script)` or length heuristics.
  - `has_avatar`: Match against a dynamically extracted list of avatars (or common names if known).
  - `has_visuals`: `(?i)(scene|background|transition)`
- **Output:** Boolean columns (`has_script`, `has_avatar`, etc.) and a summed integer column `completeness_score` (0 to 5).

### Component 3: Repetition Detection (Embeddings)

The former Component 3a (binary frustration zero-shot) was **dropped** — explicit frustration is captured by the `Dissatisfaction` intent (Component 1), more reliably than the over-flagging binary classifier. Repetition is the *implicit* frustration signal (restating a request the assistant ignored).

- **Model:** `all-MiniLM-L6-v2` (via `SentenceTransformer`).
- **Execution Strategy:**
  1. Filter to user messages only.
  2. Group by `conversation_id` and sort by `timestamp`.
  3. Embed all messages once (batched).
  4. For any message at index `i`, check if `max(cosine similarity with messages 0..i-1) > 0.80` (threshold empirically validated).
- **Output:** Boolean column `is_repetition` (plus `repetition_similarity` and `matched_prior_id` for inspection/tuning).
- **Threshold validation:** the 0.80 cutoff replaces the initial 0.85 placeholder — a real restatement pair ("make the background blue" / "the background should be blue like I said") scored **0.84**, just under 0.85, so 0.85 would have missed it. Naturally-similar edit requests ("make it blue" / "make it green") can also score high — consider restricting comparison to nearby turns rather than all priors.

### Reproducibility & Determinism (finalize at the very end)

The notebooks must be re-runnable by anyone and produce the same labels. We defer the hardening to the end of the build (once the pipeline is stable) rather than front-loading it, but the checklist is:

- **Pin model revisions** — pass an explicit `revision=` (commit hash) to every `from_pretrained` / `SentenceTransformer` load so a re-download doesn't silently swap weights.
- **Set seeds** — `torch.manual_seed`, `numpy` seed, and `transformers.set_seed` at notebook top.
- **Cache enriched outputs** — write the NLP-labelled DataFrame to `data/processed/` so `03_analysis.ipynb` reads the cached CSV and never has to re-run inference. Inference is the slow, non-deterministic-risk step; the analysis on top of it should be instant and fully deterministic.
- **Record environment** — the pinned `requirements.txt` already covers package versions.

## 4. Data I/O & Pipeline Flow

Architecturally, we will build **3 distinct NLP pipelines (modules)** in `nlp/pipelines.py`, because—as you correctly noted—they require different parameters and contexts:

1. **`IntentPipeline`**: Takes `texts` and returns two-stage intent labels.
2. **`FullBriefPipeline`**: Takes `texts` and applies regex.
3. **`RepetitionDetector`**: Takes a `(df, text_col, conversation_col, order_col)` and flags per-conversation restatements via embeddings + cosine.

In `02_nlp_pipeline.ipynb`, we will orchestrate these 3 distinct modules to sequentially enrich our master DataFrame:

**1. Input (from `01_data_prep.ipynb`)**
```json
{
  "message_id": "msg_001",
  "conversation_id": "conv_001",
  "timestamp": "2023-10-01T12:00:00Z",
  "content": "Make a video about cybersecurity with avatar Sam."
}
```

**2. Independent Execution (in `02_nlp_pipeline.ipynb`)**
The notebook imports the modules from `nlp/pipelines.py`, configures them, and passes only the necessary data to each. The modules return independent results (e.g., a Series or a DataFrame subset), without mutating the master DataFrame directly.

- `intent_preds = IntentPipeline().predict(df.set_index('message_id')['content'])`
- `fb_preds = FullBriefPipeline().extract(df.loc[intent_preds['stage1_bucket'] == 'Create', 'content'])`
- `rep_preds = RepetitionDetector().detect(df, text_col='content', conversation_col='conversation_id', order_col='timestamp')`

**3. Merging Outputs**
Once all pipelines have run, the notebook merges their outputs back onto the master DataFrame (e.g., via `pd.concat` or mapping by `message_id`).
```json
{
  "message_id": "msg_001",
  ...
  "nlp_intent": "Create",
  "nlp_is_repetition": false,
  "fb_has_script": false,               
  "fb_has_avatar": true,                
  "fb_completeness_score": 1            
}
```
This enriched DataFrame is then ready to be merged back with the `videos` and `users` tables in the final analysis notebook.

## 5. Execution Workflow

1. **Step 1:** Initialize the `uv` environment and create directory scaffolding.
2. **Step 2:** Write `clean.py` (project root, or wherever you keep cleaning code) and execute `01_data_prep.ipynb` to apply all cleaning logic and output clean CSVs.
3. **Step 3:** Implement the NLP wrapper classes in `nlp/pipelines.py`. Write and execute `02_nlp_pipeline.ipynb` to run the inference models and output the enriched datasets.
4. **Step 4:** Perform the final roll-ups in `03_analysis.ipynb` and draft the final insights for the writeup.

## 6. Appendix — Alternatives Considered for Intent Classification

We evaluated several approaches for Component 1 and chose **two-stage zero-shot** for its balance of zero cost, zero training, and structural handling of label overlap. Alternatives, kept here for the record:

### Embedding few-shot k-NN / nearest-centroid (leading fallback)

A zero-cost, **zero-training** embedding method — not to be confused with fine-tuning (SetFit). Embed ~10 labelled examples per class with `all-MiniLM-L6-v2`, embed each message, and assign it to the nearest class by cosine similarity (nearest-centroid or k-NN). No gradient updates — just `model.encode()` plus a distance comparison, so inference is seconds.

- **Why it's attractive:** anchors on real in-domain examples rather than abstract label strings, so it often beats zero-shot on nuanced classes; fully local and deterministic.
- **Why it's not primary:** needs a handful of labelled anchors per class to *run* (two-stage zero-shot needs labels only to *validate*), and its ceiling is still bounded by the embedding model's off-the-shelf quality.
- **When we'd switch:** if the validation gold set shows two-stage zero-shot underperforming even after dropping the weak edit split, this is the first thing to try — the gold set we already built doubles as the anchor set, so there's no wasted work.

### Others (rejected)

- **SetFit (few-shot fine-tuning):** strongest local option, but the fine-tuning step costs time we'd rather not spend on a time-boxed analysis.
- **LLM few-shot via API (e.g. Claude, structured output):** highest raw quality and could collapse Component 1+Component 2+Component 3 into one pass, but non-zero cost and someone without an API key couldn't re-run the analysis.
- **Unsupervised clustering (HDBSCAN/KMeans on embeddings):** no training, but produces fuzzy per-message labels and needs manual cluster inspection. Retained only as an optional light *exploratory* pass ("what's in the conversations"), not as a classifier.
