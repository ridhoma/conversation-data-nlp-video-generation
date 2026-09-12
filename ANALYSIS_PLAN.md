# AI Assistant Analysis Plan

## The Problem

The product is an AI assistant embedded in a video-creation workflow. Users converse with it to build and iterate on videos. We need to figure out: **is the assistant actually helping people make videos, and what makes the difference between a good interaction and a bad one?**

### Dataset at a Glance

| Table | Rows | What it captures |
|-------|------|------------------|
| `messages.csv` | ~15K | Individual messages (user ↔ assistant) across conversations |
| `videos.csv` | ~2K | Video lifecycle events (created → generated → published → downloaded) |
| `users.csv` | ~225 | User metadata (plan, workspace, country) |

---

## Part 1: Data Preparation

Clean and prepare the data. Document every quality issue — what it is, how we found it, how we handled it. No silent drops or imputation.

### Issues Already Spotted

- **Inconsistent casing** — `role` has both `user` and `User`; `plan` has `Starter`/`STARTER`/`growth`/`GROWTH`
- **Date format inconsistency** — `users.csv` mixes `YYYY-MM-DD` and `MM/DD/YYYY`
- **Ghost users** — `usr_ghost55` appears in messages. Placeholder? Deleted user? Needs investigation
- **Referential integrity** — do all `video_id`s in messages exist in the videos table? Do all `user_id`s match `creator_id`s?
- **Nulls with meaning** — `generated_at`, `published_at`, `downloaded_at` nulls are informative, not just missing. A video never generated ≠ a video generated but never published
- **Synthetic / templated data** — 6,862 user messages → 1,159 distinct, with identical sentences across dozens of distinct users and recurring fictional entities. We proceed *as if real* but carry the caveats (non-independence, "data tends to" not "users tend to", possible synthetic-by-construction outcomes). See **Interesting Stuff #4**.

### Things to Investigate Further

- Duplicate messages or conversations
- Timestamp ordering within conversations (are messages in logical sequence?)
- Empty or nonsensical message content
- Orphaned records across tables
- Model field completeness (should be null for user messages, populated for assistant messages)

---

## Part 2: Defining the Success Metric

We have three downstream signals forming a **funnel**:

```
Created → Generated → Published → Downloaded
```

### Signal Comparison

| Signal | What it captures | What it misses | Verdict |
|--------|-----------------|----------------|---------|
| **Generated** | The assistant produced *something* | Too low a bar — user could generate a draft and immediately abandon it because it's garbage. High generation ≠ satisfaction | ❌ Too weak |
| **Published** | The user decided the video is ready for others to see — a genuine quality endorsement | Users who create great drafts but never publish (internal review workflows); users who publish under time pressure despite low quality | ✅ Best proxy |
| **Downloaded** | Someone took the final output | Too far downstream — conflates assistant quality with external factors (does the user even download, or share via link?). Likely very sparse | ❌ Too noisy / sparse |

### Our Recommendation: Published

**Published** captures the moment where a user says *"yes, this is ready."* That's the closest signal we have to genuine satisfaction with the assistant's output.

### Caveats to Acknowledge

- Some workflows may not involve publishing (drafts for internal review)
- Publishing doesn't guarantee quality — could be "good enough under deadline"
- We should validate by checking the funnel drop-off rates and whether Published has enough volume to be statistically meaningful
- Consider whether a composite metric or secondary metrics add value

---

## Part 3: What Drives Good Outcomes — Analysis Areas

This is the core task. We'll explore multiple angles to understand what separates successful interactions from unsuccessful ones.

### A. Conversation Anatomy & Patterns

This is the quantitative backbone. We combine raw conversation metrics with aggregated NLP labels (from NLP Components 1 & 3) to build a rich picture of each conversation's shape, then cross-reference with the success metric.

**Raw conversation metrics:**

| Metric | What it captures | Why it matters |
|--------|-----------------|----------------|
| **Turn count** | Total messages in the conversation | Too short = unclear intent, too long = frustration/struggling |
| **User message length** | Average length of user messages | Are longer, more detailed messages associated with better outcomes? |
| **Response time gaps** | Time between messages | Long gaps might indicate confusion or abandonment |

**Features derived from NLP-enriched labels** (from NLP Components 1, 2 & 3):

| Feature | What it captures | Why it matters |
|---------|-----------------|----------------|
| **Opening intent** | What type of message started the conversation (full brief / script only / vague / edit) | Does the opening move predict the whole conversation's outcome? |
| **Intent composition** | Distribution of intent types across the conversation (e.g., 60% edits, 20% dissatisfaction, 20% creation) | Dissatisfaction-heavy conversations likely signal struggle |
| **Dissatisfaction rate** | Fraction of user messages expressing dissatisfaction | Direct measure of assistant failure density |
| **Edit depth** | Total number of edit rounds | More edits = more refinement needed, but is there a sweet spot? |
| **Intent diversity** | How many different intent types appear | Focused conversations vs. scattered ones |
| **Intent trajectory** | The sequence/flow of intents (creation → refinement → dissatisfaction vs. creation → done) | Does the shape of the journey matter, not just the components? |
| **Full brief completeness** | Completeness score of full brief messages (from NLP Component 2) | Do conversations with more complete briefs need fewer edits and corrections? |
| **Frustration density & timing** | Dissatisfaction + repetition signals per conversation and when they appear (from intent labels + Component 3) | Early = misunderstood intent; late = refinement rabbit hole |

**Key questions:**
- What conversation shapes lead to publishing vs. abandonment?
- What's the typical intent flow of a successful conversation vs. a failed one?
- Does dissatisfaction rate have a tipping point beyond which conversations almost always fail?
- Do conversations with higher-completeness full briefs have fewer downstream corrections?

### B. Model Performance

The `model` field gives us a natural experiment. Cross-reference with NLP component labels:

- **Success rate by model** — do different models lead to different publish rates?
- **Model × intent type** (Component 1) — do some models handle certain request types better?
- **Model × dissatisfaction / repetition** (intent + Component 3) — do certain models cause more frustration?
- **Model changes over time** — regime shifts in outcomes when models were swapped?

### C. User Segments

- **Plan type**: do Starter vs. Growth users behave differently? Different expectations?
- **Geography**: any regional patterns in how people interact?
- **Experience**: new users vs. power users — is there a learning curve? Do repeat users improve?
- **Workspace effects**: are some teams/orgs systematically more successful?

### D. Failure Mode Analysis

Frustration signals are covered by the `Dissatisfaction` intent (Component 1, explicit) and repetition (Component 3, implicit). This section focuses on **structural failure patterns** that don't require NLP:

- **Funnel drop-off**: what % of videos stall at each stage (created → generated → published)? Where's the biggest leak?
- **Abandoned conversations**: videos created but never generated — are there conversations with zero assistant responses? Very short conversations that just stop?
- **Silent abandonment**: conversations where the user stops responding after the assistant's reply — the assistant said something but the user walked away
- **Common failure scenarios**: cross-reference structural failures with NLP labels from Component 1 to identify recurring patterns (e.g., vague creation requests that lead to no generation)
- **Template-confound guard** (from Interesting Stuff #4): before trusting any "pattern → outcome" result, check whether outcomes cluster by message/conversation template. If a template family is ~always published (or never), the "signal" may be synthetic-by-construction rather than behavioural — report it as such.

### E. Temporal Patterns

- **Trends over time**: is the assistant getting better or worse month-to-month?
- **Time-of-day / day-of-week effects** on user patience or interaction quality
- **Seasonal patterns** in usage or success rates

---

## NLP Components

These are the dedicated text analysis workstreams — pure NLP labelling at the message level. The labels produced here feed into Part 3's Conversation Anatomy (section A) for quantitative analysis against the success metric.

```
NLP Components (message labels) → Part 3A: Conversation Anatomy (aggregation) → Success Metric (published?)
```

### Methodology

| Tool | Used in | What it does |
|------|---------|-------------|
| **Two-stage zero-shot classification** (`bart-large-mnli`) | Component 1 | Classifies intent hierarchically (coarse bucket → refine) so overlapping labels don't bleed; no training data |
| **Regex / pattern matching** | Component 2 | Detects concrete elements (script text, avatar names, labelled sections) |
| **Sentence embeddings** (`all-MiniLM-L6-v2`) | Component 3 | Computes semantic similarity between messages to detect repetition |

---

### Component 1: Intent Classification

**Method: Two-stage (hierarchical) zero-shot classification.**

Classify every **user message** into an intent category. We do this in two stages rather than a flat 7-way call: **Stage 1** assigns a coarse bucket (`Create / Edit / Approval / Dissatisfaction / Other`), and **Stage 2** splits only the `Edit` bucket into content vs. visual. Flat zero-shot lets overlapping labels bleed together (edit-content / edit-visual / dissatisfaction all score similarly and argmax picks near-ties); the hierarchy fixes this structurally with no training data. A flat 7-label run is kept as a baseline to quantify the lift, validated against a small hand-labelled gold set built before the real run. If Stage 2 proves unreliable, we collapse it to a single `Edit` label. See `IMPLEMENTATION_PLAN.md` for the mechanics and the fallback (embedding few-shot k-NN).

These labels are the raw material that Part 3A aggregates into conversation-level features.

**Candidate intent categories** (to be refined after exploring the data):

| Intent | Description | Example from data |
|--------|-------------|-------------------|
| **Full brief** | User provides script + visual direction + avatar/tone — a complete creation request | *"Here's everything you need: Script: ... Scene guide: ... Avatar: Sam."* |
| **Script provision** | User provides a script but limited or no visual direction | *"Here's my script: This is an introduction to leadership development..."* |
| **Vague creation** | User gives a topic-level request with no specifics | *"make a video about cybersecurity awareness"* |
| **Edit content** | User asks to change script text, wording, or structure | *"rewrite the second section"*, *"the closing line should be 'Thank you for watching'"* |
| **Edit visual** | User asks for visual/style changes (avatar, background, transitions, font) | *"change the scene transition to a fade"*, *"change the font a different style"* |
| **Dissatisfaction** | User expresses that the assistant did something wrong or unwanted | *"why did you change the avatar? I didn't ask for that"*, *"that's not right at all"* |
| **Approval** | User confirms, accepts, or expresses satisfaction with the output | *"looks good"*, *"perfect"*, *"yes, that works"* |

> **Note on Correction vs. Frustration:** The "Correction" intent (Component 1) and frustration signals (Component 3) overlap but serve different purposes. Component 1 labels *what the user is doing* (correcting the assistant). Component 3 labels *how the user is feeling* (frustrated). A correction is always a frustration signal, but frustration can also appear in non-correction messages (e.g., negative language in an edit request). Both labels are useful and complementary.

---

### Component 2: Full Brief Deep-Dive

**Method: Regex / pattern matching.**

For messages classified as **full brief** in Component 1, we do a second pass to detect which elements are actually present. Not all full briefs are equally complete — some provide everything, others are missing key pieces. This tells us *how full* the "full brief" really is.

| Element | What we're looking for | Why it matters |
|---------|----------------------|----------------|
| **Script provided** | Did the user include actual script text? | Concrete content vs. generating from scratch |
| **Visual direction** | Scene descriptions, transitions, layout instructions | Reduces ambiguity in how the video should look |
| **Avatar specified** | Named avatar choice (Sam, Casey, Riley, Drew, etc.) | One less decision for the assistant to guess |
| **Tone/style specified** | "Professional", "casual", "formal", etc. | Sets creative direction expectations |
| **Structured format** | Labelled sections (Script: ... Scene guide: ...) vs. freeform prose | Structured briefs may be easier for the assistant to parse |
| **Completeness score** | How many of the above elements are present (0–5) | Is there a threshold where success jumps? |

**Key questions:**
- Is there a "completeness threshold" where success rate jumps? (e.g., script + avatar = much better than script alone)
- Do structured briefs outperform freeform ones of similar length?
- Which missing elements hurt the most? (e.g., no avatar specified → more corrections later)

---

### Component 3: Repetition Detection (implicit frustration)

**Method: Sentence embeddings** (`all-MiniLM-L6-v2`) + cosine similarity.

> **Why there's no separate frustration classifier.** The original plan had a binary frustration flag (old Component 3a) *and* repetition (3b). We dropped 3a: **explicit** frustration is already captured — more reliably — by the `Dissatisfaction` intent (Component 1). The standalone zero-shot frustration flag badly over-flagged neutral edits (e.g. it scored "change the transition to a fade" as frustrated at 0.88), whereas the intent model correctly leaves that as `Edit`. And in this templated corpus, frustration never rides on a neutral intent — it always appears as a standalone dissatisfaction message — so a separate flag adds no signal. Frustration is therefore captured by **two complementary signals**: `Dissatisfaction` intent (explicit) and repetition (implicit).

Detect when a user re-states a request they already made — a sign the assistant didn't get it the first time (**implicit** frustration, which `Dissatisfaction` intent does *not* catch, since both the original and the restatement are usually neutral `Edit`s). For each user message, compute cosine similarity against all previous user messages in the same conversation. If similarity exceeds a threshold, flag as **repetition**.

```
Message 3: "make the background blue"
Message 7: "the background should be blue, like I said"
→ Cosine similarity: 0.84 → repetition ✅   (threshold ~0.80; see impl plan)
```

#### Timing

Both frustration signals — `Dissatisfaction` intent and repetition — are tagged with their position in the conversation (turn number). This feeds into Part 3A where we aggregate to conversation level and distinguish:

| Timing | What it indicates | Product implication |
|--------|------------------|---------------------|
| **Early** (turns 1–2) | The assistant misunderstood the user's initial intent | Improve intent parsing, add clarifying questions before acting |
| **Late** (turns 4+) | The user is stuck in a refinement rabbit hole — the assistant keeps making small errors or unwanted changes | Improve edit precision, add confirmation steps before applying changes |

---

## Part 4: Product Recommendations

These should flow directly from findings above. Each recommendation needs:
- **What** to build or test
- **Why** — grounded in analysis evidence
- **How to measure** — specific success criteria

### Example Recommendation Templates (to be filled after analysis)

Each grounded in specific analysis outputs:

1. If higher-completeness full briefs succeed more (Component 2) → **build a guided template** that prompts for script + avatar + visual direction + tone before the conversation starts
2. If a specific model outperforms on certain intent types (Component 1 × Section B) → **implement intent-based model routing**
3. If conversations beyond N turns tend to fail (Part 3A: edit depth) → **add proactive check-ins** ("Would you like to try a different approach?")
4. If early frustration is dominant (frustration timing) → **add clarifying questions** before the assistant acts on ambiguous requests
5. If late frustration is dominant (frustration timing) → **add confirmation steps** before applying changes, and improve edit precision

---

## Prioritisation

Given the "day's worth of focused work" constraint, we should prioritise depth over breadth:

| Priority | Area | Why |
|----------|------|-----|
| 🥇 | Success metric argument | Everything depends on this — nail it first |
| 🥇 | NLP Component 1: Intent classification | Core NLP task — the labels that power everything downstream |
| 🥇 | Part 3A: Conversation Anatomy | Quantitative backbone — aggregates NLP labels to conversation level |
| 🥈 | NLP Component 2: Full brief deep-dive | Tells us what makes a good brief — high product value |
| 🥈 | NLP Component 3: Repetition detection | Implicit-frustration signal; complements the Dissatisfaction intent |
| 🥉 | Model comparison | Low-hanging fruit if data supports it |
| 🥉 | User segments & temporal patterns | Supporting context, not primary |


---

## Deprioritised for Now, Pinned for Later

### Topic / Subject Matter Analysis

Analyse *what* users are making videos about (quarterly updates, compliance training, onboarding, etc.), not just *how* they interact with the assistant.

**Why we're skipping it:** The user base is predominantly corporate, so topic variance is likely small (training, updates, onboarding). More importantly, topic findings aren't directly actionable — we can't tell users to stop making certain types of videos. The *how* (brief completeness, instruction clarity, frustration patterns) is far more actionable than the *what*.

**Why it's pinned:** Two scenarios where it could matter:
- **Confounding variable** — if certain topics correlate with vague requests *and* low success, we might misattribute failures to conversation style when it's really the subject matter complexity
- **Template opportunity** — if a large % of conversations cluster around the same video type (e.g., quarterly updates), that's a product signal to build pre-made templates

**Revisit trigger:** If during intent classification we notice topic patterns emerging naturally (e.g., financial content always comes with vague requests), flag it in the write-up and consider a light topic clustering pass.

### Semantic (paraphrase) Repetition Detection

**What we wanted to do.** Component 3's whole reason for using sentence-embedding cosine similarity (rather than string matching) is to catch *implicit* frustration via **paraphrased** repetition — a user restating an earlier request in different words ("make the background blue" … later "the background should be blue, like I said"). Embeddings are the right tool *because* they catch semantic restatement that exact matching would miss.

**The limitation — the data is fully synthetic, so paraphrase repetition doesn't exist.** Running `RepetitionDetector` on the cleaned conversations shows repetition is rare (~4.5% of conversations) and **almost entirely exact verbatim re-sends** (cosine = 1.000): the user literally re-sends the identical templated sentence. There is **no genuine paraphrase repetition** in the corpus. The "near" band (0.80–0.98) that embeddings are supposed to shine on is instead dominated by **false positives** — slot-variant iterative edits that share a template but change the target:

| prior | current | cosine | reality |
|-------|---------|--------|---------|
| "background should be white" | "…light grey" | 0.94 | different colour, **not** a repeat |
| "use avatar Sam instead" | "…Alex instead" | 0.87 | different avatar, **not** a repeat |

Same root cause as the intent finding: MiniLM embeds by *structure/topic*, so template siblings score high. So the embedding machinery can't demonstrate its value here — it would only *add* false positives.

**What we do instead.** Treat repetition as an **exact (normalised) re-send within a conversation** — equivalently `RepetitionDetector(threshold≈0.97)`. It cleanly captures the true signal (verbatim re-sends = the assistant didn't act) without the slot-variant false positives, and feeds the conversation-anatomy features as the implicit-frustration marker (sparse but real, ~4% of conversations).

**Revisit trigger:** on real (non-synthetic) data where users *paraphrase*, switch back to embedding cosine at ~0.80 — and add a guard (only flag when an assistant turn intervenes between the two similar user messages) to suppress the slot-variant false positives.

---

# Interesting Stuff

*A running log of non-obvious findings worth surfacing in the final write-up — the "how we think" evidence, not just the results.*

## 1. Zero-shot intent accuracy is driven by label *wording* and the hypothesis template, not the model

The most surprising early finding: with a fixed model (`bart-large-mnli`) and fixed inputs, **just rewording the candidate labels and the hypothesis template flips the predictions** — often more than switching models would.

**The trigger.** Three messages that are unambiguously *creation* requests to a human (all provide a script/brief for a new video):

1. *"Here's my script: This is an introduction to leadership development…"*
2. *"I have prepared a full brief. Script: … Visual descriptions: … Tone: professional."*
3. *"Create a video with: This is an introduction to customer success story…"*

Off-the-shelf bart called these **Edit-content, Other, Create** — two of three wrong. Embedding cosine similarity was no help either: picking msg 2 as a gold example and scoring msg 1 against it gave ≈ **0.5** — useless — because MiniLM embeds by *topic/style*, not *intent*, so two same-intent briefs on different topics land far apart. (This is why few-shot k-NN on embeddings is a weak fallback for intent here.)

**Why it fails.** These are *implicit* creation requests — dominated by declarative script prose with **no imperative verb** ("make/create"). A surface-lexical matcher sees text-that-looks-like-existing-content and guesses Edit/Other. The creation intent lives in the framing, which bare one-word labels don't convey.

**The fix — measured, not asserted.** On a 10-message probe we compared three configs:

| Config | Labels | Template | The 3 implicit-creation msgs |
|--------|--------|----------|------------------------------|
| A | bare (`Create`, `Edit`, …) | `"This example is {}."` | Edit, Edit, Create ❌ |
| B | noun-phrase descriptions | `"This example is {}."` | Create, Approval, Approval ❌ |
| C | verb-phrase descriptions | `"The user is {}."` | **Create, Create, Create ✅** (conf 0.71–0.78) |

Descriptions *alone* (B) didn't fix it — they sent messages to `Approval`. It was descriptions **plus** the verb-phrase template `"The user is {}."` (C) that worked, and it also sharpened confidence (0.3–0.4 → 0.7+).

**Caveats we're honest about.**
- Overall accuracy on the 10-message probe was flat (60% → 70%); at that size that's ±1 message — *noise*. Config C fixed the creation cases but *broke* a vague request ("make a video about X" → Edit). The errors moved, they didn't vanish. We proved the *mechanism*, not the final numbers.
- The `Approval` description is a **magnet** for content-rich text (it swallowed a complaint, "that's incorrect…"). Its wording needs tightening.

**Implication for the method.** Label + template engineering is a first-class tuning task, and the validation gold set is used as a **tuning harness** (iterate wording, measure), not just a pass/fail gate. `IntentPipeline` now exposes `hypothesis_template` and per-stage label maps as parameters precisely so this is tunable without touching code.

**The harness in action — the "Approval magnet" fix.** On a 12-message gold set (3 each of Create/Edit/Approval/Dissatisfaction, hand-labelled) the refined pipeline scored **83% (10/12)**. Both errors were the *same* leak: content-rich messages ("Script: … I'm proud to share these results", "make the script more concise") predicted as **Approval**. Diagnosis: the label "expressing approval or **satisfaction with the result**" literally entails text containing "results"/"proud". Tightening it to "confirming the video looks good and approving it as final" took the gold set to **100% (12/12)** — kept all real approvals, killed both leaks. (Caveat: 12 examples; 100% is directional, to be reconfirmed on a larger gold set.) This is exactly the wording-sensitivity thesis, now with a fix driven by measurement rather than intuition.

## 2. Taxonomy refinement: `Correction` → `Dissatisfaction`, and `Other` is rule-based

`Correction` overlapped conceptually with `Edit` (a correction *is* a kind of edit), which muddied both the labels and the zero-shot signal. We renamed it to **`Dissatisfaction`** to capture the *sentiment* ("the assistant got it wrong / did something unwanted") rather than the *action*, keeping it cleanly distinct from a neutral `Edit`. Final Stage-1 taxonomy: **`Create`, `Edit`, `Approval`, `Dissatisfaction`, `Other`**.

**`Other` turned out to be near-empty — a finding in itself.** Scanning the distinct-message vocabulary (6,862 user messages → only 1,159 distinct; heavily templated), *every* real message maps to one of the four intents. There is no off-topic chit-chat. The only non-mapping content is **blank/nonsensical messages (144 empty user messages)** — a data-quality signal for Part 1, not an intent. So `Other` is **not a model class**: the zero-shot model chooses among the four real intents, and blank/nonsensical messages are routed to `Other` by an explicit rule. (Bonus: removing the vague `Other` from the candidate labels also stopped it absorbing borderline real messages.)

## 3. Gold-set evaluation: 93%, and the overfitting wall

The gold set (`data/processed/gold_intents.csv`) is 15 hand-labelled messages: `Create ×3`, `Edit content ×3`, `Edit visual ×3`, `Approval ×3`, `Dissatisfaction ×3`. `Other` is excluded (it's rule-routed in code, not something we tune on). We score **fine-grained exact-match** accuracy (`predict()` returns `intent` + `intent_confidence`; the coarse bucket is derivable by mapping `Edit content`/`Edit visual` → `Edit`).

**The tuning journey** (each step measured against this set, not asserted):
- Tightening the `Approval` wording (removing "…satisfaction with the result", which entailed content-rich text) took it **83% → 100%**.
- Later relabelling of `Create`/`Edit`/`Dissatisfaction` for broader-corpus coverage nudged one message off, settling at **93% (14/15)**.

**The one residual error is instructive — a genuine three-way boundary case:**
> `msg_012153`: *"make the script more concise - it's running too long"* — gold `Edit content`.

It carries cues for **three** classes: a complaint ("running too long" → `Dissatisfaction`), a creation verb+noun ("**make** the **script**" → `Create`), and the true intent (shorten existing script → `Edit`). We hit the **overfitting wall** chasing it: tightening `Dissatisfaction` stopped the complaint-leak but the message fell to `Create`; sharpening `Create`/`Edit` ("new from scratch" vs "change existing") recovered it but then **broke a clean full-brief `Create` example**. On a 15-row in-sample set, fixing one borderline case just relocates the error. We kept the tightened `Dissatisfaction` (a real precision gain, and all 3 dissatisfaction examples stay correct) and accepted the single boundary error rather than overfit further.

**Why even 93% overstates real accuracy:**
1. **In-sample tuning.** Label wording was tuned on this same set, so tuning-set = test-set.
2. **Easy examples by design.** These are unambiguous single-type messages. The corpus has harder cases this set omits — especially **mixed edits** ("change the avatar *and* the tone"), which single-label argmax must get partly wrong.

**Scoring it honestly (next step if time allows):** expand to ~40–50 labelled messages, tune on one split and report on a **held-out** split (or k-fold), and deliberately include mixed/borderline cases. If the edit split tanks on mixed edits, that triggers the agreed fallback (`collapse_edit_split=True` → single `Edit`).

## 4. The data is synthetic / template-generated — assume-real, flag caveats

Strong evidence the dataset is generated, not organic:
- **6,862 user messages → only 1,159 distinct.** The exact sentence *"use a different layout for the data slides"* appears **90 times across 64 distinct users**; *"that's not what I asked for"* across 39 users. Humans don't independently type byte-identical sentences at that rate.
- **Recurring fictional entities** — companies (Zenith Group, Horizon Financial, NovaTech, Summit Insurance) and avatars (Sam, Casey, Riley, Drew) drawn from a small fixed pool.
- **Slot-filling structure** — "switch to avatar {NAME}", "I need a video on {TOPIC}", "the script should mention {NUMBER}" — a crude slot-normalisation already collapses 1,159 → ~850, and the true template set is smaller.

**Operating stance (decided): proceed *as if the data were real* — analyse it as intended — but flag the caveats.** Synthetic data is common when the underlying conversations are privacy-sensitive; the analysis is about method and judgment, and noticing/handling this correctly is part of that. We do **not** try to reverse-engineer or exploit the generator.

**Caveats this imposes on every downstream claim:**
1. **"The data tends to…", not "users tend to…"** — frequencies reflect the generator, not organic behaviour.
2. **Non-independence → effective sample size is much smaller than the row count.** 64 users emitting one identical sentence is not 64 independent observations. Dedup for correct denominators; treat significance tests as optimistic.
3. **Outcomes may be synthetic-by-construction (Part 3 watch-out).** If messages are templated, outcomes (`published`) may be too — a strong "intent pattern → published" signal could be baked into the generator rather than a real behavioural insight. **Check whether outcomes cluster suspiciously by template**, and frame findings as "patterns in this dataset," not "laws of user behaviour."
4. **The scale rationale for NLP is weak (be honest).** With ~1,159 distinct messages we could label the vocabulary directly; the zero-shot pipeline is a *method demonstration* that would scale, not a scale necessity here.

---

# Success Metric — Empirical Decision

Part 2 argued for **Published** on first principles; the Part-1 funnel confirms it with data.

## The funnel (nested, no violations)
```
created 100%  →  generated 68%  →  published 37%  →  downloaded 14%
```
`downloaded ⊆ published ⊆ generated` holds exactly (0 nesting violations), so `published_given_downloaded = 100%` is a true nested funnel, not a coincidence.

## Verdict: **`published`** as the primary success metric

Evaluated against *"did the assistant genuinely help accomplish something?"*:

| Signal | Rate | Read | As success metric |
|--------|------|------|-------------------|
| **Generated** | 68% | Near-mechanical, fast (~1–2 min) — the assistant produced *a draft*. | ❌ Too low a bar; a generated-then-abandoned draft isn't success. |
| **Published** | 37% | A deliberate, days-later *"good enough to share"* decision. | ✅ Best validity **and** ample volume (~700 / 1,885) for segment/intent analysis. |
| **Downloaded** | 14% | Strict subset of published; sparse, confounded by share-link vs download. | ❌ Too sparse/noisy; adds little beyond published. |

**Why the distribution confirms it:** the biggest, most meaningful leak is **generated → published — only 55% of drafts ever get published.** That gap *is* the assistant's value proposition (a draft exists, but the user didn't judge it good enough to share), and `published` is exactly the label that captures that decision. Generation is a necessary floor; download is a stronger-but-sparse echo of the same signal.

## Two distinct failure modes — analyse separately
`published`-vs-not lumps together two failures with different root causes and fixes:
1. **Never generated (~32%)** — the assistant produced *nothing*; likely vague/broken opening requests → clarifying-questions / intent-parsing fixes.
2. **Generated but not published (~31%)** — a draft was made but rejected → edit-precision / brief-completeness fixes.

Model `published` (binary, at the 1:1 conversation/video level) as the headline target, but break out these two leaks — they'll have different intent/anatomy signatures.

## Caveats from the distribution
1. **Timing is synthetic-looking** — publish/download times are ~uniform over ~2 weeks (real behaviour would be right-skewed). Use the **event** (published yes/no), not *time-to-publish*, for behavioural claims (see Interesting Stuff #4).
2. **26 videos have `published_at < created_at`** (published before created — impossible; source of the negative time-to-publish lower bound), plus 3 `downloaded < published`. This is timestamp noise: the **binary `published` label is unaffected**, but any **time-to-X duration feature must null out those 26 negatives**.

## Bottom line
**Primary = `published` (binary).** Keep `generated` as a necessary-condition floor + the "never-generated" failure cut, and `downloaded` as an optional robustness check. Avoid a composite (clean nesting means it adds complexity without new signal); if a richer target is ever wanted, a **funnel-depth ordinal** (0/1/2/3) is the natural option.
