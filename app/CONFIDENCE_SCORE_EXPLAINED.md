# How the Confidence Score Is Calculated

*A focused, code-accurate reference for the per-mention **confidence score** — the
0-to-1 quality weight attached to every detected AI mention.*

This is the number that makes the whole index robust to "cheap talk": every
confidence-weighted average in the AI Adoption Index (see
`AI_ADOPTION_INDEX_EXPLAINED.md`, §2 and §9) uses it as the weight. This document
gives the exact formula, the vocabularies behind each signal, how negation
interacts with it, and how it flows downstream.

> **Two different "confidences" — don't confuse them.**
> - **Detection confidence** (this document): *how much do we trust that this
>   mention is a real, substantive AI reference?* Field: `confidence_score`.
> - **Classification confidence** (`AI_ADOPTION_INDEX_EXPLAINED.md`, §10): *given
>   it is an AI reference, which category does it belong to?* Fields:
>   `category_a_confidence`, `category_b_confidence`, `eu_confidence`.
>
> They are computed independently. Everything below is about **detection
> confidence**.

All logic lives in `_03_detection.py`, method
`AIReferenceDetector._compute_confidence_and_strength()`.

---

## 1. The formula in one line

```
confidence = clamp_0_1( base(trigger) + bonuses − penalties )
```

The score starts from a **base set by how the mention was detected**, gains
**bonuses** for signals of substance, loses **penalties** for empty framing, and
is finally clamped into `[0, 1]`. A **strength label** is then derived from the
clamped value.

---

## 2. Base score — by trigger type

Every mention is found by one of three trigger types. The base reflects how
inherently trustworthy that trigger is.

| Trigger | Base score | Reason tag |
|---------|:----------:|------------|
| **hard** — a specific, technical AI term | `0.72` | `hard_trigger` |
| **semantic** — text *resembles* AI language (embedding similarity `s`) | `0.35 + s × 0.55` | `semantic=<s>` |
| **soft** — a generic / weaker AI term | `0.35` | `soft_trigger` |

For a semantic match, `s` is the cosine similarity (clamped to `[0, 1]`), so the
base ranges from `0.35` (s = 0) up to `0.90` (s = 1).

**What counts as "hard" vs "soft":**
- **Hard patterns** (`AI_HARD_PATTERNS`) — unambiguous, technical terms:
  `generative AI`, `LLM`, `GPT`, `foundation model`, `transformer`, `RAG`,
  `vector database`, `embeddings`, `fine-tuning`, `RLHF`, `LoRA`, `MLOps`,
  `model serving/inference`, named tools (`GitHub Copilot`, `Claude`, `Gemini`,
  `Llama`, `LangChain`, `AutoGPT`), `function calling`, `tool use`, etc.
- **Soft patterns** (`AI_SOFT_PATTERNS`) — vague or marketing-prone terms:
  `artificial intelligence`, `machine learning`, `AI`, `ML`, `AI-powered`,
  `AI-driven`, `chatbot`, `automation`, `intelligent`, `predictive analytics`,
  `RPA`, etc.

---

## 3. Bonuses — signals of substance

Each bonus is added **once** if at least one matching cue appears in the
mention's surrounding context window.

| Bonus | Amount | Reason tag | Trigger vocabulary (`_feature_counts`) |
|-------|:------:|------------|----------------------------------------|
| Implementation language | `+0.12` | `implementation_signal` | `IMPLEMENTATION_VERBS`: deployed, implemented, integrated, roll-out, launched, building, scaled, production, inference, serving |
| Technical specificity | `+0.10` | `technical_specificity` | `TECHNICAL_ARTIFACTS`: model, pipeline, training, fine-tuning, embeddings, vector, tokens, prompt, guardrails, evaluation, benchmark, data scientist, ML engineer |
| Hard term in context | `+0.08` | `hard_in_context` | any `AI_HARD_PATTERNS` term appears in the context (even if the mention itself was triggered softly) |

These are cumulative: a mention can earn all three (`+0.30` total).

---

## 4. Penalties — empty framing, *only when unsubstantiated*

Penalties fire **only if none** of the three substance signals above are present
(no implementation verb, no technical artifact, no hard term in context). This
gating is deliberate: a sentence that mentions "strategy" but also says
"deployed in production" is *not* penalized.

| Penalty | Amount | Reason tag | Condition |
|---------|:------:|------------|-----------|
| Marketing-only language | `−0.18` | `marketing_only_penalty` | a `MARKETING_ONLY_CUES` term (leverage, explore, aspire, commitment, vision, strategy, roadmap, principles, policy, framework, responsible AI, AI ethics, AI governance) is present **and** no substance signal |
| ESG / boilerplate framing | `−0.12` | `esg_boilerplate_penalty` | the mention is in an ESG-like context (`ESG_DOC_CUES`: sustainability, esg, csr, responsible, governance, ethics — in the doc type or context) **and** no substance signal |

Both can apply together (`−0.30` total) to a purely aspirational ESG sentence.

---

## 5. Clamp and strength label

After base ± bonuses ∓ penalties, the raw value is clamped:

```
confidence = max(0.0, min(1.0, raw))
```

and mapped to a **strength** label:

| Final confidence | Strength | Used in the index? |
|------------------|----------|--------------------|
| `≥ 0.70` | `strong` | yes |
| `0.55 – 0.69` | `medium` | yes |
| `< 0.55` | `mention_only` | **no** — recorded for audit only |

`mention_only` references are kept in the raw output (for transparency and manual
review) but **excluded from every index computation** (`_is_mention_only` in
`_04_analysis.py`).

---

## 6. Negation overrides the score

After scoring, `NegationFilter` inspects the mention's context
(`_03_detection.py`):

- **Hard negation** ("we do **not** use AI", "AI is **not** part of our
  operations") → the reference is **dropped entirely**.
- **Soft negation** (risk-factor / hypothetical framing — "AI **may expose** us
  to liability", "AI **could** introduce risks") → the reference is kept but:
  - `is_negated_context` set to `True`,
  - `confidence_score` multiplied by `SOFT_NEGATION_CONFIDENCE_FACTOR = 0.4`,
  - `reference_strength` forced to `mention_only` (so it leaves the index),
  - reason `negation_soft` appended.

This stops defensive / disclaimer language from masquerading as adoption —
important because AI risk-factor disclosure surged post-2022.

---

## 7. Audit trail

Every score is fully reconstructable from two stored fields:

- **`confidence_reasons`** — a `;`-joined list of the reason tags above
  (e.g. `hard_trigger;implementation_signal;technical_specificity`).
- **`detection_method`** — compact metadata encoding, e.g.
  `pattern|trigger=soft|strength=mention_only|conf=0.48|reasons=soft_trigger,marketing_only_penalty`.

Both appear in `ai_references_raw.xlsx` (the `Confidence`, `Strength`, and
`Reasons` columns), so any score can be traced term-by-term back to the text.

---

## 8. How the score is used downstream

- **Index weighting** — in `_04_analysis.py`, `_ref_weight(ref)` returns the
  `confidence_score`, with a **floor of 1.0** if it is missing/zero (so a scored
  reference is never silently dropped from a weighted average). Every sub-index
  that takes a mean or a count weights by this value.
- **Exclusion of weak mentions** — `mention_only` references are filtered out
  before the index is computed (§5).
- **Deduplication** — when near-duplicate mentions are merged, the cluster's
  representative is the **highest-confidence** member, and the merged row carries
  `avg_confidence_score` (the mean confidence across the cluster).

---

## 9. Worked examples

**A. Strong, substantive mention**
> "We **deployed** a fine-tuned **LLM**-based fraud-detection **model** in
> **production** across all branches."

- Trigger: hard (`LLM`) → base `0.72`
- Bonuses: implementation (`deployed`, `production`) `+0.12`; technical artifact
  (`model`, `fine-tuned`) `+0.10`; hard-in-context `+0.08`
- Penalties: none (substance present)
- Raw = `1.02` → clamp → **1.00** → **strong** ✅ (enters the index at full weight)

**B. Soft, marketing-only mention**
> "We are **committed** to a **responsible AI** **strategy** that will shape our
> **vision**."

- Trigger: soft (`AI`) → base `0.35`
- Bonuses: none
- Penalties: marketing-only `−0.18`; ESG-boilerplate `−0.12` (responsible /
  governance framing, no substance)
- Raw = `0.05` → **0.05** → **mention_only** ✅ (recorded, excluded from index)

**C. Semantic match, medium substance**
> Sentence with embedding similarity `s = 0.72`, mentioning a `pipeline`.

- Trigger: semantic → base `0.35 + 0.72 × 0.55 = 0.746`
- Bonus: technical artifact (`pipeline`) `+0.10`
- Raw = `0.846` → **0.85** → **strong**

**D. Soft mention caught by soft negation**
> "**AI** **could** expose the company to regulatory **risk**."

- Base soft `0.35`, no bonuses, marketing/ESG penalties may or may not apply →
  suppose raw ≈ `0.35`
- Soft negation → `× 0.4` = `0.14`, strength forced to `mention_only`,
  `is_negated_context = True` → excluded from the index

---

*Source of truth: `AIReferenceDetector._compute_confidence_and_strength()`,
`_feature_counts()`, the `AI_HARD_PATTERNS` / `AI_SOFT_PATTERNS` /
`IMPLEMENTATION_VERBS` / `TECHNICAL_ARTIFACTS` / `MARKETING_ONLY_CUES` /
`ESG_DOC_CUES` lists, and `NegationFilter` — all in `_03_detection.py`.
Downstream use lives in `_04_analysis.py` (`_ref_weight`, `_is_mention_only`,
deduplication).*
