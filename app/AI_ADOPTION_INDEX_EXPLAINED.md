# How the AI Adoption Index Is Calculated

*A plain-language guide for readers with an economics background — no statistics
or programming required.*

This document explains, step by step, how the tool turns a pile of company
reports into a single **AI Adoption Index** score, and what every number behind
that score means. Each concept is given first in words, then as a formula, then
with a small worked example. A glossary of the statistical terms used is at the
end.

---

## 1. The big picture

Think of the index like a **composite economic indicator** — the way a
"competitiveness index" or a "cost-of-living index" blends several measured
components into one comparable number.

For each company in each year, the tool measures **seven separate things** (call
them sub-indices), each scaled to sit between 0 and 1. It then takes a **weighted
average** of the seven and multiplies by 100, giving a headline score from
**0 to 100**.

```
AI Adoption Index =
    ( w₁·Intensity + w₂·Semantic + w₃·Diversity + w₄·Sentiment
    + w₅·Maturity  + w₆·Future   + w₇·Commitment ) × 100
```

The weights are fixed and sum to 1 (100%):

| Sub-index | Weight | What it captures |
|-----------|:-----:|------------------|
| Intensity | 0.15 | How densely AI is discussed (per page) |
| Semantic | 0.20 | How strongly the language actually resembles AI topics |
| Diversity | 0.15 | How many different *kinds* of AI use are mentioned |
| Sentiment | 0.15 | Whether AI is discussed positively or negatively |
| Maturity | 0.15 | How advanced the described adoption is |
| Future | 0.10 | How forward-looking / planned the AI activity is |
| Commitment | 0.10 | How concrete the pledges (money, strategy, people) are |

Because it is a weighted average of seven 0-to-1 numbers, the index is itself
between 0 and 100. A score near 0 means almost no credible AI content; a score
near 100 would require a company to be dense, diverse, advanced, positive,
forward-looking and concrete about AI all at once — deliberately demanding.

> **Two scores, not one.** The index is computed **twice**: once using the
> "classic" business taxonomy and once using the European Commission's
> EU capability taxonomy. The only ingredient that differs between the two is
> **Diversity** (see §4); the other six sub-indices are identical. So you get an
> `AI Adoption Index` and an `AI Adoption Index (EU)` that differ only in how
> "variety of AI use" is measured.

---

## 2. The building block: a *confidence-weighted* count

Before the sub-indices, one idea runs through everything: **not every mention of
AI counts equally.**

A vague, marketing-style sentence ("we are excited about the future of AI")
should count for far less than a concrete one ("we deployed a machine-learning
fraud-detection system across all branches"). So every detected mention carries a
**confidence score** between 0 and 1 (how that score is built is explained in
§9). Throughout the index, mentions are **weighted by their confidence** rather
than simply counted.

In economic terms: instead of a simple headcount, the tool uses a
**quality-weighted count**, the way you might weight survey responses by
reliability. (Technically, a mention with no usable confidence is given a floor
weight of 1.0 so it is never silently dropped, and the very weakest mentions —
"mention only" — are excluded from the index entirely but kept on file for audit.)

---

## 3. Sub-index: Intensity — *how much AI talk, per page*

**In words:** How concentrated is the AI content, relative to the length of the
document? We divide by page count so that a long report doesn't automatically
look more "AI-heavy" than a short one.

**Formula:**

```
Intensity = min( 1 , (sum of all mentions' confidence) ÷ total_pages × 10 )
```

The `× 10` rescales a typical density into the 0–1 range; `min(1, …)` caps it at
1 so a few very dense pages can't push the score above the ceiling.

**Worked example:** A 50-page report has mentions whose confidence scores add up
to 4.0. Intensity = 4.0 ÷ 50 × 10 = **0.80**. A 200-page report with the same
total (4.0) scores 4.0 ÷ 200 × 10 = **0.20** — same amount of AI talk, but much
more diluted.

---

## 4. Sub-index: Diversity — *how many kinds of AI use*

**In words:** Does the company touch many different areas of AI (strategy,
operations, customer service, computer vision, …), or is everything concentrated
in one bucket? Spreading across many categories scores higher.

This is the one sub-index that uses a concept worth pausing on: **Shannon
entropy**, a standard measure of "spread" or "evenness" (closely related to how
economists use the inverse of a concentration index like Herfindahl).

**Formula:**

```
For each category i, let pᵢ = its confidence-weighted share of all mentions.
entropy   = − Σ  pᵢ × log₂(pᵢ)
Diversity = entropy ÷ log₂(number of categories)
```

Dividing by `log₂(number of categories)` rescales the result to 0–1, where:
- **0** = everything is in a single category (maximum concentration),
- **1** = mentions are spread perfectly evenly across all categories.

**Worked example:** Suppose (after confidence-weighting) 100% of a company's
AI mentions fall in one category. Then entropy = 0, so Diversity = **0**. If
instead they are split evenly across 4 categories (25% each), entropy =
2 bits, and with, say, 15 possible classic categories, Diversity =
2 ÷ log₂(15) = 2 ÷ 3.91 ≈ **0.51**.

**Classic vs EU:** The classic score divides by `log₂(15)` (7 applications + 8
technologies); the EU score groups mentions by EU subdomain and divides by
`log₂(12)`. Mentions that match no EU category are simply ignored in the EU
version. This is the *only* difference between the classic and EU index.

**Both classic axes count.** Because classic classification is now dual-axis,
each mention contributes its confidence weight to *two* buckets — its
Applications category (`category_a`) and its Technologies category
(`category_b`). An axis equal to `none` (no match) contributes nothing for that
axis. A mention that is, say, "customer-service + generative-AI" therefore adds
spread on both the A and B sides, which the entropy rewards.

---

## 5. Sub-index: Semantic — *how strongly the text "reads as" AI*

**In words:** Beyond keyword matching, the tool measures how closely each mention
resembles genuine AI language, using a similarity score from 0 to 1 (see
*cosine similarity* in the glossary). Semantic is the **confidence-weighted
average** of those similarity scores, counting only mentions above a quality
threshold (0.60).

**Formula:**

```
Semantic = confidence-weighted average of the similarity scores
           of all mentions whose similarity > 0.60
```

**Worked example:** Two qualifying mentions, with similarity 0.70 (confidence 1.0)
and 0.90 (confidence 2.0). Weighted average = (0.70×1.0 + 0.90×2.0) ÷ (1.0+2.0)
= 2.5 ÷ 3.0 ≈ **0.83**.

---

## 6. Sub-index: Sentiment — *positive or negative framing*

**In words:** Is AI discussed as an opportunity or as a threat? A sentiment
model rates each mention from −1 (very negative) to +1 (very positive). We take
the confidence-weighted average and then **rescale** it onto 0–1, so that purely
neutral coverage lands at 0.5.

**Formula:**

```
mean      = confidence-weighted average sentiment   (between −1 and +1)
Sentiment = (mean + 1) ÷ 2                           (now between 0 and 1)
```

**Worked example:** A weighted-average sentiment of +0.4 becomes
(0.4 + 1) ÷ 2 = **0.70**. Neutral (0) becomes 0.5; fully negative (−1) becomes 0.

*(If a mention sits in risk-factor or disclaimer language — "AI may expose us to
liability" — it is caught earlier by the negation filter and heavily
down-weighted, so such defensive language does not masquerade as adoption.)*

---

## 7. Sub-index: Maturity — *how advanced the adoption is*

**In words:** Is the company merely *exploring* AI, *piloting* it, or *scaling*
it across the business? Each mention is matched to a maturity stage, and the
sub-index is the **confidence-weighted average stage score** across all mentions.

Using an average (rather than taking the single most advanced stage found
anywhere) means one stray word like "scaling" cannot pin the whole company at the
top — you get a smooth, representative picture suitable for tracking change over
time.

**Worked example:** If most mentions describe pilots (a middling stage score)
and a few describe full deployment (a high stage score), the average lands
between the two, tilted by how confident and frequent each is.

---

## 8. Sub-indices: Future & Commitment — *plans and pledges, per page*

These two reward concrete forward intent, and like Intensity they are
**normalized per page** so that longer reports don't score higher just for being
longer. (Reports have grown longer over time, so per-page rates keep scores
comparable across years.)

**Future — forward-looking language** ("we will deploy…", "by 2027 we aim to…"):

```
Future = min( 1 , (confidence-weighted count of forward-looking mentions)
                  ÷ total_pages × 30 )
```

The `× 30` means roughly one solid forward-looking mention per 30 pages already
moves the needle meaningfully; `min(1, …)` caps it at 1.

**Commitment — concrete pledges** (named budgets, "AI is a strategic priority",
hiring an AI team). Different pledge types carry different weights (a dollar
figure counts more than a generic statement):

```
Commitment = min( 1 , (confidence-weighted, pledge-type-weighted count)
                      ÷ total_pages × 2.5 )
```

**Worked example (Future):** 60-page report, confidence-weighted forward-looking
count = 3. Future = 3 ÷ 60 × 30 = 1.5 → capped to **1.0**. A 300-page report with
the same count: 3 ÷ 300 × 30 = **0.30**.

---

## 9. The per-mention confidence score (the quality weight)

This is the number that powers the "weighted count" everywhere above. Each
mention starts from a **base score set by how it was found**, then earns
**bonuses** for substance and **penalties** for fluff. The result is clamped to
0–1.

> **Full detail:** the exact trigger vocabularies, the gating rules for
> penalties, how negation overrides the score, and the audit fields are
> documented in the companion file **`CONFIDENCE_SCORE_EXPLAINED.md`**. The
> summary below is the essential version.

**Base score (by trigger type):**

| How the mention was detected | Base |
|------------------------------|:----:|
| "Hard" trigger (specific, technical AI term) | 0.72 |
| Semantic match (text resembles AI language, similarity *s*) | 0.35 + *s* × 0.55 |
| "Soft" trigger (generic / weaker term) | 0.35 |

**Bonuses (added):**
- +0.12 — implementation language present ("deployed", "in production")
- +0.10 — technical specificity (named systems, models, tools)
- +0.08 — a hard AI term also appears nearby in the context

**Penalties (subtracted):**
- −0.18 — marketing-only language with no substance signal
- −0.12 — ESG/boilerplate framing with no substance signal

**Then:** clamp to the 0–1 range and assign a **strength label**:

| Final confidence | Strength | Used in the index? |
|------------------|----------|--------------------|
| ≥ 0.70 | strong | yes |
| 0.55 – 0.69 | medium | yes |
| below 0.55 | mention-only | **no** (kept for audit only) |

**Worked example:** A hard-trigger mention (0.72) that also shows implementation
language (+0.12) and technical specificity (+0.10) scores 0.94 → **strong**. A
soft-trigger mention (0.35) buried in marketing language (−0.18) scores 0.17 →
**mention-only**, so it is recorded but excluded from the index.

**Why this matters economically:** it makes the index robust to "cheap talk."
Companies cannot inflate their score simply by repeating AI buzzwords; the score
responds to *substantive, specific, implemented* AI activity.

---

## 10. How a mention gets its category (classification confidence)

Separately from the quality weight above, each mention is sorted into a category.
The classic taxonomy is scored on **two independent axes** — Applications
(`category_a`) and Technologies (`category_b`) — and the EU taxonomy in parallel.
The tool scores how well the surrounding text matches each category's vocabulary:

```
category score = 0.5 × (number of distinct patterns matched)
               + 0.2 × (number of distinct keywords matched)
classification confidence = min( 1 , category score ÷ 1.5 )
```

The highest-scoring category on each axis wins. Three practical points:
- It counts **distinct** matching terms, not repeats — saying "machine learning"
  five times adds no more than saying it once.
- Each **classic** axis requires a minimum score (0.25) to accept a category;
  below that it is `none` — there is **no fallback**, so a mention may be an
  Application only, a Technology only, both, or neither.
- The **EU** taxonomy also has no minimum — any single keyword match assigns a
  category, otherwise the mention is left "Unclassified" in the EU system. A
  mention that is `none` on Applications, Technologies *and* EU is flagged by an
  integrity check (it should not normally happen for a genuine AI reference).

For the EU taxonomy (which is keyword-driven), this produces a simple ladder:

| Distinct EU keywords matched | EU classification confidence |
|:---:|:---:|
| 1 | ≈ 0.13 |
| 2 | ≈ 0.27 |
| 3 | ≈ 0.40 |
| 5 | ≈ 0.67 |
| 8 or more | 1.00 |

---

## 11. Putting it together — a miniature example

Suppose for *Company X, 2024* the seven sub-indices come out as:

| Sub-index | Value | Weight | Contribution |
|-----------|:----:|:----:|:----:|
| Intensity | 0.60 | 0.15 | 0.090 |
| Semantic | 0.80 | 0.20 | 0.160 |
| Diversity | 0.50 | 0.15 | 0.075 |
| Sentiment | 0.70 | 0.15 | 0.105 |
| Maturity | 0.55 | 0.15 | 0.0825 |
| Future | 0.40 | 0.10 | 0.040 |
| Commitment | 0.30 | 0.10 | 0.030 |
| **Sum** | | **1.00** | **0.5825** |

**AI Adoption Index = 0.5825 × 100 ≈ 58.3.**

If Company X's AI mentions were *more varied* under the EU taxonomy than under the
classic one, its `AI Adoption Index (EU)` would be slightly higher — and that
gap is entirely due to the Diversity term.

---

## 12. Statistical glossary

- **Weighted average (weighted mean).** An average where some items count more
  than others. Instead of `(a + b)/2`, you use `(wₐ·a + w_b·b) / (wₐ + w_b)`.
  Here the weights are mention confidences — reliable mentions pull the average
  toward themselves.

- **Normalization / capping.** Rescaling a raw number onto a fixed range (here
  0–1). "Capping" with `min(1, …)` simply prevents any component from exceeding
  its ceiling, so no single dimension can dominate the composite.

- **Shannon entropy.** A measure of how *evenly spread* something is across
  categories. It is 0 when everything is in one category and largest when items
  are split evenly across all of them. Economists will recognise it as a
  diversity/evenness measure — conceptually the opposite of a concentration index
  such as the Herfindahl-Hirschman Index. We divide by its maximum possible value
  so the result is a clean 0–1 "evenness" score.

- **Cosine similarity.** A 0-to-1 measure of how alike two pieces of text are in
  meaning (not just shared words). The tool converts each sentence into a list of
  numbers ("an embedding") and measures the angle between them; closer angle =
  higher similarity = the text "reads more like" genuine AI discussion.

- **Confidence score.** A 0-to-1 quality weight for a single AI mention, built
  from how it was detected plus substance bonuses and fluff penalties (§9). It is
  the weight used in every confidence-weighted average above.

---

*All formulas above are drawn directly from the code: the sub-indices and the
composite live in `_04_analysis.py`; the per-mention confidence and the
category-classification scoring live in `_03_detection.py`; the dimension weights
live in `_02_core.py`.*
