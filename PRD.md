# PRD — Credit Risk Threshold Optimizer

**Elevator pitch:** A portfolio project that turns a credit-default classifier into an
actual business decision — finding the dollar-optimal loan approval threshold instead
of defaulting to 0.5, and explaining who gets declined and why in plain English.

**Suggested repo name:** `credit-risk-threshold-optimizer`
**Suggested one-line GitHub description:** "Where should a lender draw the approval
line? A cost-sensitive threshold analysis for credit default risk, with SHAP
explanations in plain English."

---

## 1. Background / problem

Most credit-risk portfolio projects stop at "I trained a model and got 0.78 AUC."
That proves you can call `.fit()`. It doesn't prove you understand that a probability
isn't a decision — someone still has to pick the cutoff that turns "12% chance of
default" into "approve" or "decline," and almost nobody's portfolio shows that step
done deliberately instead of defaulted to 0.5.

This project closes that specific gap: build a reasonable baseline model, then treat
the *threshold* — not the model — as the thing worth optimizing, in real dollars, and
make the result explainable enough that a non-technical stakeholder could sign off on
it.

## 2. Goals

- Demonstrate real data-cleaning judgment on a genuinely messy, well-known dataset
  (not a from a toy/synthetic-only source).
- Ship a cost-sensitive decision layer as the clear centerpiece — the part almost no
  competing portfolio has.
- Make the model's decisions explainable in plain language, not just accurate.
- Produce a repo a hiring manager can open and understand the point of within ~60
  seconds via the README, and can optionally reproduce line-for-line.

## 3. Non-goals

- Not chasing leaderboard accuracy / maximum AUC. The differentiator is the decision
  layer, not model performance.
- Not building a production scoring service, API, or deployed app.
- Not using real underwriting economics — cost inputs are clearly-labeled illustrative
  assumptions, not a real bank's numbers.

## 4. Target audience

| Audience | What they need from this repo |
|---|---|
| Recruiter skimming GitHub | README tells the story in under a minute, one clear plot |
| Technical interviewer | Code is clean enough to survive "walk me through this" |
| Future you | Enough structure to extend it later without re-learning it |

## 5. Requirements

### Functional (what the MVP must do)
1. Load and clean a real, publicly available credit application dataset.
2. Handle at least one authentic real-world data quirk visibly (e.g. a sentinel/
   placeholder value in a numeric column) rather than silently dropping or ignoring it.
3. Train a baseline default-risk classifier with a proper train/test split.
4. Sweep approval thresholds and compute total dollar cost at each one, using two
   named, configurable cost assumptions (cost of an approved default vs. cost of a
   wrongly rejected good applicant).
5. Identify and visualize the cost-minimizing threshold against a naive 0.5 baseline.
6. Produce a model explanation (global top drivers + at least one individual,
   plain-English "why this applicant" example).
7. README that documents: the business question, the data and its limitations, the
   approach and why, and the results and their limitations.

### Non-functional
- **Reproducible** — a stranger clones the repo and gets the same numbers.
- **Runs without gated access** — works on a small bundled sample with no Kaggle
  login required; real data is an opt-in upgrade, documented in the README.
- **Legible commit history** — the cleaning step should be visible in the log, not
  squashed into one "add project" commit.

## 6. Technical approach (system design)

- **Data:** Home Credit Default Risk (Kaggle) as the primary choice — richer and more
  commonly benchmarked than Lending Club, so there's a lot of prior art to sanity-check
  results against. Lending Club is a fine drop-in alternative if preferred.
- **Cleaning:** fix at least the `DAYS_EMPLOYED` sentinel-value anomaly (a well-known
  quirk of this dataset — some rows encode "not employed" as a nonsense day-count
  instead of a missing value), plus standard imputation *fit on the training split
  only* to avoid leakage.
- **Model:** a single, interpretable-enough baseline (tree ensemble) — model choice is
  intentionally not the focus, and using something well-supported by SHAP matters more
  than squeezing out extra AUC.
- **Cost layer:** threshold swept across its full range; at each point, sum
  `(defaults approved × loss-given-default × loan amount)` and
  `(good applicants rejected × expected margin × loan amount)`. Plot both curves and
  their sum; the minimum is the recommended threshold.
- **Explainability:** SHAP values off the trained model — one global ranking (top
  drivers overall) and one local explanation (top drivers for a specific declined
  applicant), each written back out as a sentence, not just a chart.
- **Presentation:** a notebook that narrates the above end to end with embedded plots,
  plus a README that stands alone without requiring anyone to open the notebook.

## 7. Success metrics

- **Technical:** baseline model beats a random/majority-class baseline by a clear
  margin (rough target: test ROC AUC ≥ 0.68–0.70 on a single-table baseline).
  Optimal threshold shows a measurable $ improvement over the naive 0.5 cutoff.
- **Narrative:** someone with no ML background can read the README and correctly
  explain, in their own words, why the threshold isn't just 0.5.
- **Portfolio:** the project title reads as a question, not a dataset name, and the
  first thing visible on the repo is the cost-curve plot, not a wall of code.

## 8. Suggested build order (milestones)

| # | Milestone | Done when |
|---|---|---|
| M1 | Repo skeleton + data understanding | Raw data loaded, schema and quirks documented in a scratch notebook |
| M2 | Cleaning + baseline model | Train/test split working, baseline AUC recorded |
| M3 | Cost curve | Threshold sweep plotted, optimal threshold identified and sanity-checked |
| M4 | Explainability | Global + one local SHAP explanation, translated to plain English |
| M5 | README + polish | Someone outside the project can follow it start to finish |

## 9. Risks / open questions

- Real Kaggle data requires a free account + API token — document this clearly so it's
  a two-minute setup, not a blocker.
- Cost assumptions (loss-given-default %, expected margin %) are subjective by
  necessity — the README should defend the chosen numbers explicitly rather than
  presenting them as fact.
- Class imbalance (defaults are a small minority) needs deliberate handling
  (class weighting or resampling) or the model will just learn to predict "no
  default" for everyone and look falsely accurate.

## 10. Out of scope for v1 (future work)

- Join in the auxiliary tables (`bureau.csv`, `previous_application.csv`) for a richer
  feature set and higher AUC.
- Compare against a gradient-boosted model (LightGBM/XGBoost).
- A small interactive demo (e.g. Streamlit) where a viewer can drag the threshold
  slider and watch both cost curves move live.
