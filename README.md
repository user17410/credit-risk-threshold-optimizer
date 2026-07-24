# Where Should a Lender Draw the Approval Line?

![Cost curve: total dollar cost vs. approval threshold, with the optimal threshold marked against the naive 0.5 cutoff](outputs/figures/cost_curve.png)

A credit-default model outputs a probability. It does not output a decision.
Somebody still has to pick the cutoff that turns "9% chance of default" into
"approve" or "decline" -- and most portfolio projects skip that step entirely,
defaulting to 0.5 because that's what `.predict()` uses internally. This
project treats the **threshold**, not the model, as the thing worth
optimizing: it prices both ways a lending decision can go wrong in real
dollars, sweeps the cutoff across its full range, and shows exactly where the
total cost is minimized -- then explains individual decisions in plain
English via SHAP.

## The problem

"I trained a model and got 0.74 AUC" proves you can call `.fit()`. It doesn't
prove you understand that someone still has to act on that probability. A
non-technical stakeholder (a VP of Risk, a compliance reviewer, the applicant
themselves) doesn't care about AUC -- they care about how many dollars a
given cutoff costs the business, and whether they can explain any single
decision without a statistics background. This project is built around
answering both of those, not around squeezing out extra accuracy.

## The data

**Primary dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(Kaggle), `application_train.csv` -- a single-table subset of a larger,
real, messy loan-application dataset with ~300k rows and a well-known set of
data quirks. Chosen over Lending Club because it's more commonly benchmarked,
so there's prior art to sanity-check results against.

**This repo's default run uses a bundled synthetic sample** (`data/sample.csv`,
5,500 rows), not the real file, so the whole pipeline runs with zero setup and
no Kaggle account. The synthetic data is built to mimic the real dataset's
known structure rather than being random noise:

- ~9% default rate, matching the real dataset's actual base rate.
- `EXT_SOURCE_1/2/3` (external bureau scores) are the strongest predictors of
  default, same as in the real data -- the synthetic target is a function of
  these three plus a debt-to-income term and noise.
- The real dataset's `DAYS_EMPLOYED` sentinel-value anomaly is reproduced
  deliberately (see below).
- `EXT_SOURCE_1` is 50%+ missing and `OWN_CAR_AGE` is missing whenever the
  applicant doesn't own a car, matching the real data's missingness pattern.

**Limitations of the data, stated plainly:**
- The synthetic sample is a stand-in, not the real thing. Real applicant
  behavior has correlations and edge cases no generator captures. Treat every
  number in this README as "what the pipeline produces on the bundled
  sample," not as a claim about real-world credit risk.
- Even the real dataset is a single table. It excludes bureau history,
  previous applications, and other joinable context that would likely
  materially change both the AUC and the shape of the cost curve.

## The approach, and why

1. **Clean deliberately, don't just impute and move on.** The real Home
   Credit dataset encodes "not currently employed" as the literal value
   `365243` in `DAYS_EMPLOYED` -- about 1,000 years of tenure -- instead of a
   missing value. Reproducing and then explicitly handling this quirk (flag
   it, then null it before any numeric use) is meant to demonstrate real
   data-cleaning judgment, not just a `.fillna()` call. Imputation for other
   missing values is fit on the training split only -- fitting on the full
   dataset first would leak test-set distribution into training.

   ![Left: bar chart showing 1,072 of 5,500 applicants carry the DAYS_EMPLOYED sentinel value before cleaning. Right: histogram of the legitimate YEARS_EMPLOYED distribution after the sentinel is flagged and nulled out.](outputs/figures/days_employed_anomaly.png)
2. **Use a model that's good enough, not maximal.** A `RandomForestClassifier`
   with `class_weight="balanced"` (defaults are a small minority of
   applicants, so an unweighted model just learns to predict "no default" for
   everyone and looks falsely accurate). RandomForest was chosen specifically
   because SHAP's `TreeExplainer` supports it cleanly across library
   versions -- model choice is deliberately not the differentiator here.
3. **Treat the threshold as the deliverable.** Sweep the approval threshold
   from 0.01 to 0.99. At each point, sum the dollar cost of both error types
   (below) on the test set. Plot both cost curves and their sum; the minimum
   is the recommended threshold.
4. **Explain decisions, not just accuracy.** SHAP values off the trained
   model: a global ranking of what drives risk overall, and a specific
   declined applicant's top factors translated into a paragraph a
   non-technical manager could read without a legend.

### Cost assumptions (illustrative, not real underwriting economics)

| Constant | Value | Meaning |
|---|---|---|
| `LOSS_GIVEN_DEFAULT` | 60% of loan amount | Cost when an **approved** applicant defaults |
| `EXPECTED_MARGIN` | 12% of loan amount | Forgone profit when a **good** applicant is wrongly declined |

Both live at the top of `src/cost_analysis.py`, clearly labeled. They are
reasonable, round, illustrative numbers chosen to make the cost curve's shape
legible -- they are **not** a real lender's actual loss-given-default or
margin figures, and the optimal threshold below is only as meaningful as
these two assumptions. Change them and the optimal threshold moves; that
sensitivity is the point, not a flaw.

## What we found (and its limitations)

On the bundled synthetic sample, with a fixed random seed throughout:

- **Baseline model:** ROC AUC = **0.743** on a held-out, stratified test
  split -- comfortably above a random baseline, using only the single
  `application` table.
- **Optimal threshold: 0.52**, vs. the naive 0.5 cutoff.
  - Total cost at the optimal threshold: **$42,157,848**
  - Total cost at the naive 0.5 threshold: **$44,582,772**
  - Total cost approving every applicant: **$56,520,780**
  - **Savings vs. naive 0.5: ~$2.4M. Savings vs. approving everyone: ~$14.4M**
    (on this test set, under the cost assumptions above).
- **Global SHAP ranking**, top 3 drivers of predicted risk: `EXT_SOURCE_2`,
  `EXT_SOURCE_3`, `EXT_SOURCE_1` -- the three external bureau scores, which
  is exactly what the synthetic data was constructed to reward, and mirrors
  what's reported for the real dataset.

  ![Horizontal bar chart of mean absolute SHAP value per feature, ranked. The three external bureau scores (EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1) dominate, followed by debt-to-income.](outputs/figures/shap_importance.png)

- **Local explanation**, one declined applicant (from the notebook): *"Applicant
  103791 was predicted to have a 75% chance of default and would be declined
  under this model. Factors that pushed their risk score up: a second
  external credit bureau score that is low for this applicant pool (0.37 vs.
  a typical 0.63) ... Factors that pushed their risk score down: applicant
  age that is high for this applicant pool (53.74 vs. a typical 45.48) ..."*

**A finding worth calling out rather than glossing over:** the optimal
threshold (0.52) landed *close* to the naive 0.5 cutoff here, not far from
it. That's not the pipeline failing to find a dramatic result -- it's because
`class_weight="balanced"` already pushes the model's predicted probabilities
into a reasonably well-calibrated range, so 0.5 turns out to be a decent
guess in this particular run. The point of this project was never "0.5 is
always wrong" -- it's that nobody could have known it was *this* close to
right without doing the cost-sensitive sweep. On a differently-calibrated
model, or with different cost assumptions, this gap could be far larger, and
the only way to know is to check, not assume.

**Limitations of the results:**
- All numbers above come from the synthetic sample. Re-run against the real
  Kaggle file (see below) before treating any of them as a claim about real
  credit risk.
- SHAP explanations were generated and verified to run against the actually
  installed `shap` version in this environment (`0.52.0`) -- the shape
  normalization in `explain.py` was written against the documented
  `TreeExplainer` API but validated against real output, not assumed.
- The model is not tuned for maximum AUC on purpose (see Non-goals in
  `PRD.md`); a gradient-boosted model or richer joined feature set would very
  likely score higher.

### Validated against the real Kaggle dataset

Everything above is the synthetic-sample run, which is what this repo
reproduces by default. The pipeline was also run, once, against the real
`application_train.csv` (307,511 rows) to confirm it actually works end to
end on the real thing and isn't just tuned to its own synthetic stand-in.
That real file isn't committed (see "How to reproduce" below for why), so
these numbers are reported as one-time validation evidence rather than
something a fresh clone reproduces automatically -- re-download the real
data yourself if you want to regenerate them.

- **Baseline model:** ROC AUC = **0.739** -- within 0.004 of the synthetic
  run's 0.743, which is a reasonable sign the synthetic data wasn't
  accidentally miscalibrated relative to the real thing.
- **Optimal threshold: 0.63**, clearly apart from the naive 0.5 cutoff this
  time -- and unlike the synthetic run, this isn't a subtle effect:

  ![Cost curve on the real 307,511-row dataset: the naive 0.5 threshold sits well up the descending slope, far from the true minimum at 0.63, and even costs more than approving every applicant.](outputs/figures/cost_curve_real_data.png)

  - Total cost at the optimal threshold: **$1,849,876,999**
  - Total cost at the naive 0.5 threshold: **$2,192,470,526**
  - Total cost approving every applicant: **$2,080,401,335**
  - **Savings vs. naive 0.5: ~$343M. Savings vs. approving everyone: ~$231M.**
  - Worth sitting with: at the naive 0.5 cutoff, total cost is *higher* than
    just approving every single applicant. On this run, defaulting to 0.5
    wouldn't just be a suboptimal decision -- it would actively lose more
    money than doing nothing. That's the entire argument of this project in
    one number, and it only showed up on the real data, not the synthetic
    sample -- which is itself the reason the synthetic run's README
    disclaimer above ("the only way to know is to check, not assume") is
    there rather than a stronger claim.
- **Global SHAP ranking**, top 3: `EXT_SOURCE_3`, `EXT_SOURCE_2`,
  `EXT_SOURCE_1` (same three features as the synthetic run, reordered) --
  consistent with what's widely reported for this dataset elsewhere.
- **Bug found and fixed during this validation run:** the local explanation's
  phrasing for one-hot categorical features didn't check whether the
  applicant's actual value was 1 or 0 before saying "a family status of X" --
  on real data this surfaced as one applicant's narrative claiming both
  "Married" *and* "Single / not married" in the same sentence, which is
  impossible. Fixed in `explain.py` (`_describe_feature`) to phrase a 0-value
  dummy as "not having a family status of X" instead. This is exactly the
  kind of bug that stays invisible on a small, low-cardinality synthetic
  sample and only shows up once you run against messier real-world data.

  ![Global SHAP feature importance on the real dataset -- same top-3 external bureau scores as the synthetic run, confirming the synthetic data's signal structure matches the real thing.](outputs/figures/shap_importance_real_data.png)

Cost totals here are two orders of magnitude larger than the synthetic run's
purely because the real test set has ~56x more rows -- that scale difference
is not itself meaningful. What's meaningful is the *shape*: a materially
different optimal threshold, and a naive 0.5 cutoff that's demonstrably
worse than the do-nothing baseline.

## How to reproduce

```bash
git clone <this-repo>
cd credit-risk-threshold-optimizer
python -m venv .venv && source .venv/bin/activate   # Python 3.11-3.13 recommended
pip install -r requirements.txt
```

### With the bundled synthetic sample (default, no setup)

```bash
python src/data_prep.py      # (re)generates data/sample.csv, saves outputs/figures/days_employed_anomaly.png
python src/model.py          # trains the baseline model, prints AUC + classification report
python src/cost_analysis.py  # runs the threshold sweep, prints $ results, saves outputs/figures/cost_curve.png
python src/explain.py        # prints global SHAP ranking + local explanation, saves outputs/figures/shap_importance.png
```

Or open `notebooks/analysis.ipynb` for the narrated version with everything
already executed and baked in.

### With the real Kaggle data (opt-in upgrade)

The synthetic sample is the default so nothing below is required to run this
project -- but if you want the real ~300k-row dataset instead of the
synthetic stand-in, here's the full path, either through the browser or the
Kaggle CLI.

**1. Create a Kaggle account (skip if you have one).**
Sign up free at [kaggle.com](https://www.kaggle.com/account/login).

**2. Accept the competition's rules.**
Go to the [Home Credit Default Risk competition page](https://www.kaggle.com/c/home-credit-default-risk),
click **Join Competition** / **Late Submission**, and accept the terms. This
step is required -- Kaggle blocks the download otherwise, even with a valid
account.

**3. Get the file, using either option:**

- **Option A -- browser (no setup, larger click-through):**
  1. On the competition page, open the **Data** tab.
  2. Scroll to the file list and download `application_train.csv` directly
     (it's the largest file at ~160MB; you don't need the other files in
     the zip -- `bureau.csv`, `previous_application.csv`, etc. -- this
     project only reads the one table).
  3. If Kaggle gives you a `.zip`, unzip it so you're left with the plain
     `application_train.csv`.

- **Option B -- Kaggle CLI (faster if you're doing this more than once):**
  1. Install the CLI into your project's virtual environment:
     ```bash
     pip install kaggle
     ```
  2. Get an API token: on Kaggle, go to
     **Account Settings** (click your profile picture -> Settings) ->
     scroll to the **API** section -> **Create New Token**. This downloads
     a `kaggle.json` file containing your username and key.
  3. Place that file where the CLI expects it, and lock down its
     permissions (it's a credential):
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
     ```
  4. Download and unzip straight into place from the project root:
     ```bash
     kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw/
     unzip data/raw/application_train.csv.zip -d data/raw/
     rm data/raw/application_train.csv.zip
     ```
     **If that 403s** (this happened during testing, even with rules
     accepted and a valid token -- the single-file endpoint seems flaky):
     fall back to downloading the whole competition archive and pulling out
     just the one file you need:
     ```bash
     kaggle competitions download -c home-credit-default-risk -p data/raw/
     unzip data/raw/home-credit-default-risk.zip application_train.csv -d data/raw/
     rm data/raw/home-credit-default-risk.zip
     ```
     This one's ~690MB zipped (vs. ~160MB for the single file) and takes a
     few minutes, but it's the reliable path if the targeted download fails.

**4. Confirm it landed in the right spot.**
The file must be at exactly `data/raw/application_train.csv` (this path is
gitignored, so it's never accidentally committed):
```bash
ls -lh data/raw/application_train.csv
```

**5. Re-run anything -- no code changes needed.**
```bash
python src/data_prep.py
```
`load_raw_data()` in `src/data_prep.py` checks for that exact path first and
prints which source it used. You should now see:
```
[data_prep] Loading real data from /.../data/raw/application_train.csv
```
instead of the "Real data not found... using synthetic sample" message. Every
other script (`model.py`, `cost_analysis.py`, `explain.py`) and the notebook
pick this up automatically the same way -- nothing else to configure.

**Note:** the real file only needs the columns this project actually reads
(`SK_ID_CURR`, `TARGET`, `EXT_SOURCE_1/2/3`, etc. -- the full list is
`COLUMNS` in `src/data_prep.py`); the other ~100+ columns in the real file
are simply ignored on load, not an error.

## Repo layout

```
credit-risk-threshold-optimizer/
├── README.md
├── PRD.md                  # source-of-truth scope doc for this project
├── requirements.txt
├── data/
│   ├── raw/                 # real Kaggle file goes here (gitignored)
│   └── sample.csv            # synthetic fallback, committed
├── src/
│   ├── data_prep.py          # synthetic data generation, loading, cleaning
│   ├── model.py               # train/test split, imputation, RandomForest baseline
│   ├── cost_analysis.py       # threshold sweep, cost curve plot
│   └── explain.py             # SHAP global + local explanations
├── notebooks/
│   └── analysis.ipynb        # narrated, fully-executed walkthrough
└── outputs/figures/
    ├── cost_curve.png                 # the centerpiece: $ cost vs. threshold (synthetic sample)
    ├── days_employed_anomaly.png      # the DAYS_EMPLOYED sentinel, before/after cleaning
    ├── shap_importance.png            # global SHAP feature ranking (synthetic sample)
    ├── cost_curve_real_data.png       # same cost curve, one-time validation run on real Kaggle data
    └── shap_importance_real_data.png  # same SHAP ranking, real Kaggle data
```
