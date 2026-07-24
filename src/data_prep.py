"""Data loading, synthetic fallback generation, and cleaning for the
Home Credit Default Risk problem.

Design note: this module is deliberately self-contained (no notebook-only
logic) so every step can be unit-tested or re-run from the CLI.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample.csv"

RANDOM_SEED = 42

# The real Home Credit dataset encodes "currently not employed" as this
# literal sentinel day-count in DAYS_EMPLOYED instead of a missing value.
# Anyone who doesn't check for it will treat it as ~1000 years of tenure.
DAYS_EMPLOYED_SENTINEL = 365243

COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OWN_CAR_AGE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]


def generate_synthetic_sample(n_rows: int = 5500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a synthetic stand-in for application_train.csv.

    Mirrors the real dataset's known quirks on purpose (rather than being
    plain random) so the rest of the pipeline has something real to catch:
    - ~8-9% default rate (the real base rate)
    - EXT_SOURCE_1/2/3 are the strongest signal for TARGET, same as in Kaggle
    - DAYS_EMPLOYED sentinel value (365243) for pensioners/unemployed
    - EXT_SOURCE_1 heavily missing (50%+), OWN_CAR_AGE missing when no car
    """
    rng = np.random.default_rng(seed)

    sk_id = np.arange(100001, 100001 + n_rows)

    code_gender = rng.choice(["F", "M"], size=n_rows, p=[0.65, 0.35])
    flag_own_car = rng.choice(["Y", "N"], size=n_rows, p=[0.34, 0.66])
    flag_own_realty = rng.choice(["Y", "N"], size=n_rows, p=[0.69, 0.31])
    cnt_children = rng.poisson(0.4, size=n_rows).clip(0, 6)

    name_education = rng.choice(
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
        size=n_rows,
        p=[0.71, 0.24, 0.035, 0.01, 0.005],
    )
    name_family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n_rows,
        p=[0.64, 0.15, 0.10, 0.07, 0.04],
    )
    # income type drives who is a pensioner (relevant to the DAYS_EMPLOYED anomaly)
    name_income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
        size=n_rows,
        p=[0.52, 0.23, 0.18, 0.06, 0.01],
    )

    # Age: 21-69 years old, expressed as negative days like the real data
    age_years = rng.uniform(21, 69, size=n_rows)
    days_birth = -(age_years * 365.25).astype(int)

    # Employment tenure in days (negative = days before application), capped by age.
    max_tenure_years = np.clip(age_years - 18, 0, None)
    tenure_years = rng.exponential(4.0, size=n_rows).clip(0, max_tenure_years)
    days_employed = -(tenure_years * 365.25).astype(int)

    is_pensioner_or_unemployed = np.isin(name_income_type, ["Pensioner", "Unemployed"])
    # Real dataset: sentinel appears mainly (not exclusively) for pensioners/unemployed.
    sentinel_mask = is_pensioner_or_unemployed & (rng.random(n_rows) < 0.92)
    # A small stray fraction of working applicants also carry the sentinel,
    # matching the real data's messiness.
    stray_mask = (~is_pensioner_or_unemployed) & (rng.random(n_rows) < 0.02)
    days_employed = np.where(sentinel_mask | stray_mask, DAYS_EMPLOYED_SENTINEL, days_employed)

    income_total = rng.lognormal(mean=11.9, sigma=0.45, size=n_rows).round(-2)
    income_total = np.clip(income_total, 25650, 4_000_000)

    credit_amt = (income_total * rng.uniform(1.5, 6.5, size=n_rows)).round(-2)
    annuity = (credit_amt / rng.uniform(10, 30, size=n_rows)).round(1)

    own_car_age = np.where(
        flag_own_car == "Y",
        rng.uniform(0, 20, size=n_rows).round(1),
        np.nan,
    )

    # EXT_SOURCE_1/2/3: normalized external bureau scores, the strongest
    # real-world predictors of default. Higher score = lower risk.
    ext_source_1 = rng.beta(5, 3, size=n_rows)
    ext_source_2 = rng.beta(5, 3, size=n_rows)
    ext_source_3 = rng.beta(5, 3, size=n_rows)

    # EXT_SOURCE_1 is famously sparse in the real data (50%+ missing).
    ext1_missing_mask = rng.random(n_rows) < 0.56
    ext_source_1_observed = ext_source_1.copy()
    ext_source_1_observed[ext1_missing_mask] = np.nan
    # light missingness on the other two, as in the real data
    ext_source_2_observed = ext_source_2.copy()
    ext_source_2_observed[rng.random(n_rows) < 0.02] = np.nan
    ext_source_3_observed = ext_source_3.copy()
    ext_source_3_observed[rng.random(n_rows) < 0.20] = np.nan

    # Build TARGET as a real function of the EXT_SOURCE scores (using the
    # true underlying values, not the masked/observed ones -- missingness in
    # a bureau score shouldn't itself leak the label) plus a debt ratio term
    # and noise, so the model has genuine signal to recover.
    debt_to_income = credit_amt / income_total
    risk_score = (
        -4.2 * ext_source_1
        - 3.6 * ext_source_2
        - 3.2 * ext_source_3
        + 0.35 * (debt_to_income - debt_to_income.mean()) / debt_to_income.std()
        + rng.normal(0, 0.75, size=n_rows)
    )
    # Calibrate intercept so the base default rate lands at ~8.5%.
    target_prob = 1 / (1 + np.exp(-(risk_score + 3.85)))
    target = (rng.random(n_rows) < target_prob).astype(int)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": sk_id,
            "TARGET": target,
            "CODE_GENDER": code_gender,
            "FLAG_OWN_CAR": flag_own_car,
            "FLAG_OWN_REALTY": flag_own_realty,
            "CNT_CHILDREN": cnt_children,
            "AMT_INCOME_TOTAL": income_total,
            "AMT_CREDIT": credit_amt,
            "AMT_ANNUITY": annuity,
            "NAME_EDUCATION_TYPE": name_education,
            "NAME_FAMILY_STATUS": name_family_status,
            "NAME_INCOME_TYPE": name_income_type,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "OWN_CAR_AGE": own_car_age,
            "EXT_SOURCE_1": ext_source_1_observed,
            "EXT_SOURCE_2": ext_source_2_observed,
            "EXT_SOURCE_3": ext_source_3_observed,
        }
    )
    return df[COLUMNS]


def load_raw_data() -> pd.DataFrame:
    """Load application data, preferring the real Kaggle file if present."""
    if RAW_PATH.exists():
        print(f"[data_prep] Loading real data from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH, usecols=lambda c: c in COLUMNS)
        return df[COLUMNS]
    print(f"[data_prep] Real data not found at {RAW_PATH}; using synthetic sample at {SAMPLE_PATH}")
    return pd.read_csv(SAMPLE_PATH)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quirks and add features that a stakeholder can reason about.

    Imputation is intentionally NOT done here -- it happens after the
    train/test split (see model.py) so the training-only imputer never
    sees test-set values.
    """
    df = df.copy()

    # --- DAYS_EMPLOYED sentinel fix ---
    # Flag first, then null out, so the anomaly itself remains a usable
    # (and highly predictive -- it correlates with being a pensioner) feature
    # instead of silently vanishing.
    df["FLAG_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df.loc[df["FLAG_EMPLOYED_ANOMALY"] == 1, "DAYS_EMPLOYED"] = np.nan

    # --- engineered features ---
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25
    df["DEBT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    return df


def build_sample_csv(n_rows: int = 5500, seed: int = RANDOM_SEED) -> Path:
    """Generate the synthetic dataset and write it to data/sample.csv."""
    df = generate_synthetic_sample(n_rows=n_rows, seed=seed)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_PATH, index=False)
    return SAMPLE_PATH


if __name__ == "__main__":
    path = build_sample_csv()
    df = pd.read_csv(path)
    print(f"Wrote {len(df)} rows to {path}")
    print(f"Default rate: {df['TARGET'].mean():.3%}")
    print(f"DAYS_EMPLOYED sentinel count: {(df['DAYS_EMPLOYED'] == DAYS_EMPLOYED_SENTINEL).sum()}")
    print(f"EXT_SOURCE_1 missing: {df['EXT_SOURCE_1'].isna().mean():.1%}")

    cleaned = clean_and_engineer(df)
    print("\nAfter cleaning:")
    print(cleaned[["FLAG_EMPLOYED_ANOMALY", "AGE_YEARS", "YEARS_EMPLOYED", "DEBT_TO_INCOME"]].describe())
