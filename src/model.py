"""Baseline credit-default classifier.

A RandomForest, not because it's the best possible model, but because this
project's differentiator is the threshold decision layer downstream, not
squeezing out extra AUC -- and RandomForest's SHAP support (TreeExplainer)
is the most reliable across library versions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_prep import RANDOM_SEED, clean_and_engineer, load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NUMERIC_FEATURES = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "OWN_CAR_AGE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "FLAG_EMPLOYED_ANOMALY",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "DEBT_TO_INCOME",
]
CATEGORICAL_FEATURES = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
]


@dataclass
class TrainedModel:
    model: RandomForestClassifier
    numeric_imputer: SimpleImputer
    encoder: OneHotEncoder
    feature_names: list[str]
    X_test: pd.DataFrame  # transformed, model-ready
    y_test: pd.Series
    id_test: pd.Series
    test_raw: pd.DataFrame  # original (cleaned, untransformed) test rows, for cost analysis / SHAP narration


def _split_features_target(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["TARGET"]
    ids = df["SK_ID_CURR"]
    return X, y, ids


def _transform(
    X: pd.DataFrame,
    numeric_imputer: SimpleImputer,
    encoder: OneHotEncoder,
    fit: bool,
) -> pd.DataFrame:
    """Apply (or fit+apply) imputation and encoding.

    Imputer/encoder are fit ONLY when fit=True (i.e. on the training split).
    Calling with fit=False on test data reuses statistics learned from train,
    which is the whole point -- fitting on the full dataset before the split
    would leak test-set distribution into training.
    """
    numeric = X[NUMERIC_FEATURES]
    categorical = X[CATEGORICAL_FEATURES]

    if fit:
        numeric_imputed = numeric_imputer.fit_transform(numeric)
        categorical_encoded = encoder.fit_transform(categorical)
    else:
        numeric_imputed = numeric_imputer.transform(numeric)
        categorical_encoded = encoder.transform(categorical)

    cat_cols = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    numeric_df = pd.DataFrame(numeric_imputed, columns=NUMERIC_FEATURES, index=X.index)
    categorical_df = pd.DataFrame(categorical_encoded, columns=cat_cols, index=X.index)
    return pd.concat([numeric_df, categorical_df], axis=1)


def train_baseline_model(df: pd.DataFrame | None = None, seed: int = RANDOM_SEED) -> TrainedModel:
    if df is None:
        df = clean_and_engineer(load_raw_data())

    X, y, ids = _split_features_target(df)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.25, stratify=y, random_state=seed
    )

    # Imputer/encoder are fit on the training split only (see _transform docstring).
    numeric_imputer = SimpleImputer(strategy="median")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_t = _transform(X_train, numeric_imputer, encoder, fit=True)
    X_test_t = _transform(X_test, numeric_imputer, encoder, fit=False)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_t, y_train)

    test_raw = df.loc[X_test.index]

    return TrainedModel(
        model=clf,
        numeric_imputer=numeric_imputer,
        encoder=encoder,
        feature_names=list(X_train_t.columns),
        X_test=X_test_t,
        y_test=y_test,
        id_test=id_test,
        test_raw=test_raw,
    )


def evaluate_baseline(trained: TrainedModel) -> dict:
    proba = trained.model.predict_proba(trained.X_test)[:, 1]
    auc = roc_auc_score(trained.y_test, proba)
    preds_at_half = (proba >= 0.5).astype(int)
    report = classification_report(trained.y_test, preds_at_half, digits=3)
    return {"auc": auc, "report": report, "proba": proba}


if __name__ == "__main__":
    trained = train_baseline_model()
    results = evaluate_baseline(trained)

    print(f"Test set size: {len(trained.y_test)}")
    print(f"Test set default rate: {trained.y_test.mean():.3%}")
    print(f"\nBaseline ROC AUC: {results['auc']:.4f}")
    print("\nClassification report @ naive 0.5 threshold:")
    print(results["report"])
