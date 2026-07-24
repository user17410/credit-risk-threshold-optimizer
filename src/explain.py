"""SHAP explainability: global feature ranking + plain-English local explanations.

A probability of default is not, by itself, something a non-technical
manager can act on or defend to a customer who asks "why was I declined."
This module turns SHAP values into (a) a ranked list of what drives risk
overall and (b) a short paragraph for one specific applicant.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from model import TrainedModel, train_baseline_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Human-readable feature descriptions, used to phrase local explanations.
FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_1": "an external credit bureau score",
    "EXT_SOURCE_2": "a second external credit bureau score",
    "EXT_SOURCE_3": "a third external credit bureau score",
    "DEBT_TO_INCOME": "the ratio of requested credit to annual income",
    "AGE_YEARS": "applicant age",
    "YEARS_EMPLOYED": "years at current employment",
    "AMT_INCOME_TOTAL": "total annual income",
    "AMT_CREDIT": "the requested credit amount",
    "AMT_ANNUITY": "the loan annuity payment",
    "CNT_CHILDREN": "number of children",
    "OWN_CAR_AGE": "the age of the applicant's car",
    "FLAG_EMPLOYED_ANOMALY": "having no verifiable current employment record",
}

# One-hot encoded categorical prefixes get their own phrasing template
# ("{value} for {label}") rather than falling through to a raw column name.
CATEGORICAL_LABELS = {
    "CODE_GENDER": "gender",
    "FLAG_OWN_CAR": "car ownership",
    "FLAG_OWN_REALTY": "property ownership",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_FAMILY_STATUS": "family status",
    "NAME_INCOME_TYPE": "income type",
}


def positive_class_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Return a plain (n_samples, n_features) array of SHAP values for the positive class.

    shap's return shape for classifiers is not consistent across versions:
    - a list of per-class arrays, e.g. [shap_class0, shap_class1]
    - a single 3D array (n_samples, n_features, n_classes)
    - already a 2D (n_samples, n_features) array for the positive class

    Assuming any one of these will break on some installed version, so all
    three are handled explicitly here instead.
    """
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        # list of per-class arrays -- take the positive class (index 1)
        return np.asarray(raw[1])

    arr = np.asarray(raw)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) -- take the positive class
        return arr[:, :, 1]
    if arr.ndim == 2:
        # already per-sample-per-feature for a single (positive) class
        return arr

    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def global_feature_ranking(trained: TrainedModel, sample_size: int = 300, seed: int = 42) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, ranked, on a sample of the test set."""
    X_sample = trained.X_test.sample(n=min(sample_size, len(trained.X_test)), random_state=seed)
    explainer = shap.TreeExplainer(trained.model)
    shap_vals = positive_class_shap_values(explainer, X_sample)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": trained.feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return ranking


def explain_applicant(trained: TrainedModel, row_position: int, top_n: int = 4) -> dict:
    """Produce SHAP values + a plain-English explanation for one test-set applicant.

    row_position is a positional index into trained.X_test / trained.test_raw
    (i.e. 0-based, not SK_ID_CURR).
    """
    x_row = trained.X_test.iloc[[row_position]]
    explainer = shap.TreeExplainer(trained.model)
    shap_vals = positive_class_shap_values(explainer, x_row)[0]

    proba = trained.model.predict_proba(x_row)[0, 1]
    sk_id = trained.id_test.iloc[row_position]

    # Median of each feature across the test set, used to phrase a numeric
    # value as "low" or "high" relative to peers instead of a bare number.
    medians = trained.X_test.median()

    contributions = pd.DataFrame(
        {"feature": trained.feature_names, "shap_value": shap_vals, "value": x_row.iloc[0].to_numpy()}
    ).sort_values("shap_value", key=np.abs, ascending=False)

    top_pushing_up = contributions[contributions["shap_value"] > 0].head(top_n)
    top_pushing_down = contributions[contributions["shap_value"] < 0].head(top_n)

    narrative = _build_narrative(sk_id, proba, top_pushing_up, top_pushing_down, medians)

    return {
        "sk_id": sk_id,
        "predicted_proba": proba,
        "contributions": contributions,
        "narrative": narrative,
    }


def _describe_feature(feature: str, value: float | None = None, median: float | None = None) -> str:
    if feature in FEATURE_DESCRIPTIONS:
        base = FEATURE_DESCRIPTIONS[feature]
        if value is not None and median is not None:
            qualifier = "low" if value < median else "high"
            return f"{base} that is {qualifier} for this applicant pool ({value:.2f} vs. a typical {median:.2f})"
        return base
    for prefix, label in CATEGORICAL_LABELS.items():
        if feature.startswith(prefix + "_"):
            cat_value = feature[len(prefix) + 1:]
            # One-hot columns are 0/1: a SHAP value on a 0 (i.e. "does NOT
            # belong to this category") is a real, valid contribution, but
            # phrasing it as "a family status of X" would misreport the
            # applicant as belonging to X when they don't -- and since only
            # one dummy per group is ever 1, two "belongs to X" phrases in
            # the same narrative would describe an impossible applicant.
            if value is not None and value < 0.5:
                return f"not having a {label} of \"{cat_value}\""
            return f"a {label} of \"{cat_value}\""
    return feature.replace("_", " ").lower()


def _build_narrative(sk_id, proba: float, up: pd.DataFrame, down: pd.DataFrame, medians: pd.Series) -> str:
    decision = "declined" if proba > 0.5 else "approved"

    up_phrases = [_describe_feature(row.feature, row.value, medians[row.feature]) for row in up.itertuples()]
    down_phrases = [_describe_feature(row.feature, row.value, medians[row.feature]) for row in down.itertuples()]

    sentences = [
        f"Applicant {sk_id} was predicted to have a {proba:.0%} chance of default and would be {decision} "
        f"under this model."
    ]
    if up_phrases:
        sentences.append(
            "Factors that pushed their risk score up: " + ", ".join(up_phrases) + "."
        )
    if down_phrases:
        sentences.append(
            "Factors that pushed their risk score down (worked in their favor): "
            + ", ".join(down_phrases) + "."
        )
    return " ".join(sentences)


def pick_declined_applicant(trained: TrainedModel, proba: np.ndarray) -> int:
    """Return the positional index of a clearly-declined applicant (highest predicted risk)."""
    return int(np.argmax(proba))


def plot_global_importance(ranking: pd.DataFrame, top_n: int = 10, save_path: Path | None = None):
    """Horizontal bar chart of mean |SHAP| per feature, styled to match the cost curve."""
    top = ranking.head(top_n).iloc[::-1]  # reverse so the top feature ends up at the top of the barh

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    ax.barh(top["feature"], top["mean_abs_shap"], color="#2a78d6")

    ax.set_xlabel("Mean |SHAP value| (impact on predicted default risk)", color="#52514e")
    ax.set_title("What drives predicted default risk?", color="#0b0b0b", fontsize=13)
    ax.tick_params(colors="#898781")
    ax.tick_params(axis="y", colors="#0b0b0b")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(alpha=0.3, color="#e1e0d9", axis="x")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved SHAP importance chart to {save_path}")
    return fig


if __name__ == "__main__":
    trained = train_baseline_model()
    proba = trained.model.predict_proba(trained.X_test)[:, 1]

    print("=== Global feature importance (mean |SHAP|, sample of 300 test rows) ===")
    ranking = global_feature_ranking(trained)
    print(ranking.head(10).to_string(index=False))
    print("\nTop 3 drivers overall:")
    for _, row in ranking.head(3).iterrows():
        print(f"  - {row['feature']}: mean |SHAP| = {row['mean_abs_shap']:.4f}")
    plot_global_importance(ranking, save_path=FIGURES_DIR / "shap_importance.png")

    print("\n=== Local explanation: one declined applicant ===")
    idx = pick_declined_applicant(trained, proba)
    result = explain_applicant(trained, idx)
    print(result["narrative"])
