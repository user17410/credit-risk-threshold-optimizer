"""Cost-sensitive threshold analysis -- the centerpiece of this project.

A predicted probability isn't a decision. Someone has to pick the cutoff
that turns "9% chance of default" into "approve" or "decline," and that
cutoff should be chosen by weighing the dollar cost of each error type,
not defaulted to 0.5. This module sweeps the threshold, prices both error
types in dollars, and reports the minimum.

LOSS_GIVEN_DEFAULT and EXPECTED_MARGIN below are illustrative assumptions,
not real underwriting economics -- see the README for the reasoning behind
these specific numbers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from model import evaluate_baseline, train_baseline_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# --- Illustrative cost assumptions (see README for rationale) ---
# Fraction of the loan amount lost when an approved applicant defaults.
LOSS_GIVEN_DEFAULT = 0.60
# Fraction of the loan amount in margin/profit forgone when a good
# applicant is wrongly declined.
EXPECTED_MARGIN = 0.12

THRESHOLD_GRID = np.arange(0.01, 1.0, 0.01)


def sweep_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    loan_amount: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
    loss_given_default: float = LOSS_GIVEN_DEFAULT,
    expected_margin: float = EXPECTED_MARGIN,
) -> pd.DataFrame:
    """Compute total $ cost at each threshold.

    Approval rule: approve if predicted P(default) <= threshold.
    - Approved defaults cost loan_amount * loss_given_default.
    - Declined good applicants cost loan_amount * expected_margin (forgone profit).
    """
    y_true = np.asarray(y_true)
    loan_amount = np.asarray(loan_amount)

    rows = []
    for t in thresholds:
        approved = proba <= t
        approved_default_mask = approved & (y_true == 1)
        declined_good_mask = (~approved) & (y_true == 0)

        default_cost = (loan_amount[approved_default_mask] * loss_given_default).sum()
        rejection_cost = (loan_amount[declined_good_mask] * expected_margin).sum()

        rows.append(
            {
                "threshold": t,
                "default_cost": default_cost,
                "rejection_cost": rejection_cost,
                "total_cost": default_cost + rejection_cost,
                "n_approved": int(approved.sum()),
                "n_approved_defaults": int(approved_default_mask.sum()),
                "n_declined_good": int(declined_good_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def cost_at_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    loan_amount: np.ndarray,
    threshold: float,
    loss_given_default: float = LOSS_GIVEN_DEFAULT,
    expected_margin: float = EXPECTED_MARGIN,
) -> float:
    df = sweep_thresholds(
        proba, y_true, loan_amount, thresholds=np.array([threshold]),
        loss_given_default=loss_given_default, expected_margin=expected_margin,
    )
    return float(df["total_cost"].iloc[0])


def approve_everyone_cost(
    y_true: np.ndarray, loan_amount: np.ndarray, loss_given_default: float = LOSS_GIVEN_DEFAULT
) -> float:
    """Cost if every applicant is approved (threshold = 1.0): only default cost, no rejection cost."""
    y_true = np.asarray(y_true)
    loan_amount = np.asarray(loan_amount)
    return float((loan_amount[y_true == 1] * loss_given_default).sum())


def _millions_formatter(x, _pos):
    return f"${x / 1e6:,.0f}M"


def plot_cost_curves(sweep_df: pd.DataFrame, optimal_threshold: float, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    ax.plot(sweep_df["threshold"], sweep_df["rejection_cost"], label="Cost of wrongly rejected good applicants",
            color="#2a78d6", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["default_cost"], label="Cost of approved defaults",
            color="#eb6834", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["total_cost"], label="Total cost",
            color="#0b0b0b", linewidth=2.5)

    ax.axvline(optimal_threshold, color="#0ca30c", linestyle="--", linewidth=1.5,
               label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.axvline(0.5, color="#898781", linestyle=":", linewidth=1.5, label="Naive 0.5 threshold")

    ax.set_xlabel("Approval threshold (approve if predicted P(default) ≤ threshold)", color="#52514e")
    ax.set_ylabel("Total dollar cost on test set", color="#52514e")
    ax.set_title("Where should the lender draw the approval line?", color="#0b0b0b", fontsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(_millions_formatter))
    ax.tick_params(colors="#898781")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    legend = ax.legend(loc="upper center", fontsize=9, frameon=True)
    legend.get_frame().set_facecolor("#fcfcfb")
    legend.get_frame().set_edgecolor("#e1e0d9")
    ax.grid(alpha=0.3, color="#e1e0d9")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved cost curve to {save_path}")
    return fig


def run_cost_analysis():
    trained = train_baseline_model()
    results = evaluate_baseline(trained)
    proba = results["proba"]
    y_true = trained.y_test.to_numpy()
    loan_amount = trained.test_raw["AMT_CREDIT"].to_numpy()

    sweep_df = sweep_thresholds(proba, y_true, loan_amount)
    best_row = sweep_df.loc[sweep_df["total_cost"].idxmin()]
    optimal_threshold = float(best_row["threshold"])
    optimal_cost = float(best_row["total_cost"])

    cost_at_naive = cost_at_threshold(proba, y_true, loan_amount, 0.5)
    cost_approve_all = approve_everyone_cost(y_true, loan_amount)

    savings_vs_naive = cost_at_naive - optimal_cost
    savings_vs_approve_all = cost_approve_all - optimal_cost

    print(f"Loss given default assumption: {LOSS_GIVEN_DEFAULT:.0%} of loan amount")
    print(f"Expected margin assumption: {EXPECTED_MARGIN:.0%} of loan amount")
    print()
    print(f"Optimal threshold: {optimal_threshold:.2f}  ->  total cost ${optimal_cost:,.0f}")
    print(f"Naive 0.5 threshold: total cost ${cost_at_naive:,.0f}")
    print(f"Approve-everyone (threshold=1.0): total cost ${cost_approve_all:,.0f}")
    print()
    print(f"$ saved vs. naive 0.5 threshold: ${savings_vs_naive:,.0f}")
    print(f"$ saved vs. approving everyone: ${savings_vs_approve_all:,.0f}")

    fig_path = FIGURES_DIR / "cost_curve.png"
    plot_cost_curves(sweep_df, optimal_threshold, save_path=fig_path)

    return {
        "sweep_df": sweep_df,
        "optimal_threshold": optimal_threshold,
        "optimal_cost": optimal_cost,
        "cost_at_naive": cost_at_naive,
        "cost_approve_all": cost_approve_all,
        "savings_vs_naive": savings_vs_naive,
        "savings_vs_approve_all": savings_vs_approve_all,
    }


if __name__ == "__main__":
    run_cost_analysis()
