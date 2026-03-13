"""
evaluate.py
-----------
Model evaluation utilities for the student score prediction pipeline.

Provides two levels of evaluation granularity:

1. Overall metrics  : RMSE, MAE, R², CV RMSE — used for model selection.
2. Per-band metrics : RMSE and MAE broken down by O-level grade band —
                      directly answers the business question of whether the
                      model identifies Fail-band students (<50) accurately.

Metric justification (from prototype.ipynb §9)
----------------------------------------------
- RMSE    : Primary metric. Penalises large errors more than MAE; critical
            because a large mis-prediction on a borderline Fail student has
            real consequences (student receives no intervention).
- MAE     : Secondary metric. Interpretable as mean absolute score error in
            the same unit as final_test (0-100 scale).
- R²      : Tertiary metric. Proportion of variance explained; useful for
            communicating overall model quality to non-technical stakeholders.
- CV RMSE : Used for model selection (not final reporting). Computed on
            training data with 5-fold cross-validation to prevent test-set
            leakage during hyperparameter tuning.

Public API
----------
evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
    -> dict of metrics (train + CV + test)

evaluate_by_band(pipeline, X_test, y_test)
    -> pd.DataFrame with per-band RMSE, MAE, N, and % of dataset

print_report(results)
    -> formatted console output (no return value)

compare_models(results_list)
    -> pd.DataFrame summary table, sorted by CV RMSE ascending
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

logger = logging.getLogger(__name__)

# ── Score band definitions (mirrors eda.ipynb §11 and run.py split strategy) ──
SCORE_BANDS = {
    "Fail (<50)": (0,  50),
    "C   (50-59)": (50, 60),
    "B   (60-69)": (60, 70),
    "A2  (70-79)": (70, 80),
    "A1  (80-100)": (80, 101),
}

RANDOM_STATE = 42


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    cv: int = 5,
) -> dict:
    """Fit a pipeline and compute train / CV / test metrics.

    The pipeline is fitted on X_train inside this function. Do NOT pass a
    pre-fitted pipeline — the training metrics would then reflect re-prediction
    on seen data rather than a genuine fit.

    Parameters
    ----------
    name : str
        Human-readable model name used as the 'Model' key in the returned dict.
    pipeline : sklearn.pipeline.Pipeline
        An unfitted sklearn Pipeline ending in a regressor.
    X_train, y_train : training features and target.
    X_test,  y_test  : held-out test features and target.
    cv : int
        Number of folds for cross-validation (default 5).

    Returns
    -------
    dict with keys:
        Model, Train RMSE, Test RMSE, CV RMSE, CV RMSE std,
        Train MAE, Test MAE, Train R2, Test R2, Overfit Gap
    """
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
    )

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse  = root_mean_squared_error(y_test,  y_test_pred)

    results = {
        "Model"       : name,
        "Train RMSE"  : round(train_rmse, 4),
        "Test RMSE"   : round(test_rmse,  4),
        "CV RMSE"     : round(-cv_scores.mean(), 4),
        "CV RMSE std" : round(cv_scores.std(),   4),
        "Train MAE"   : round(mean_absolute_error(y_train, y_train_pred), 4),
        "Test MAE"    : round(mean_absolute_error(y_test,  y_test_pred),  4),
        "Train R2"    : round(r2_score(y_train, y_train_pred), 4),
        "Test R2"     : round(r2_score(y_test,  y_test_pred),  4),
        "Overfit Gap" : round(train_rmse - test_rmse, 4),
    }

    logger.info(
        "[%s] Test RMSE=%.4f  CV RMSE=%.4f±%.4f  R²=%.4f",
        name, results["Test RMSE"], results["CV RMSE"],
        results["CV RMSE std"], results["Test R2"],
    )
    return results, pipeline


# ── Per-band evaluation ────────────────────────────────────────────────────────

def evaluate_by_band(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compute RMSE and MAE for each O-level grade band on the test set.

    This directly addresses the business goal: identifying weaker students
    (Fail band) prior to the examination. A model with a low overall RMSE
    but high Fail-band RMSE would be unsuitable for the use case.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline.
    X_test   : test features.
    y_test   : true test scores.

    Returns
    -------
    pd.DataFrame
        Columns: Band, N, Pct, RMSE, MAE
        Sorted by ascending band boundary.
    """
    y_pred = pipeline.predict(X_test)
    y_test = np.array(y_test)
    rows = []

    for band_name, (lo, hi) in SCORE_BANDS.items():
        mask = (y_test >= lo) & (y_test < hi)
        n = mask.sum()
        if n == 0:
            rows.append({
                "Band": band_name, "N": 0,
                "Pct (%)": 0.0, "RMSE": np.nan, "MAE": np.nan,
            })
            continue

        band_rmse = root_mean_squared_error(y_test[mask], y_pred[mask])
        band_mae  = mean_absolute_error(y_test[mask],  y_pred[mask])
        rows.append({
            "Band"   : band_name,
            "N"      : int(n),
            "Pct (%)": round(n / len(y_test) * 100, 1),
            "RMSE"   : round(band_rmse, 4),
            "MAE"    : round(band_mae,  4),
        })

    return pd.DataFrame(rows)


# ── Reporting helpers ──────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    """Print a formatted evaluation summary for a single model.

    Parameters
    ----------
    results : dict
        Output from evaluate_model() (the first return value).
    """
    sep = "-" * 44
    print(sep)
    print(f"  Model      : {results['Model']}")
    print(sep)
    print(f"  Train RMSE : {results['Train RMSE']:.4f}")
    print(f"  CV    RMSE : {results['CV RMSE']:.4f} ± {results['CV RMSE std']:.4f}")
    print(f"  Test  RMSE : {results['Test RMSE']:.4f}")
    print(f"  Test  MAE  : {results['Test MAE']:.4f}")
    print(f"  Test  R²   : {results['Test R2']:.4f}")
    print(f"  Overfit Gap: {results['Overfit Gap']:+.4f}")
    print(sep)


def print_band_report(band_df: pd.DataFrame, model_name: str = "") -> None:
    """Print per-band evaluation results.

    Parameters
    ----------
    band_df    : output from evaluate_by_band().
    model_name : optional label for the header.
    """
    header = f"Per-Band Evaluation{f' — {model_name}' if model_name else ''}"
    print(f"\n  {header}")
    print("  " + "-" * 52)
    print(f"  {'Band':<14} {'N':>5} {'Pct':>6}  {'RMSE':>7}  {'MAE':>7}")
    print("  " + "-" * 52)
    for _, row in band_df.iterrows():
        rmse_str = f"{row['RMSE']:.4f}" if not np.isnan(row['RMSE']) else "   N/A"
        mae_str  = f"{row['MAE']:.4f}"  if not np.isnan(row['MAE'])  else "   N/A"
        print(f"  {row['Band']:<14} {int(row['N']):>5} {row['Pct (%)']:>5.1f}%"
              f"  {rmse_str:>7}  {mae_str:>7}")
    print()


def compare_models(results_list: list) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of evaluate_model() results.

    Parameters
    ----------
    results_list : list of dicts
        Each dict is the first return value of evaluate_model().

    Returns
    -------
    pd.DataFrame
        Sorted by CV RMSE ascending (best model first).
        Duplicate model names are dropped (last entry wins).
    """
    df = (
        pd.DataFrame(results_list)
        .drop_duplicates(subset="Model", keep="last")
        .sort_values("CV RMSE")
        .reset_index(drop=True)
    )
    return df
