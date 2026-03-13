"""
models.py
---------
Model registry and pipeline builder for the student score prediction pipeline.

Each model is built as a sklearn Pipeline (preprocessor + estimator). Hyperparameters
are drawn from two sources, in priority order:
  1. Values passed at runtime from config.yaml (via run.py)
  2. Hardcoded defaults here — the locked results from prototype.ipynb tuning

This means the pipeline is fully configurable via config.yaml without touching
any .py files, satisfying the assessment's configurability requirement.

Locked hyperparameters (prototype.ipynb §3b, §3d):
  Ridge          : RidgeCV, best alpha selected by inner CV over [0.01 .. 1000]
  Random Forest  : n_estimators=326, max_depth=15, max_features=0.5, min_samples_leaf=1
  Gradient Boost : n_estimators=548, learning_rate=0.01, max_depth=8,
                   min_samples_leaf=6, subsample=0.8

Public API
----------
build_pipeline(name, preprocessor, params) -> sklearn.pipeline.Pipeline
AVAILABLE_MODELS : list[str]
MODEL_DISPLAY_NAMES : dict[str, str]
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

# ── Registry ───────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = ["ridge", "random_forest", "gradient_boosting"]

MODEL_DISPLAY_NAMES = {
    "ridge":               "Ridge Regression (RidgeCV)",
    "random_forest":       "Random Forest Regressor",
    "gradient_boosting":   "Gradient Boosting Regressor",
}

# Default hyperparameters — locked from prototype.ipynb tuning runs.
# These are overridden by any matching keys in config.yaml at runtime.
_DEFAULTS = {
    "ridge": {
        "alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "cv": 5,
    },
    "random_forest": {
        "n_estimators":     326,
        "max_depth":        15,
        "max_features":     0.5,
        "min_samples_leaf": 1,
        "n_jobs":           -1,
        "random_state":     42,
    },
    "gradient_boosting": {
        "n_estimators":     548,
        "learning_rate":    0.01,
        "max_depth":        8,
        "min_samples_leaf": 6,
        "subsample":        0.8,
        "random_state":     42,
    },
}


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_estimator(name: str, params: dict):
    """Instantiate a sklearn estimator from its registry name and param dict."""
    if name == "ridge":
        return RidgeCV(**params)
    elif name == "random_forest":
        return RandomForestRegressor(**params)
    elif name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    raise ValueError(
        f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}"
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_pipeline(
    name: str,
    preprocessor: ColumnTransformer,
    params: dict = None,
) -> Pipeline:
    """Build an unfitted sklearn Pipeline for the named model.

    Hyperparameters are resolved by merging the locked defaults from
    prototype.ipynb with any overrides supplied via *params* (typically
    read from config.yaml). This allows full runtime configurability
    without modifying source files.

    Parameters
    ----------
    name : str
        Model identifier. Must be one of AVAILABLE_MODELS.
    preprocessor : ColumnTransformer
        An *unfitted* preprocessor from preprocessor.build_preprocessor().
        A fresh instance must be passed for each pipeline to prevent
        state sharing between models.
    params : dict, optional
        Hyperparameter overrides from config.yaml. Keys must match the
        sklearn estimator's __init__ parameter names exactly.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted two-step pipeline: ('preprocessor', ...) + ('model', ...).

    Raises
    ------
    ValueError
        If *name* is not in AVAILABLE_MODELS.

    Examples
    --------
    >>> from preprocessor import build_preprocessor
    >>> from models import build_pipeline
    >>> pipe = build_pipeline("gradient_boosting", build_preprocessor())
    >>> pipe.fit(X_train, y_train)
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}"
        )

    # Config overrides take priority; fall back to locked prototype defaults
    merged = {**_DEFAULTS[name], **(params or {})}
    estimator = _build_estimator(name, merged)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",        estimator),
    ])
