"""
preprocessor.py
---------------
Defines and builds the shared ColumnTransformer used by all models in the
pipeline.

The feature routing here is the direct implementation of the decisions made
in eda.ipynb (Sections 4, 7, 9) and prototype.ipynb (Section 2):

  Numerical (7)  : median imputation + StandardScaler
  Categorical (5): mode imputation  + OneHotEncoder(drop='first')
  Binary (1)     : passthrough (already 0/1 integers from data_loader)

Key design decisions (justified in eda.ipynb)
---------------------------------------------
- StandardScaler for numericals: no numerical feature has outlier % > 5%
  (eda.ipynb §10), so RobustScaler provides no meaningful advantage.
- OHE drop='first': avoids dummy variable trap for linear models; tree
  models are unaffected by the redundant reference category.
- All 16 post-OHE features retained: Lasso at optimal alpha=0.01 zeroed
  none, confirming every feature carries signal (prototype.ipynb §1c).
- sleep_duration low-variance note: std ≈ 0.60 with IQR = 0. StandardScaler
  handles this safely (no division by zero). See prototype.ipynb §2.

Public API
----------
build_preprocessor() -> ColumnTransformer
    Returns an *unfitted* ColumnTransformer ready to be embedded in a
    sklearn Pipeline.

NUMERICAL_COLS, CATEGORICAL_COLS, BINARY_COLS
    Module-level lists used by models.py to reconstruct feature names after
    OHE for coefficient / importance reporting.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Feature column definitions ─────────────────────────────────────────────────
# These are post-engineering column names (after data_loader.engineer_features).

NUMERICAL_COLS = [
    "age",
    "hours_per_week",
    "attendance_rate",
    "number_of_siblings",
    "sleep_duration",
    "class_size",
    "male_ratio",
]

CATEGORICAL_COLS = [
    "direct_admission",
    "CCA",
    "learning_style",
    "gender",
    "mode_of_transport",
]

BINARY_COLS = [
    "tuition",  # already mapped to {0, 1} by data_loader.clean()
]


# ── Builder ────────────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for the full feature set.

    The transformer is intentionally returned unfitted so it can be composed
    inside a sklearn Pipeline. Fitting happens only on X_train inside run.py,
    preventing data leakage from the test split.

    Returns
    -------
    ColumnTransformer
        A configured (unfitted) sklearn ColumnTransformer with three
        transformer groups: numerical, categorical, and binary passthrough.

    Notes
    -----
    - ``remainder='drop'`` silently ignores any unexpected columns (e.g.
      student_id) that may slip through if the caller forgets to drop them.
    - ``sparse_output=False`` on OneHotEncoder ensures a dense array output,
      which is required by GradientBoostingRegressor (no sparse support).
    - ``handle_unknown='ignore'`` on OHE ensures that unseen categories at
      inference time produce all-zero rows rather than raising an error.
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer,  NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("bin", "passthrough",           BINARY_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list:
    """Return ordered feature names after OHE expansion.

    Useful for coefficient and feature importance reporting in evaluate.py.

    Parameters
    ----------
    fitted_preprocessor : ColumnTransformer
        A *fitted* ColumnTransformer (i.e. after pipeline.fit() has been
        called).

    Returns
    -------
    list of str
        Feature names in the same column order as the transformed array:
        [numerical_cols..., ohe_expanded_categorical_cols..., binary_cols...]
    """
    ohe = fitted_preprocessor.named_transformers_["cat"]["ohe"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    return NUMERICAL_COLS + cat_feature_names + BINARY_COLS
