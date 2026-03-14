"""
loader.py
---------
Model loading and inference utilities for the Streamlit web application.

Imports feature engineering from src.data_loader and column constants from
src.preprocessor as SSOT — no preprocessing logic is duplicated here.

Public API
----------
load_model(model_key)           -> fitted sklearn Pipeline  (cached)
preprocess_for_inference(df)    -> model-ready DataFrame
predict(df, model_key)          -> DataFrame with predicted_score + score_band columns
score_to_band(score)            -> str  (Fail / C / B / A2 / A1)
get_csv_template()              -> pd.DataFrame  (empty template for download)

Constants
---------
BAND_ORDER, BAND_COLORS, MODEL_OPTIONS, INPUT_COLS, DEFAULT_MODEL
"""

import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup: allow imports from project root ───────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_loader import (
    AGE_FIX_MAP,
    TUITION_MAP,
    VALID_AGES,
    engineer_features,
)
from src.models import AVAILABLE_MODELS, MODEL_DISPLAY_NAMES
from src.preprocessor import BINARY_COLS, CATEGORICAL_COLS, NUMERICAL_COLS

# ── Constants ─────────────────────────────────────────────────────────────────

_MODELS_DIR = os.path.join(_ROOT, "models")
DEFAULT_MODEL = "gradient_boosting"

BAND_ORDER  = ["Fail", "C", "B", "A2", "A1"]
BAND_COLORS = {
    "Fail": "#d62728",
    "C":    "#ff7f0e",
    "B":    "#FFCD56",
    "A2":   "#2ca02c",
    "A1":   "#1f77b4",
}

# Display label → model key mapping for the UI selector
MODEL_OPTIONS = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}

# Raw input columns expected in a teacher-uploaded CSV (post-rename, pre-engineering)
INPUT_COLS = [
    "student_id",          # optional — used for display only
    "number_of_siblings",
    "direct_admission",
    "CCA",
    "learning_style",
    "gender",
    "tuition",
    "n_male",
    "n_female",
    "age",
    "hours_per_week",
    "attendance_rate",
    "sleep_time",          # HH:MM  e.g. "22:00"
    "wake_time",           # HH:MM  e.g. "06:00"
    "mode_of_transport",
]
REQUIRED_INPUT_COLS = [c for c in INPUT_COLS if c != "student_id"]


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(model_key: str):
    """Load a fitted pipeline from disk. Cached for the Streamlit session.

    Parameters
    ----------
    model_key : str
        One of AVAILABLE_MODELS: 'ridge', 'random_forest', 'gradient_boosting'.

    Raises
    ------
    FileNotFoundError
        If the .pkl file does not exist. Run 'bash run.sh' to generate models.
    """
    path = os.path.join(_MODELS_DIR, f"{model_key}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path!r}.\n"
            "Run 'bash run.sh --model all' to train and save all models."
        )
    return joblib.load(path)


# ── Score band helpers ────────────────────────────────────────────────────────

def score_to_band(score: float) -> str:
    """Map a predicted score to its O-level grade band."""
    s = float(score)
    if s < 50:  return "Fail"
    if s < 60:  return "C"
    if s < 70:  return "B"
    if s < 80:  return "A2"
    return "A1"


# ── Preprocessing for inference ───────────────────────────────────────────────

def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features for an inference DataFrame.

    Applies the same transformations as data_loader.clean() and
    data_loader.engineer_features(), minus the target-dependent steps
    (no final_test column is expected at inference time).

    Steps
    -----
    1. Map tuition to binary int (Yes/Y -> 1, No/N -> 0).
    2. Capitalise CCA values (resolves mixed-case from teacher input).
    3. Fix erroneous age values using AGE_FIX_MAP from data_loader.
    4. Call engineer_features() to add sleep_duration, class_size, male_ratio
       and drop sleep_time, wake_time, n_male, n_female.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame with INPUT_COLS columns. student_id may be present
        or absent; it is ignored during prediction.

    Returns
    -------
    pd.DataFrame
        Model-ready DataFrame containing NUMERICAL_COLS + CATEGORICAL_COLS +
        BINARY_COLS. Suitable for pipeline.predict().
    """
    df = df.copy()

    # 1. Tuition → binary
    if df["tuition"].dtype == object or df["tuition"].dtype.name == "string":
        df["tuition"] = df["tuition"].map(TUITION_MAP)
    df["tuition"] = pd.to_numeric(df["tuition"], errors="coerce").fillna(0).astype(int)

    # 2. CCA capitalisation
    df["CCA"] = df["CCA"].astype(str).str.strip().str.capitalize()

    # 3. Age fix
    df["age"] = pd.to_numeric(df["age"], errors="coerce").replace(AGE_FIX_MAP)

    # 4. Feature engineering (sleep_duration, class_size, male_ratio)
    df = engineer_features(df)

    return df


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, model_key: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Run end-to-end inference: preprocess -> pipeline.predict -> annotate bands.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame with REQUIRED_INPUT_COLS columns.
    model_key : str
        Model to use for prediction.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two appended columns:
        - predicted_score  : float, rounded to 1 dp
        - score_band       : str  (Fail / C / B / A2 / A1)
    """
    pipeline = load_model(model_key)
    processed = preprocess_for_inference(df)

    # Keep only the feature columns the pipeline expects; drop ID if present
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS + BINARY_COLS
    X = processed[[c for c in feature_cols if c in processed.columns]]

    scores = pipeline.predict(X)

    result = df.copy()
    result["predicted_score"] = np.round(scores, 1)
    result["score_band"]      = result["predicted_score"].apply(score_to_band)
    return result


# ── CSV template ──────────────────────────────────────────────────────────────

def get_csv_template() -> pd.DataFrame:
    """Return an empty template DataFrame that matches the expected upload format.

    Teachers download this, fill it in with their class data, and re-upload it.
    """
    example = {
        "student_id":         ["S001", "S002"],
        "number_of_siblings": [1,      0     ],
        "direct_admission":   ["No",   "Yes" ],
        "CCA":                ["Sports","None"],
        "learning_style":     ["Visual","Auditory"],
        "gender":             ["Female","Male"],
        "tuition":            ["Yes",   "No"  ],
        "n_male":             [14,      20    ],
        "n_female":           [10,      8     ],
        "age":                [15,      16    ],
        "hours_per_week":     [10,      8     ],
        "attendance_rate":    [95.0,    88.0  ],
        "sleep_time":         ["22:00", "23:00"],
        "wake_time":          ["06:00", "06:30"],
        "mode_of_transport":  ["private transport",   "walk" ],
    }
    return pd.DataFrame(example)
