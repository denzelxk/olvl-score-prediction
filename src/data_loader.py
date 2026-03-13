"""
data_loader.py
--------------
Data ingestion, cleaning, and feature engineering for the student score
prediction pipeline.

All logic mirrors eda.ipynb (Sections 3 and 6) exactly so that the pipeline
and the EDA notebook remain consistent.

Public API
----------
load_data(db_path)  ->  pd.DataFrame
    Full pipeline: load_raw -> clean -> engineer_features.
    Returns a model-ready DataFrame with target column 'final_test'.
"""

import logging
import sqlite3

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_COL   = "final_test"
DROP_COLS    = ["index", "bag_color"]
VALID_AGES   = {15, 16}
AGE_FIX_MAP  = {5: 15, 6: 16, -5: 15}
TUITION_MAP  = {"Yes": 1, "Y": 1, "No": 0, "N": 0}
TIME_COLS    = ("sleep_time", "wake_time")
CLASS_COLS   = ("n_male", "n_female")


# ── Private helpers ────────────────────────────────────────────────────────────

def _parse_time_to_minutes(t) -> float:
    """Convert 'HH:MM' string to total minutes since midnight.

    Returns np.nan for any unparseable value.
    """
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return np.nan


# ── Public functions ───────────────────────────────────────────────────────────

def load_raw(db_path: str = "data/score.db") -> pd.DataFrame:
    """Load the raw table from a SQLite database.

    Automatically detects the first available table, so the function remains
    correct if the table name changes between dataset versions.

    Parameters
    ----------
    db_path : str
        Relative or absolute path to the .db file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified DataFrame as stored in the database.

    Raises
    ------
    FileNotFoundError
        If *db_path* does not exist.
    RuntimeError
        If the database contains no tables.
    """
    if not __import__("os").path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path!r}")

    conn = sqlite3.connect(db_path)
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        )
        if tables.empty:
            raise RuntimeError(f"No tables found in database: {db_path!r}")

        table_name = tables["name"].iloc[0]
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    finally:
        conn.close()

    logger.info("Loaded raw data: %d rows x %d columns from '%s'",
                df.shape[0], df.shape[1], table_name)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps documented in eda.ipynb Section 3.

    Steps (in order):
    1. Drop rows where the target (final_test) is missing.
    2. Drop uninformative columns: 'index' and 'bag_color'.
       - 'index' is an auto-generated row number.
       - 'bag_color' was found to be the sole source of non-exact duplicate
         student records and carries no predictive signal (eda.ipynb §3).
    3. Drop exact duplicate rows (identical across all remaining columns).
    4. Fix erroneous age values: {5->15, 6->16, -5->15} are clear typos.
    5. Filter to valid ages {15, 16} — removes 4 rows with age=-4 that
       cannot be confidently mapped.
    6. Standardise 'tuition' to binary int: Yes/Y->1, No/N->0.
    7. Capitalise 'CCA' values to resolve mixed-case duplicates
       (e.g. 'SPORTS' and 'Sports' are the same category).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with reset index.
    """
    df = df.copy()
    n0 = len(df)

    # 1. Drop missing target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    logger.info("Dropped %d rows with missing target.", n0 - len(df))

    # 2. Drop uninformative columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 3. Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d exact duplicate rows.", n_before - len(df))

    # 4. Fix erroneous age typos
    df["age"] = df["age"].replace(AGE_FIX_MAP)

    # 5. Filter to valid ages
    n_before = len(df)
    df = df[df["age"].isin(VALID_AGES)].reset_index(drop=True)
    logger.info("Removed %d rows with unresolvable age values.", n_before - len(df))

    # 6. Standardise tuition encoding
    df["tuition"] = df["tuition"].map(TUITION_MAP)

    # 7. Normalise CCA capitalisation
    df["CCA"] = df["CCA"].str.strip().str.capitalize()

    logger.info("Cleaning complete: %d rows x %d columns.", df.shape[0], df.shape[1])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering documented in eda.ipynb Section 6.

    Engineered features
    -------------------
    sleep_duration : float
        Hours of sleep per night, derived from sleep_time and wake_time.
        Computed as (wake_min - sleep_min) % 1440 / 60 to correctly handle
        overnight sleep (e.g. sleep at 23:00, wake at 06:00 = 7 h).
    class_size : int
        Total number of students in the class: n_male + n_female.
    male_ratio : float
        Fraction of male students: n_male / class_size.
        Returns np.nan for classes with class_size = 0 (edge-case guard).

    Raw columns removed after engineering
    --------------------------------------
    sleep_time, wake_time, n_male, n_female

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean().

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features and raw source columns removed.
    """
    df = df.copy()

    # sleep_duration
    if all(c in df.columns for c in TIME_COLS):
        sleep_min = df["sleep_time"].apply(_parse_time_to_minutes)
        wake_min  = df["wake_time"].apply(_parse_time_to_minutes)
        df["sleep_duration"] = (wake_min - sleep_min) % (24 * 60) / 60
        df = df.drop(columns=["sleep_time", "wake_time"])
        logger.info("Engineered 'sleep_duration'.")

    # class_size and male_ratio
    if all(c in df.columns for c in CLASS_COLS):
        df["class_size"] = df["n_male"] + df["n_female"]
        df["male_ratio"]  = df["n_male"] / df["class_size"].replace(0, np.nan)
        df = df.drop(columns=["n_male", "n_female"])
        logger.info("Engineered 'class_size' and 'male_ratio'.")

    logger.info("Feature engineering complete: %d columns.", df.shape[1])
    return df


def load_data(db_path: str = "data/score.db") -> pd.DataFrame:
    """Full ingestion pipeline: load -> clean -> engineer_features.

    This is the single entry point used by run.py. Calling this function
    guarantees the exact same transformations applied in eda.ipynb and
    prototype.ipynb.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    pd.DataFrame
        Model-ready DataFrame containing 'final_test' as the target column
        and all engineered features. student_id is retained for traceability
        but must be excluded from the feature matrix (X) before training.
    """
    df = load_raw(db_path)
    df = clean(df)
    df = engineer_features(df)
    logger.info("load_data complete: %d rows x %d columns.", df.shape[0], df.shape[1])
    return df
