"""
run.py
------
Main entry point for the student score prediction pipeline.

Orchestrates the full workflow:
  1. Load config.yaml
  2. Ingest and prepare data  (data_loader.load_data)
  3. Stratified train/test split
  4. For each selected model:
       a. Build pipeline      (models.build_pipeline)
       b. Train + evaluate    (evaluate.evaluate_model)
       c. Per-band evaluation (evaluate.evaluate_by_band)
       d. Save fitted pipeline to disk (joblib)
  5. Print comparison table if multiple models were trained
  6. Save results.csv to results/

Usage
-----
    bash run.sh                                  # default model from config.yaml
    bash run.sh --model all                      # train all three models
    bash run.sh --model gradient_boosting        # train GB only
    bash run.sh --model random_forest            # train RF only
    bash run.sh --model ridge                    # train Ridge only
    bash run.sh --config config.yaml             # explicit config path
    bash run.sh --db-path data/score.db          # override DB path
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Ensure src/ modules are importable when called as `python src/run.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from evaluate import (
    compare_models,
    evaluate_by_band,
    evaluate_model,
    print_band_report,
    print_report,
)
from models import AVAILABLE_MODELS, MODEL_DISPLAY_NAMES, build_pipeline
from preprocessor import build_preprocessor

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config.yaml"


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load and parse a YAML configuration file.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path!r}. "
            "Run from the project root or pass --config <path>."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(config: dict):
    """Load, clean, engineer features, and split into train/test sets.

    Uses a stratified split so the Fail-band students (<50, ~14% of data)
    are proportionally represented in both sets — mirrors prototype.ipynb §3.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    df = load_data(config["data"]["db_path"])

    X = df.drop(columns=["final_test", "student_id"], errors="ignore")
    y = df["final_test"]

    train_cfg = config["train"]
    bins   = train_cfg.get("stratify_bins", [0, 50, 60, 70, 80, 100])
    strata = pd.cut(y, bins=bins, labels=range(len(bins) - 1), include_lowest=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = train_cfg.get("test_size",    0.20),
        random_state = train_cfg.get("random_state", 42),
        stratify     = strata,
    )

    logger.info(
        "Split complete — train: %d rows | test: %d rows | "
        "target mean: train=%.2f  test=%.2f",
        len(X_train), len(X_test), y_train.mean(), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_model(pipeline, name: str, models_dir: str) -> str:
    """Serialise a fitted pipeline to disk with joblib.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    name       : model identifier used as the filename stem
    models_dir : directory to write to (created if absent)

    Returns
    -------
    str : absolute path of the saved file
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(pipeline, path)
    logger.info("Model saved -> %s", os.path.abspath(path))
    return path


def save_results(results_list: list, results_dir: str) -> str:
    """Write evaluation results to a CSV file.

    Parameters
    ----------
    results_list : list of dicts from evaluate.evaluate_model()
    results_dir  : directory to write to (created if absent)

    Returns
    -------
    str : absolute path of the saved CSV
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "results.csv")
    pd.DataFrame(results_list).to_csv(path, index=False)
    logger.info("Results saved -> %s", os.path.abspath(path))
    return path


# ── Core training loop ────────────────────────────────────────────────────────

def run(model_names: list, config: dict) -> None:
    """Train, evaluate, and save each model in *model_names*.

    For each model:
      - Builds a fresh preprocessor (prevents state sharing between models)
      - Merges config.yaml hyperparameters with locked prototype defaults
      - Runs evaluate_model (train + CV + test RMSE / MAE / R²)
      - Runs evaluate_by_band (per O-level grade band breakdown)
      - Saves the fitted pipeline to models/<name>.pkl

    If more than one model is trained, prints a sorted comparison table.
    All results are written to results/results.csv.

    Parameters
    ----------
    model_names : list[str]
        Subset of AVAILABLE_MODELS to train.
    config : dict
        Parsed config.yaml.
    """
    X_train, X_test, y_train, y_test = prepare_data(config)

    train_cfg   = config["train"]
    models_dir  = config["output"]["models_dir"]
    results_dir = config["output"]["results_dir"]
    cv          = train_cfg.get("cv_folds", 5)

    all_results = []

    for name in model_names:
        display = MODEL_DISPLAY_NAMES.get(name, name)
        logger.info("Training: %s", display)

        # Fresh preprocessor instance for each model — prevents state leakage
        preprocessor  = build_preprocessor()
        model_params  = config.get("models", {}).get(name, {})
        pipeline      = build_pipeline(name, preprocessor, model_params)

        results, fitted_pipeline = evaluate_model(
            name     = display,
            pipeline = pipeline,
            X_train  = X_train, y_train = y_train,
            X_test   = X_test,  y_test  = y_test,
            cv       = cv,
        )
        all_results.append(results)

        print_report(results)

        band_df = evaluate_by_band(fitted_pipeline, X_test, y_test)
        print_band_report(band_df, model_name=display)

        save_model(fitted_pipeline, name, models_dir)
        print()

    if len(all_results) > 1:
        sep = "=" * 60
        print(sep)
        print("  FINAL MODEL COMPARISON  (sorted by CV RMSE ascending)")
        print(sep)
        print(compare_models(all_results).to_string(index=False))
        print()

    save_results(all_results, results_dir)
    logger.info("Pipeline complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    valid = AVAILABLE_MODELS + ["all"]
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Student Score Prediction Pipeline — AIAP Technical Assessment\n"
            "U.A Secondary School | O-level Mathematics Score Prediction"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/run.py                            # default model\n"
            "  python src/run.py --model all                # all three models\n"
            "  python src/run.py --model gradient_boosting  # GB only\n"
            "  python src/run.py --model random_forest      # RF only\n"
            "  python src/run.py --model ridge              # Ridge only\n"
            "  python src/run.py --db-path data/score.db   # override DB path\n"
        ),
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = None,
        metavar = "{" + "|".join(valid) + "}",
        help    = (
            "Model to train. "
            f"Options: {', '.join(valid)}. "
            "Defaults to 'default_model' in config.yaml."
        ),
    )
    parser.add_argument(
        "--config",
        type    = str,
        default = DEFAULT_CONFIG,
        metavar = "PATH",
        help    = f"Path to YAML config file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--db-path",
        type    = str,
        default = None,
        metavar = "PATH",
        help    = "Override the db_path set in config.yaml.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.db_path:
        config["data"]["db_path"] = args.db_path
        logger.info("DB path overridden via CLI: %s", args.db_path)

    # Resolve which models to train
    model_arg = args.model or config.get("default_model", "gradient_boosting")

    if model_arg == "all":
        model_names = AVAILABLE_MODELS
    elif model_arg in AVAILABLE_MODELS:
        model_names = [model_arg]
    else:
        parser.error(
            f"Unknown model {model_arg!r}. "
            f"Choose from: {', '.join(AVAILABLE_MODELS + ['all'])}"
        )

    logger.info(
        "Running pipeline | model(s): %s | config: %s | db: %s",
        ", ".join(model_names),
        args.config,
        config["data"]["db_path"],
    )

    run(model_names, config)


if __name__ == "__main__":
    main()
