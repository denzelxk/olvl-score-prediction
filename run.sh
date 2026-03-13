#!/bin/bash
# =============================================================================
# run.sh  —  Student Score Prediction Pipeline entry point
# =============================================================================
# Usage:
#   bash run.sh                                    run default model (gradient_boosting)
#   bash run.sh --model all                        train all three models
#   bash run.sh --model gradient_boosting          train Gradient Boosting only
#   bash run.sh --model random_forest              train Random Forest only
#   bash run.sh --model ridge                      train Ridge Regression only
#   bash run.sh --config config.yaml               explicit config path
#   bash run.sh --db-path data/score.db            override database path
#
# First-time setup:
#   python -m venv .venv
#   source .venv/bin/activate        # Windows: .venv\Scripts\activate
#   pip install -r requirements.txt
# =============================================================================

set -e   # exit immediately on any error

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

python src/run.py "$@"
