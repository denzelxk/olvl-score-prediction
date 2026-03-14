"""
4_model_info.py — Model Information & Transparency
Shows the model comparison table, feature importances (RF/GB),
and Ridge coefficients so teachers can understand what drives predictions.
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
from app.utils.charts import feature_importance_bar
from app.utils.loader import MODEL_OPTIONS, DEFAULT_MODEL, load_model, _ROOT

st.set_page_config(page_title="Model Info — UA Secondary", page_icon="📈", layout="wide")
st.title("📈 Model Information & Transparency")
st.caption("Understanding what the models learned and how accurate they are.")

# ── 1. Model comparison table ─────────────────────────────────────────────────
st.subheader("1. Model Performance (held-out test set)")
results_path = os.path.join(_ROOT, "results", "results.csv")
if os.path.exists(results_path):
    res = pd.read_csv(results_path)
    display_cols = [c for c in ["Model","CV RMSE","CV RMSE std","Test RMSE",
                                  "Test MAE","Test R2","Overfit Gap"] if c in res.columns]
    st.dataframe(res[display_cols].set_index("Model") if "Model" in res.columns else res,
                 use_container_width=True)
    st.caption(
        "**RMSE** (Root Mean Squared Error): average prediction error in score points. "
        "Lower is better. A Dummy model that always predicts the mean scores RMSE=13.99. "
        "**R²**: fraction of score variance explained (1.0 = perfect, 0.0 = Dummy). "
        "**Overfit Gap**: Train RMSE − Test RMSE; near zero is ideal."
    )
else:
    st.warning(f"results.csv not found at {results_path}. Run `bash run.sh --model all` first.")

st.markdown("---")

# ── 2. Feature importances ────────────────────────────────────────────────────
st.subheader("2. What drives predictions?")

col_rf, col_gb, col_ridge = st.columns(3)

for col, key, title in [
    (col_rf,    "random_forest",     "Random Forest — Feature Importances"),
    (col_gb,    "gradient_boosting", "Gradient Boosting — Feature Importances"),
    (col_ridge, "ridge",             "Ridge Regression — Coefficients"),
]:
    with col:
        try:
            pipeline = load_model(key)
            model    = pipeline.named_steps["model"]
            prep     = pipeline.named_steps["preprocessor"]

            # Recover feature names after OHE
            ohe       = prep.named_transformers_["cat"]["ohe"]
            cat_names = list(ohe.get_feature_names_out())
            from src.preprocessor import NUMERICAL_COLS, CATEGORICAL_COLS, BINARY_COLS
            feat_names = NUMERICAL_COLS + cat_names + BINARY_COLS

            if hasattr(model, "feature_importances_"):
                vals = model.feature_importances_
            elif hasattr(model, "coef_"):
                vals = model.coef_
            else:
                st.info(f"No importances available for {key}.")
                continue

            imp_df = pd.DataFrame({"feature": feat_names, "importance": vals})
            st.plotly_chart(feature_importance_bar(imp_df, title=title),
                            use_container_width=True)
        except FileNotFoundError:
            st.warning(f"{key}.pkl not found. Run `bash run.sh --model {key}` to generate.")
        except Exception as e:
            st.error(f"Could not load {key}: {e}")

st.markdown("---")

# ── 3. Plain-English guide ────────────────────────────────────────────────────
st.subheader("3. How to interpret these results")
st.markdown("""
| Feature | Direction | What it means |
|---|---|---|
| `class_size` | Negative | Larger classes → lower predicted scores (less individual attention) |
| `number_of_siblings` | Negative | More siblings → lower scores (less quiet study time at home) |
| `learning_style_Visual` | Positive | Visual learners tend to score higher, controlling for other variables |
| `tuition` | Positive | Private tuition adds ~4 predicted score points, all else equal |
| `hours_per_week` | Negative | Counterintuitive — weaker students compensate by studying more (Simpson's paradox) |
| `attendance_rate` | Positive | Higher attendance → higher predicted score |
| `CCA_None` | Positive | Students with no CCA tend to score higher (more study time) |

> These patterns come from historical data. They describe correlations, not
> guaranteed causal effects. The most actionable lever is **tuition** (+4 pts on average)
> and **attendance rate** — both can be directly influenced.
""")
