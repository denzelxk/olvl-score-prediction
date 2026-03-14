import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="UA Secondary — Score Predictor",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 U.A. Secondary School")
st.subheader("O-Level Mathematics Score Prediction System")
st.markdown("""
This tool helps teachers identify students who may need additional support
before their O-level Mathematics examination.

---

### How to use

| Step | Page | What to do |
|------|------|------------|
| 1 | **🔮 Predict** | Upload your class CSV or enter a student manually to get predicted scores |
| 2 | **📊 Dashboard** | View your class score distribution and the at-risk student list |
| 3 | **🔍 What-If** | Simulate how interventions (tuition, study hours) affect a student's score |
| 4 | **📈 Model Info** | Understand the models and what drives predictions |

Use the **sidebar** on the left to navigate between pages.

---

### About the models

Predictions are made by three machine learning models trained on historical
student data. The default model is **Gradient Boosting** (Test RMSE: 5.28,
R²: 0.858). Switch to **Ridge Regression** on the Predict page for a
plain-English explanation of why a score was predicted.

> ⚠️ Predictions are estimates based on historical patterns. They should
> inform — not replace — teacher judgement.
""")
