import sys, os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from app.utils.charts import feature_snapshot, score_band_bar, score_histogram

st.set_page_config(page_title="Dashboard — UA Secondary", page_icon="📊", layout="wide")
st.title("📊 Class Dashboard")

if "predictions_df" not in st.session_state:
    st.info(
        "No class data loaded yet. Upload a CSV or run a manual prediction first.",
        icon="📂",
    )
    st.page_link("pages/1_predict.py", label="Go to Predict →", icon="🔮")
    st.stop()

df        = st.session_state["predictions_df"]
model_key = st.session_state.get("model_key", "gradient_boosting")
st.caption(f"**{len(df)} students** · Model: **{model_key.replace('_', ' ').title()}**")

n_fail = (df["score_band"] == "Fail").sum()
n_c    = (df["score_band"] == "C").sum()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Students",    len(df))
m2.metric("Mean Score",        f"{df['predicted_score'].mean():.1f}")
m3.metric("Median Score",      f"{df['predicted_score'].median():.1f}")
m4.metric("At-Risk (Fail)",    n_fail, delta=f"{n_fail/len(df)*100:.0f}%", delta_color="inverse")
m5.metric("Needs Support (C)", n_c,    delta=f"{n_c/len(df)*100:.0f}%",    delta_color="inverse")

st.markdown("---")
cl, cr = st.columns([3, 2])
with cl:
    st.plotly_chart(score_histogram(df, title=f"Score Distribution (n={len(df)})"), use_container_width=True)
with cr:
    st.plotly_chart(score_band_bar(df), use_container_width=True)

st.markdown("---")
cr2, cs = st.columns([2, 3])
with cr2:
    st.subheader("⚠️ At-Risk Students (< 50)")
    at_risk = df[df["score_band"] == "Fail"].sort_values("predicted_score")
    if at_risk.empty:
        st.success("No students predicted to fail. 🎉")
    else:
        disp = (["student_id"] if "student_id" in at_risk.columns else []) + ["predicted_score", "score_band"]
        for c in ["tuition", "hours_per_week", "attendance_rate"]:
            if c in at_risk.columns:
                disp.append(c)
        st.dataframe(at_risk[disp].reset_index(drop=True), use_container_width=True, height=320)
        st.download_button(
            "⬇️ Download At-Risk List",
            data=at_risk.to_csv(index=False).encode(),
            file_name="at_risk_students.csv",
            mime="text/csv",
        )
        st.info(f"💡 Use **🔍 What-If** to simulate interventions for these {n_fail} students.")
with cs:
    if {"hours_per_week", "attendance_rate", "tuition"}.issubset(df.columns):
        st.plotly_chart(feature_snapshot(df), use_container_width=True)

with st.expander("📋 Full Predictions Table"):
    st.dataframe(
        df.sort_values("predicted_score").reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
