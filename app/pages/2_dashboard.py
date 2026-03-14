"""
2_dashboard.py — Class Dashboard
Requires st.session_state["predictions_df"] set by 1_predict.py.
"""
import pandas as pd
import streamlit as st
from app.utils.charts import feature_snapshot, score_band_bar, score_histogram
from app.utils.loader import BAND_COLORS

st.set_page_config(page_title="Dashboard — UA Secondary", page_icon="📊", layout="wide")
st.title("📊 Class Dashboard")

if "predictions_df" not in st.session_state:
    st.warning("No class data loaded yet. Go to **🔮 Predict** and upload a class CSV first.")
    st.stop()

df        = st.session_state["predictions_df"]
model_key = st.session_state.get("model_key", "gradient_boosting")
st.caption(f"**{len(df)} students** · Model: **{model_key.replace('_',' ').title()}**")

n_fail = (df["score_band"] == "Fail").sum()
n_c    = (df["score_band"] == "C").sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Students",    len(df))
m2.metric("Mean Score",        f"{df['predicted_score'].mean():.1f}")
m3.metric("Median Score",      f"{df['predicted_score'].median():.1f}")
m4.metric("At-Risk (Fail)",    n_fail,  delta=f"{n_fail/len(df)*100:.0f}%", delta_color="inverse")
m5.metric("Needs Support (C)", n_c,     delta=f"{n_c/len(df)*100:.0f}%",    delta_color="inverse")

st.markdown("---")
col_l, col_r = st.columns([3, 2])
with col_l:
    st.plotly_chart(score_histogram(df, title=f"Predicted Score Distribution (n={len(df)})"),
                    use_container_width=True)
with col_r:
    st.plotly_chart(score_band_bar(df), use_container_width=True)

st.markdown("---")
col_risk, col_snap = st.columns([2, 3])

with col_risk:
    st.subheader("⚠️ At-Risk Students (predicted < 50)")
    at_risk = df[df["score_band"] == "Fail"].sort_values("predicted_score")
    if at_risk.empty:
        st.success("No students are predicted to fail. 🎉")
    else:
        disp = (["student_id"] if "student_id" in at_risk.columns else []) + ["predicted_score","score_band"]
        for c in ["tuition","hours_per_week","attendance_rate"]:
            if c in at_risk.columns: disp.append(c)
        st.dataframe(at_risk[disp].reset_index(drop=True), use_container_width=True, height=320)
        st.download_button("⬇️ Download At-Risk List",
                           data=at_risk.to_csv(index=False).encode(),
                           file_name="at_risk_students.csv", mime="text/csv")
        st.info(f"💡 {n_fail} student(s) at risk. Use **🔍 What-If** to simulate interventions.")

with col_snap:
    raw_cols = {"hours_per_week","attendance_rate","tuition"}
    if raw_cols.issubset(df.columns):
        st.plotly_chart(feature_snapshot(df), use_container_width=True)
    else:
        st.info("Feature snapshot unavailable — raw columns not retained in predictions DataFrame.")

with st.expander("📋 Full Predictions Table", expanded=False):
    sort_col = st.selectbox("Sort by", ["predicted_score","score_band"])
    st.dataframe(df.sort_values(sort_col).reset_index(drop=True),
                 use_container_width=True, height=400)
