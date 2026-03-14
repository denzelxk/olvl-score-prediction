import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st
from app.utils.loader import (DEFAULT_MODEL, MODEL_OPTIONS, BAND_COLORS,
                               get_csv_template, predict, score_to_band)
from app.utils.validator import validate_csv

st.set_page_config(page_title="Predict — UA Secondary", page_icon="🔮", layout="wide")
st.title("🔮 Predict Student Scores")

with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("Select model", list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(
            next(k for k,v in MODEL_OPTIONS.items() if v == DEFAULT_MODEL)),
        help="GB = most accurate | RF = stable | Ridge = interpretable")
    model_key = MODEL_OPTIONS[model_label]

tab_manual, tab_csv = st.tabs(["✏️ Manual Entry (1 student)", "📂 CSV Upload (whole class)"])

with tab_manual:
    st.subheader("Enter student details")
    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics**")
            age      = st.selectbox("Age", [15, 16])
            gender   = st.selectbox("Gender", ["Female", "Male"])
            siblings = st.selectbox("No. of siblings", [0, 1, 2])
            direct   = st.selectbox("Direct admission", ["No", "Yes"])
        with c2:
            st.markdown("**School factors**")
            cca      = st.selectbox("CCA", ["Sports","Arts","Clubs","None"])
            learning = st.selectbox("Learning style", ["Visual","Auditory"])
            tuition  = st.selectbox("Private tuition", ["No","Yes"])
            transport= st.selectbox("Mode of transport", ["private transport","walk","public transport"])
        with c3:
            st.markdown("**Performance indicators**")
            hours    = st.slider("Study hours/week", 0, 30, 10)
            attend   = st.slider("Attendance rate (%)", 50.0, 100.0, 93.0, 0.5)
            n_male   = st.number_input("Male classmates",   0, 40, 14)
            n_female = st.number_input("Female classmates", 0, 40, 10)
        sc1, sc2 = st.columns(2)
        sleep_t = sc1.text_input("Bedtime (HH:MM)",   "22:00")
        wake_t  = sc2.text_input("Wake time (HH:MM)", "06:00")
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

    if submitted:
        row = pd.DataFrame([{"student_id":"Manual","number_of_siblings":siblings,
            "direct_admission":direct,"CCA":cca,"learning_style":learning,"gender":gender,
            "tuition":tuition,"n_male":n_male,"n_female":n_female,"age":age,
            "hours_per_week":hours,"attendance_rate":attend,
            "sleep_time":sleep_t,"wake_time":wake_t,"mode_of_transport":transport}])
        try:
            res   = predict(row, model_key)
            score = res["predicted_score"].iloc[0]
            band  = res["score_band"].iloc[0]
            st.markdown("---")
            m1,m2,m3 = st.columns(3)
            m1.metric("Predicted Score", f"{score:.1f} / 100")
            m2.metric("Grade Band", band)
            m3.metric("Model", model_label)
            if band == "Fail":
                st.error("⚠️ At risk (< 50). Consider intervention.")
            elif band == "C":
                st.warning("C band (50–59). Additional support may help.")
            else:
                st.success(f"Predicted **{band}** ({score:.1f}).")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab_csv:
    st.subheader("Upload class CSV")
    st.download_button("⬇️ Download CSV Template",
                       data=get_csv_template().to_csv(index=False).encode(),
                       file_name="class_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload your class CSV", type=["csv"])
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}"); st.stop()
        is_valid, errors = validate_csv(df_raw)
        if not is_valid:
            st.error("❌ Validation failed:")
            for err in errors: st.markdown(f"- {err}")
        else:
            st.success("✅ File validated.")
            with st.spinner("Running predictions…"):
                results = predict(df_raw, model_key)
            st.session_state["predictions_df"] = results
            st.session_state["model_key"]       = model_key
            n_fail = (results["score_band"]=="Fail").sum()
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Students",       len(results))
            m2.metric("Mean Score",     f"{results['predicted_score'].mean():.1f}")
            m3.metric("At-Risk (< 50)", n_fail,
                      delta=f"{n_fail/len(results)*100:.0f}%", delta_color="inverse")
            m4.metric("Model", model_label)
            disp = (["student_id"] if "student_id" in results.columns else []) +                    ["predicted_score","score_band"]
            st.dataframe(results[disp], use_container_width=True, height=400)
            st.download_button("⬇️ Download Predictions CSV",
                               data=results.to_csv(index=False).encode(),
                               file_name="predictions.csv", mime="text/csv")
            st.info("📊 Go to the **Dashboard** page for class-wide analysis.")
