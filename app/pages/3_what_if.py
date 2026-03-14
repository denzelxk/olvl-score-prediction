import sys, os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st
from app.utils.charts import whatif_comparison
from app.utils.loader import DEFAULT_MODEL, MODEL_OPTIONS, load_model, score_to_band

st.set_page_config(page_title="What-If — UA Secondary", page_icon="🔍", layout="wide")
st.title("🔍 What-If Intervention Simulator")
st.caption("Adjust sliders to simulate interventions and see the predicted score change.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    model_label = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(
            next(k for k, v in MODEL_OPTIONS.items() if v == DEFAULT_MODEL)
        ),
    )
    model_key = MODEL_OPTIONS[model_label]

    has_class = "predictions_df" in st.session_state
    if has_class:
        use_uploaded = st.checkbox("Use uploaded class", value=True)
    else:
        use_uploaded = False

# ── Gate: no data, no checkbox ────────────────────────────────────────────────
if not has_class:
    st.info(
        "No class data loaded. You can still explore with **default values** below, "
        "or go to Predict first to load a real student.",
        icon="💡",
    )
    st.page_link("pages/1_predict.py", label="Go to Predict →", icon="🔮")
    st.markdown("---")

# ── Defaults ──────────────────────────────────────────────────────────────────
def _defaults():
    return dict(
        age=15.0, hours_per_week=10.0, attendance_rate=93.0,
        number_of_siblings=1.0, sleep_duration=8.0, class_size=22.0,
        male_ratio=0.5, direct_admission="No", CCA="Sports",
        learning_style="Visual", gender="Female",
        mode_of_transport="public transport", tuition=0,
    )

# ── Resolve base values ───────────────────────────────────────────────────────
if use_uploaded:
    df_pred = st.session_state["predictions_df"]
    id_col  = "student_id" if "student_id" in df_pred.columns else None
    options = df_pred[id_col].tolist() if id_col else [f"Student {i+1}" for i in range(len(df_pred))]
    sel     = st.selectbox("Select student", options)
    idx     = options.index(sel)
    row     = df_pred.iloc[idx]

    _TUITION_MAP = {"Yes": 1, "Y": 1, "No": 0, "N": 0}

    def g(c, d):
        if c not in row.index:
            return d
        val = row[c]
        if c == "tuition":
            return _TUITION_MAP.get(str(val), int(float(val)) if str(val).lstrip("-").isdigit() else d)
        try:
            return float(val)
        except (ValueError, TypeError):
            return d

    base = dict(
        age=g("age", 15), hours_per_week=g("hours_per_week", 10),
        attendance_rate=g("attendance_rate", 93),
        number_of_siblings=g("number_of_siblings", 1),
        sleep_duration=g("sleep_duration", 8),
        class_size=g("class_size", 22), male_ratio=g("male_ratio", 0.5),
        direct_admission=row.get("direct_admission", "No"),
        CCA=str(row.get("CCA", "Sports")).capitalize(),
        learning_style=row.get("learning_style", "Visual"),
        gender=row.get("gender", "Female"),
        mode_of_transport=row.get("mode_of_transport", "public transport"),
        tuition=int(g("tuition", 0)),
    )
    baseline_score = float(row.get("predicted_score", 0.0))
else:
    base = _defaults()
    baseline_score = None

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown("---")
ci, cf = st.columns([2, 1])
with ci:
    st.markdown("**🎛️ Intervention Features**")
    new_tval   = 1 if "Yes" in st.selectbox("Tuition", ["No (0)", "Yes (1)"], index=base["tuition"]) else 0
    new_hours  = st.slider("Study hrs/week",  0.0,  30.0, float(base["hours_per_week"]),  0.5)
    new_attend = st.slider("Attendance (%)", 50.0, 100.0, float(base["attendance_rate"]), 0.5)
    new_sleep  = st.slider("Sleep (hrs)",     4.0,  10.0, float(base["sleep_duration"]),  0.5)
    cca_opts   = ["Sports", "Arts", "Clubs", "None"]
    new_cca    = st.selectbox("CCA", cca_opts,
                              index=cca_opts.index(base["CCA"]) if base["CCA"] in cca_opts else 0)
with cf:
    st.markdown("**🔒 Fixed (demographic)**")
    for lbl, k in [("Age", "age"), ("Gender", "gender"), ("Siblings", "number_of_siblings"),
                   ("Direct admission", "direct_admission"), ("Class size", "class_size"),
                   ("Male ratio", "male_ratio"), ("Learning style", "learning_style"),
                   ("Transport", "mode_of_transport")]:
        v = base[k]
        if isinstance(v, float) and v != int(v):
            v_str = f"{v:.2f}"
        elif isinstance(v, (float, int)):
            v_str = str(int(v))
        else:
            v_str = str(v)
        st.write(f"{lbl}: **{v_str}**")

# ── Prediction ────────────────────────────────────────────────────────────────
def _df(f):
    return pd.DataFrame([{
        "age": f["age"], "hours_per_week": f["hours_per_week"],
        "attendance_rate": f["attendance_rate"], "number_of_siblings": f["number_of_siblings"],
        "sleep_duration": f["sleep_duration"], "class_size": f["class_size"],
        "male_ratio": f["male_ratio"], "direct_admission": f["direct_admission"],
        "CCA": str(f["CCA"]).capitalize(), "learning_style": f["learning_style"],
        "gender": f["gender"], "mode_of_transport": f["mode_of_transport"],
        "tuition": int(f["tuition"]),
    }])

try:
    pipeline = load_model(model_key)
    if baseline_score is None:
        baseline_score = float(pipeline.predict(_df(base))[0])
    updated = {**base, "tuition": new_tval, "hours_per_week": new_hours,
               "attendance_rate": new_attend, "sleep_duration": new_sleep, "CCA": new_cca}
    updated_score = float(pipeline.predict(_df(updated))[0])

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Baseline",           f"{baseline_score:.1f}")
    r2.metric("After Intervention", f"{updated_score:.1f}",
              delta=f"{updated_score - baseline_score:+.1f} pts")
    r3.metric("Band", f"{score_to_band(baseline_score)} → {score_to_band(updated_score)}")
    st.plotly_chart(whatif_comparison(baseline_score, updated_score), use_container_width=True)

    changes = []
    if new_tval != base["tuition"]:
        changes.append(f"Tuition: {'added ✅' if new_tval else 'removed'}")
    if abs(new_hours  - base["hours_per_week"])  > 0.1:
        changes.append(f"Hours: {base['hours_per_week']:.0f}→{new_hours:.0f}")
    if abs(new_attend - base["attendance_rate"]) > 0.1:
        changes.append(f"Attendance: {base['attendance_rate']:.1f}→{new_attend:.1f}%")
    if new_cca != base["CCA"]:
        changes.append(f"CCA: {base['CCA']}→{new_cca}")
    if changes:
        st.info("**Applied:** " + " | ".join(changes))

except FileNotFoundError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Error: {e}")
