import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from app.utils.loader import BAND_COLORS, BAND_ORDER


# ── Score distribution ────────────────────────────────────────────────────────
def score_histogram(df, col="predicted_score",
                    title="Predicted Score Distribution", cohort_mean=67.18):
    fig = px.histogram(df, x=col, nbins=20, title=title,
                       color_discrete_sequence=["#1f77b4"])
    fig.add_vline(x=df[col].mean(), line_dash="dash", line_color="#d62728",
                  annotation_text=f"Class mean: {df[col].mean():.1f}",
                  annotation_position="top right")
    fig.add_vline(x=cohort_mean, line_dash="dot", line_color="#7f7f7f",
                  annotation_text=f"Cohort mean: {cohort_mean:.1f}",
                  annotation_position="top left")
    fig.update_xaxes(title_text="Predicted Score", range=[0, 105])
    fig.update_yaxes(title_text="No. of Students")
    return fig


# ── Grade band breakdown ──────────────────────────────────────────────────────
def score_band_bar(df):
    counts = (df["score_band"].value_counts()
                .reindex(BAND_ORDER, fill_value=0).reset_index())
    counts.columns = ["Band", "Count"]
    pct = (counts["Count"] / len(df) * 100).round(1)
    fig = go.Figure(go.Bar(
        x=counts["Count"], y=counts["Band"], orientation="h",
        marker_color=[BAND_COLORS[b] for b in counts["Band"]],
        text=[f"{c} ({p}%)" if c > 0 else "" for c, p in zip(counts["Count"], pct)],
        textposition="outside",
    ))
    fig.update_xaxes(title_text="No. of Students")
    fig.update_yaxes(title_text="Grade Band", categoryorder="array",
                     categoryarray=BAND_ORDER)
    fig.update_layout(title="Predicted Grade Band Breakdown", showlegend=False)
    return fig


# ── Feature snapshot ──────────────────────────────────────────────────────────
_COHORT = {"hours_per_week": 10.31, "attendance_rate": 93.27, "tuition_pct": 56.70}

def feature_snapshot(df):
    metrics, norms = {}, {}
    if "hours_per_week"  in df.columns:
        _h = pd.to_numeric(df["hours_per_week"], errors="coerce")
        if _h.notna().any():
            metrics["Avg hrs/week"] = _h.mean()
            norms["Avg hrs/week"]   = _COHORT["hours_per_week"]
    if "attendance_rate" in df.columns:
        _a = pd.to_numeric(df["attendance_rate"], errors="coerce")
        if _a.notna().any():
            metrics["Avg attendance %"] = _a.mean()
            norms["Avg attendance %"]   = _COHORT["attendance_rate"]
    if "tuition"         in df.columns:
        _tuition_num = pd.to_numeric(
            df["tuition"].map({"Yes": 1, "Y": 1, "No": 0, "N": 0}),
            errors="coerce",
        )
        if _tuition_num.notna().any():
            metrics["% with tuition"] = _tuition_num.mean() * 100
            norms["% with tuition"]   = _COHORT["tuition_pct"]
    labels = list(metrics)
    fig = go.Figure()
    fig.add_bar(name="This Class",     x=labels, y=list(metrics.values()),
                marker_color="#1f77b4",
                text=[f"{v:.1f}" for v in metrics.values()], textposition="outside")
    fig.add_bar(name="Cohort Average", x=labels, y=[norms[l] for l in labels],
                marker_color="#aec7e8",
                text=[f"{v:.1f}" for v in norms.values()], textposition="outside")
    fig.update_layout(
        title="Class Feature Snapshot vs. Cohort Average", barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="Value")
    return fig


# ── What-If comparison ────────────────────────────────────────────────────────
def whatif_comparison(baseline: float, updated: float):
    delta = updated - baseline
    arrow = "▲" if delta >= 0 else "▼"
    color = "#2ca02c" if delta >= 0 else "#d62728"
    fig = go.Figure(go.Bar(
        x=["Baseline", "After Intervention"],
        y=[baseline, updated],
        marker_color=["#aec7e8", color],
        text=[f"{baseline:.1f}", f"{updated:.1f} ({arrow}{abs(delta):.1f})"],
        textposition="outside",
    ))
    fig.update_yaxes(title_text="Predicted Score", range=[0, 105])
    fig.update_layout(
        title=f"What-If: Score Change {arrow} {abs(delta):.1f} pts",
        showlegend=False,
    )
    return fig


# ── Feature importance ────────────────────────────────────────────────────────
def feature_importance_bar(importance_df, title="Feature Importances"):
    df = importance_df.sort_values("importance", ascending=True).tail(15)
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in df["importance"]]
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"],
        orientation="h", marker_color=colors,
    ))
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    fig.update_xaxes(title_text="Importance / Coefficient")
    fig.update_yaxes(title_text="Feature")
    fig.update_layout(title=title, showlegend=False)
    return fig
