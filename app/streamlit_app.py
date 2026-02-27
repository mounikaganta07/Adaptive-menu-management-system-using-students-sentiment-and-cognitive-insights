import os
import sys

# Make sure VS Code/Streamlit can find the src/ package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import Config
from src.pipeline import run_pipeline

st.set_page_config(page_title="Menu Management System", layout="wide")
st.title("🍽️ Hostel Menu Management Intelligence System")

cfg = Config()

st.caption("Input CSV: data/raw/synthetic_feedback_data.csv (change in src/config.py if needed)")

col_run, col_paths = st.columns([1, 2])
with col_run:
    if st.button("Run Pipeline"):
        try:
            paths = run_pipeline(cfg)
            st.session_state["paths"] = paths
            st.success("Pipeline completed!")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            raise

with col_paths:
    if "paths" in st.session_state:
        st.json(st.session_state["paths"])

# Load processed dataset
if not os.path.exists(cfg.processed_data_path):
    st.info("Run Pipeline to generate processed data.")
    st.stop()

df = pd.read_csv(cfg.processed_data_path)

# Filters
st.subheader("Filters")
c1, c2, c3 = st.columns(3)
menu_items = ["All"] + sorted(df["menu_item"].dropna().unique().tolist())
meal_times = ["All"] + sorted(df["meal_time"].dropna().unique().tolist())
sentiments = ["All", "Positive", "Neutral", "Negative"]

with c1:
    menu_choice = st.selectbox("Menu Item", menu_items)
with c2:
    meal_choice = st.selectbox("Meal Time", meal_times)
with c3:
    sentiment_choice = st.selectbox("Sentiment", sentiments)

f = df.copy()
if menu_choice != "All":
    f = f[f["menu_item"] == menu_choice]
if meal_choice != "All":
    f = f[f["meal_time"] == meal_choice]
if sentiment_choice != "All":
    f = f[f["sentiment_label"] == sentiment_choice]

# ✅ Overview (B) — always show Positive/Neutral/Negative percentages
st.subheader("Overview")
m1, m2, m3, m4, m5 = st.columns(5)

total = len(f)
m1.metric("Feedback Count", total)

avg_score = float(f["sentiment_score"].mean()) if total else 0.0
m2.metric("Avg Sentiment Score", round(avg_score, 3))

pos_pct = float((f["sentiment_label"] == "Positive").mean() * 100) if total else 0.0
neu_pct = float((f["sentiment_label"] == "Neutral").mean() * 100) if total else 0.0
neg_pct = float((f["sentiment_label"] == "Negative").mean() * 100) if total else 0.0

m3.metric("Positive %", round(pos_pct, 2))
m4.metric("Neutral %", round(neu_pct, 2))
m5.metric("Negative %", round(neg_pct, 2))

st.subheader("Sentiment Distribution")
if len(f) == 0:
    st.info("No rows match current filters.")
else:
    dist = f["sentiment_label"].value_counts().reset_index()
    dist.columns = ["sentiment_label", "count"]
    st.plotly_chart(px.bar(dist, x="sentiment_label", y="count"), use_container_width=True)
st.subheader("Monthly Trend")
if len(f) == 0:
    st.info("No rows match current filters.")
else:
    trend_df = f.copy()
    trend_df["feedback_timestamp"] = pd.to_datetime(trend_df["feedback_timestamp"], errors="coerce")
    trend_df = trend_df.dropna(subset=["feedback_timestamp"])

    if trend_df.empty:
        st.info("No valid timestamps to plot after parsing.")
    else:
        trend_df["year_month"] = trend_df["feedback_timestamp"].dt.to_period("M").astype(str)
        trend = trend_df.groupby(["year_month", "sentiment_label"]).size().reset_index(name="count")
        st.plotly_chart(
            px.line(trend, x="year_month", y="count", color="sentiment_label", markers=True),
            use_container_width=True
        )

st.subheader("🧠 Admin Actions (Menu Health)")
health_path = os.path.join(cfg.output_dir, "menu_health.csv")
if os.path.exists(health_path):
    health = pd.read_csv(health_path)
    st.dataframe(health, use_container_width=True)
else:
    st.info("Run Pipeline to generate menu health table.")

st.subheader("🧾 Top Complaint Keywords (Negative feedback)")
kw_path = os.path.join(cfg.output_dir, "complaint_keywords.csv")
if os.path.exists(kw_path):
    kws = pd.read_csv(kw_path)

    # If your extractor adds 'category', sort nicely
    if "category" in kws.columns:
        kws = kws.sort_values(["menu_item", "category", "count"], ascending=[True, True, False])
    else:
        kws = kws.sort_values(["menu_item", "count"], ascending=[True, False])

    st.dataframe(kws, use_container_width=True)
else:
    st.info("Run Pipeline to generate complaint keywords.")

st.subheader("✅ ML Model Evaluation (TF-IDF + Logistic Regression)")
metrics_file = os.path.join(cfg.output_dir, "ml_metrics.json")
cm_file = os.path.join(cfg.output_dir, "ml_confusion_matrix.csv")

if os.path.exists(metrics_file) and os.path.exists(cm_file):
    with open(metrics_file, "r", encoding="utf-8") as fmetrics:
        metrics = json.load(fmetrics)

    cm = np.loadtxt(cm_file, delimiter=",")
    st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")

    labels = metrics["labels_order"]
    st.dataframe(
        pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels],
            columns=[f"pred_{l}" for l in labels]
        ),
        use_container_width=True
    )

    rep = metrics["classification_report"]
    st.json({
        "Negative_f1": rep["Negative"]["f1-score"],
        "Neutral_f1": rep["Neutral"]["f1-score"],
        "Positive_f1": rep["Positive"]["f1-score"],
        "macro_avg_f1": rep["macro avg"]["f1-score"],
        "weighted_avg_f1": rep["weighted avg"]["f1-score"],
    })
else:
    st.info("Run Pipeline to generate ML metrics.")

st.subheader("Sample Feedback")
if len(f) == 0:
    st.info("No rows match current filters.")
else:
    st.dataframe(
        f[["menu_item", "meal_time", "feedback_text", "sentiment_label", "sentiment_score"]].head(30),
        use_container_width=True
    )
    