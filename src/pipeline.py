import os
import time
import logging
import pandas as pd

from src.config import Config
from src.preprocessing import clean_text
from src.sentiment import add_vader_sentiment
from src.trends import menu_item_trend, meal_time_trend, monthly_sentiment_trend, monthly_menu_analysis
from src.ml_train import train_tfidf_logreg
from src.menu_intelligence import compute_menu_health, top_complaint_keywords
from src.reports import save_df, save_ml_metrics
from src.logging_utils import setup_logging

REQUIRED_COLS = ["feedback_text", "feedback_timestamp", "menu_item", "meal_time"]
log = logging.getLogger("pipeline")

def run_pipeline(cfg: Config = Config()) -> dict:
    setup_logging()
    start = time.time()

    df = pd.read_csv(cfg.raw_data_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["cleaned_feedback"] = df["feedback_text"].apply(clean_text)
    df = add_vader_sentiment(df, "cleaned_feedback", cfg.pos_threshold, cfg.neg_threshold)

    os.makedirs(os.path.dirname(cfg.processed_data_path), exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    df.to_csv(cfg.processed_data_path, index=False)

    # Trend reports
    paths = {}
    paths["processed_data"] = cfg.processed_data_path
    paths["menu_trend"] = save_df(menu_item_trend(df), cfg.output_dir, "menu_trend_analysis.csv")
    paths["meal_time_trend"] = save_df(meal_time_trend(df), cfg.output_dir, "meal_time_trend_analysis.csv")
    paths["monthly_trend"] = save_df(monthly_sentiment_trend(df), cfg.output_dir, "monthly_sentiment_trend.csv")
    paths["monthly_menu_analysis"] = save_df(monthly_menu_analysis(df), cfg.output_dir, "monthly_menu_analysis.csv")

    # Menu intelligence
    health = compute_menu_health(df, cfg.negative_penalty, cfg.min_samples_per_item)
    paths["menu_health"] = save_df(health, cfg.output_dir, "menu_health.csv")
    keywords = top_complaint_keywords(df, top_k=10)
    paths["complaint_keywords"] = save_df(keywords, cfg.output_dir, "complaint_keywords.csv")

    # ML model (pseudo-labels)
    ml = train_tfidf_logreg(df, text_col="cleaned_feedback", label_col="sentiment_label")
    paths.update(save_ml_metrics(cfg.output_dir, ml.metrics, ml.confusion))

    log.info("Pipeline finished in %.2fs", time.time() - start)
    return paths
