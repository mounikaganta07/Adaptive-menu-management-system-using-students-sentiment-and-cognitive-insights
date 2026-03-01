import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class MLResults:
    model: Any
    metrics: Dict[str, Any]
    confusion: Any


def train_tfidf_logreg(
    df: pd.DataFrame,
    text_col: str = "cleaned_feedback",
    label_col: str = "sentiment_label",
    group_col: str = "menu_item",
    test_size: float = 0.2,
    random_state: int = 42,
    gold_path: str = "data/raw/gold_test.csv",
    gold_train_path: str = "data/raw/gold_train.csv",
) -> MLResults:
    """
    Train TF-IDF + Logistic Regression on pseudo-labeled data (VADER labels),
    but evaluate on a HUMAN-LABELED gold test set if available.

    If gold_test.csv doesn't exist, fall back to a GROUP split by menu_item
    (better than random split for templated synthetic data).
    """

    # -------------------------
    # 1) TRAIN DATA (pseudo labels)
    # -------------------------
    train_data = df[[text_col, label_col, group_col, "meal_time"]].dropna()
    X_all = (
    train_data[group_col].astype(str) + " " +
    train_data["meal_time"].astype(str) + " " +
    train_data[text_col].astype(str)
)
    y_all = train_data[label_col].astype(str)
    groups = train_data[group_col].astype(str)

    # ✅ Add human-labeled gold train samples (no leakage)
    if os.path.exists(gold_train_path):
        gt = pd.read_csv(gold_train_path, engine="python", on_bad_lines="skip")
    if {"feedback_text", "sentiment_label"}.issubset(gt.columns):
        gt = gt.dropna(subset=["feedback_text", "sentiment_label"])
        gt["sentiment_label"] = gt["sentiment_label"].astype(str).str.strip()
        # Upsample gold train so it has meaningful influence
        gt = pd.concat([gt] * 10, ignore_index=True)   # repeat 10x

        # Build gold train text same way (menu + meal_time not available → use text only)
        gt_text = gt["feedback_text"].astype(str)

        X_all = pd.concat([X_all, gt_text], ignore_index=True)
        y_all = pd.concat([y_all, gt["sentiment_label"].astype(str)], ignore_index=True)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    # -------------------------
    # 2) TEST DATA (gold if exists else group split)
    # -------------------------
    used_gold = False
    X_test: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None

    if os.path.exists(gold_path):
        gold_df = pd.read_csv(gold_path, engine="python", on_bad_lines="skip")
        if {"feedback_text", "sentiment_label"}.issubset(gold_df.columns):
            # Train on all pseudo-labeled data
            pipeline.fit(X_all, y_all)

            # Evaluate on human labeled gold data
            X_test = gold_df["feedback_text"].astype(str)
            y_test = gold_df["sentiment_label"].astype(str)

            used_gold = True

    if not used_gold:
        # Fallback: GROUP split (menu_item)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))

        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

        pipeline.fit(X_train, y_train)

    # -------------------------
    # 3) EVALUATE
    # -------------------------
    y_pred = pipeline.predict(X_test)

    labels_order: List[str] = ["Negative", "Neutral", "Positive"]

    split_info = {
        "evaluation_set": "gold_test.csv" if used_gold else "group_shuffle_split",
        "gold_path": gold_path if used_gold else None,
    }

    if not used_gold:
        split_info.update({
            "group_col": group_col,
            "train_groups": int(groups.iloc[train_idx].nunique()),
            "test_groups": int(groups.iloc[test_idx].nunique()),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        })
    else:
        split_info.update({
            "train_size": int(len(X_all)),
            "test_size": int(len(X_test)),
        })

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "labels_order": labels_order,
        "gold_train_upsample": 10,
        **split_info
    }

    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    return MLResults(model=pipeline, metrics=metrics, confusion=cm)