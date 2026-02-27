import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split
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
    test_size: float = 0.2,
    random_state: int = 42,
) -> MLResults:
    data = df[[text_col, label_col]].dropna()
    X = data[text_col].astype(str)
    y = data[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels_order: List[str] = ["Negative", "Neutral", "Positive"]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "labels_order": labels_order,
    }
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    return MLResults(model=pipeline, metrics=metrics, confusion=cm)
