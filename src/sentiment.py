import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def _ensure_vader() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

def add_vader_sentiment(df: pd.DataFrame, text_col: str, pos_th: float, neg_th: float) -> pd.DataFrame:
    """Adds sentiment_score (compound) + sentiment_label."""
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()

    df = df.copy()
    df["sentiment_score"] = df[text_col].fillna("").astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])

    def label(s: float) -> str:
        if s >= pos_th:
            return "Positive"
        if s <= neg_th:
            return "Negative"
        return "Neutral"

    df["sentiment_label"] = df["sentiment_score"].apply(label)
    return df
