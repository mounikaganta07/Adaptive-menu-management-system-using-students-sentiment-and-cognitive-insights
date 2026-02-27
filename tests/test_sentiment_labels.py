import pandas as pd
from src.sentiment import add_vader_sentiment

def test_sentiment_label_column_exists():
    df = pd.DataFrame({"t": ["i love it", "i hate it"]})
    out = add_vader_sentiment(df, "t", 0.05, -0.05)
    assert "sentiment_label" in out.columns
    assert set(out["sentiment_label"].unique()).issubset({"Positive","Neutral","Negative"})
