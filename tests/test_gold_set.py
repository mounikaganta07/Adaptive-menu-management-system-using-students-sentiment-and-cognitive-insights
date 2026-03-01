import pandas as pd

VALID = {"Negative", "Neutral", "Positive"}

def test_gold_set_schema():
    df = pd.read_csv("data/raw/gold_test.csv", engine="python", on_bad_lines="skip")
    assert {"feedback_text", "sentiment_label"}.issubset(df.columns)
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip()
    assert set(df["sentiment_label"].unique()).issubset(VALID)
    assert len(df) >= 10