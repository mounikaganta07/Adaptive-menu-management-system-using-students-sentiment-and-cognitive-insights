import pandas as pd

def menu_item_trend(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["menu_item", "sentiment_label"]).size().unstack(fill_value=0).reset_index()

def meal_time_trend(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["meal_time", "sentiment_label"]).size().unstack(fill_value=0).reset_index()

def monthly_sentiment_trend(df: pd.DataFrame, timestamp_col: str = "feedback_timestamp") -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["year_month"] = df[timestamp_col].dt.to_period("M").astype(str)
    return df.groupby(["year_month", "sentiment_label"]).size().unstack(fill_value=0).reset_index().sort_values("year_month")

def monthly_menu_analysis(df: pd.DataFrame, timestamp_col: str = "feedback_timestamp") -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["year_month"] = df[timestamp_col].dt.to_period("M").astype(str)

    counts = df.groupby(["year_month", "menu_item", "sentiment_label"]).size().unstack(fill_value=0).reset_index()
    avg = df.groupby(["year_month", "menu_item"])["sentiment_score"].mean().reset_index().rename(columns={"sentiment_score": "average_sentiment_score"})
    out = counts.merge(avg, on=["year_month", "menu_item"], how="left")
    return out.sort_values(["year_month", "menu_item"])
