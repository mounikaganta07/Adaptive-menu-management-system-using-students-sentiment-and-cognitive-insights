import pandas as pd
from src.trends import menu_item_trend

def test_menu_item_trend_shape():
    df = pd.DataFrame({
        "menu_item": ["A","A","B"],
        "sentiment_label": ["Positive","Negative","Positive"]
    })
    out = menu_item_trend(df)
    assert "menu_item" in out.columns
