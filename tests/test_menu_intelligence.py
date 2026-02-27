import pandas as pd
from src.menu_intelligence import compute_menu_health

def test_compute_menu_health_returns_action():
    df = pd.DataFrame({
        "menu_item": ["A"]*30,
        "sentiment_score": [-0.5]*20 + [0.2]*10,
        "sentiment_label": ["Negative"]*20 + ["Positive"]*10
    })
    out = compute_menu_health(df, negative_penalty=0.7, min_samples=20)
    assert "action" in out.columns
