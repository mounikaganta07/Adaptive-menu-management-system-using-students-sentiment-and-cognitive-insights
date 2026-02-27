import re
import pandas as pd
from collections import Counter
import nltk
import numpy as np



def _ensure_stopwords():
    try:
        nltk.data.find("corpora/stopwords.zip")
    except LookupError:
        nltk.download("stopwords")


def compute_menu_health(df: pd.DataFrame, negative_penalty: float, min_samples: int) -> pd.DataFrame:
    """
    Health score per menu_item:
      health_score = avg_sentiment - negative_penalty * negative_ratio

    Suggested action (only if count >= min_samples):
      - Replace: health_score < -0.20 OR negative_ratio > 0.50
      - Improve: health_score < 0.10 OR negative_ratio > 0.25
      - Keep: otherwise
      - Review: if count < min_samples
    """
    grp = df.groupby("menu_item").agg(
        count=("sentiment_score", "size"),
        avg_sentiment=("sentiment_score", "mean"),
        neg_count=("sentiment_label", lambda s: (s == "Negative").sum()),
        pos_count=("sentiment_label", lambda s: (s == "Positive").sum()),
        neu_count=("sentiment_label", lambda s: (s == "Neutral").sum()),
    ).reset_index()

    grp["negative_ratio"] = grp["neg_count"] / grp["count"]
    grp["health_score"] = grp["avg_sentiment"] - negative_penalty * grp["negative_ratio"]

    def action(row):
        if row["count"] < min_samples:
            return "Review"
        if (row["health_score"] < -0.20) or (row["negative_ratio"] > 0.50):
            return "Replace"
        if (row["health_score"] < 0.10) or (row["negative_ratio"] > 0.25):
            return "Improve"
        return "Keep"

    grp["action"] = grp.apply(action, axis=1)

    # Sort so admin can see worst items first within each action bucket
    return grp.sort_values(["action", "health_score"], ascending=[True, True])

def top_complaint_keywords(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Production-style complaint extraction:
    - Rule-based tags only (no TF-IDF fallback).
    - Adds CATEGORY to make dashboard actionable.
    Output columns: menu_item, category, keyword, count
    """

    neg = df[df["sentiment_label"] == "Negative"].copy()
    if neg.empty:
        return pd.DataFrame(columns=["menu_item", "category", "keyword", "count"])

    # category, regex pattern, normalized keyword
    RULES: list[tuple[str, re.Pattern, str]] = [
        # Missing
        ("Missing", re.compile(r"\bmissing\b.*\bingredient(s)?\b", re.I), "ingredient missing"),
        ("Missing", re.compile(r"\bmissing\b", re.I), "missing item"),

        # Temperature
        ("Temperature", re.compile(r"\btoo\s+cold\b|\bserved\s+too\s+cold\b", re.I), "too cold"),
        ("Temperature", re.compile(r"\btoo\s+hot\b|\bserved\s+too\s+hot\b", re.I), "too hot"),
        ("Temperature", re.compile(r"\btemperature\b", re.I), "temperature issue"),

        # Taste / Seasoning
        ("Taste", re.compile(r"\btoo\s+salty\b", re.I), "too salty"),
        ("Taste", re.compile(r"\bnot\s+salty\s+enough\b|\bneeds?\s+more\s+salt\b", re.I), "needs more salt"),
        ("Taste", re.compile(r"\btoo\s+spicy\b", re.I), "too spicy"),
        ("Taste", re.compile(r"\bbland\b|\bneeds?\s+more\s+seasoning\b|\bcould\s+use\s+more\s+seasoning\b", re.I), "needs seasoning"),
        ("Taste", re.compile(r"\btasted?\s+strange\b|\bweird\b.*\bflavor\b|\bbad\s+taste\b", re.I), "flavor issue"),

        # Oil / Greasy
        ("Oil", re.compile(r"\btoo\s+oily\b|\bgreasy\b", re.I), "too oily/greasy"),

        # Cooking / Doneness
        ("Cooking", re.compile(r"\bundercooked\b|\bnot\s+cooked\s+properly\b|\braw\b", re.I), "undercooked"),
        ("Cooking", re.compile(r"\bovercooked\b|\bburnt\b", re.I), "overcooked/burnt"),

        # Texture
        ("Texture", re.compile(r"\btexture\b", re.I), "texture issue"),
        ("Texture", re.compile(r"\btoo\s+hard\b|\bhard\b.*\broti\b", re.I), "too hard"),
        ("Texture", re.compile(r"\bwatery\b", re.I), "too watery"),
        ("Texture", re.compile(r"\btoo\s+dry\b", re.I), "too dry"),

        # Quality / Freshness
        ("Quality", re.compile(r"\bstale\b|\bsmell(ed)?\s+bad\b|\bbad\s+smell\b", re.I), "stale/smell issue"),
        ("Quality", re.compile(r"\binconsistent\b", re.I), "inconsistent quality"),
        ("Quality", re.compile(r"\bpoor\b.*\bquality\b|\bquality\b.*\bpoor\b|\bquality\b", re.I), "quality issue"),

        # Portion / Quantity
        ("Portion", re.compile(r"\bportion\b.*\bsmall\b|\btoo\s+small\s+portion\b", re.I), "portion too small"),
        ("Portion", re.compile(r"\bquantity\b.*\bnot\s+enough\b|\bserving\b.*\bnot\s+sufficient\b|\bnot\s+enough\s+quantity\b", re.I), "quantity not enough"),

        # Service / Timing
        ("Service", re.compile(r"\bdelay(ed)?\b|\blate\b|\bserved\s+very\s+late\b", re.I), "service delayed"),
        ("Service", re.compile(r"\bservice\b", re.I), "service issue"),
    ]

    def extract(text: str) -> list[tuple[str, str]]:
        t = str(text or "").lower().strip()
        if not t:
            return []
        out: list[tuple[str, str]] = []
        for cat, pat, label in RULES:
            if pat.search(t):
                out.append((cat, label))
        # dedupe
        out = list(dict.fromkeys(out))
        return out

    rows = []
    for item, sub in neg.groupby("menu_item"):
        c = Counter()
        for text in sub["cleaned_feedback"].astype(str):
            c.update(extract(text))

        if not c:
            c.update([("Other", "general complaint")])

        for (cat, kw), cnt in c.most_common(top_k):
            rows.append({"menu_item": item, "category": cat, "keyword": kw, "count": cnt})

    return pd.DataFrame(rows).sort_values(["menu_item", "count"], ascending=[True, False])