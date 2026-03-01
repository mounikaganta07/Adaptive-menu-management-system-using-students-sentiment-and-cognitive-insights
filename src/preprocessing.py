import re
import pandas as pd

def clean_text(text: str) -> str:
    """Basic cleaning suitable for short hostel feedback."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    t = text.lower()
    if " but " in f" {t} " or " however " in f" {t} " or " though " in f" {t} ":
        text = text + " __CONTRAST__"
    return text
