from src.preprocessing import clean_text

def test_clean_text_lowercase_and_strip():
    assert clean_text("  HELLO  ") == "hello"

def test_clean_text_removes_url():
    t = clean_text("check this http://example.com now")
    assert "http" not in t
