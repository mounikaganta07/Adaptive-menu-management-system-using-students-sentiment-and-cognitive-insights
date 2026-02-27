# Hostel Menu Management Intelligence System

An internal-style analytics + automation tool for hostel/cafeteria menu management.

## What it does (implemented)
- Cleans feedback text and scores sentiment with VADER (Positive/Neutral/Negative)
- Trains an ML sentiment classifier (TF-IDF + Logistic Regression) using VADER pseudo-labels
- Evaluates ML model (accuracy, precision/recall/F1, confusion matrix)
- Generates trend reports (menu item, meal time, monthly, monthly per item)
- Computes a **Menu Health Score** per item and produces **admin actions** (Keep / Improve / Replace)
- Extracts **top complaint keywords** from negative feedback per menu item
- Streamlit dashboard for filtering + charts + admin action table
- FastAPI service with `/health` and `/run` endpoints (internal platform style)
- Unit tests + CI workflow

## Project structure
- `data/raw/` input CSV (required: feedback_text, feedback_timestamp, menu_item, meal_time)
- `data/processed/` generated processed dataset
- `outputs/` generated reports + metrics
- `src/` pipeline modules + CLI
- `app/` Streamlit dashboard
- `api/` FastAPI service
- `.github/workflows/` CI

## Quick start (VS Code)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# 1) Run dashboard
streamlit run app/streamlit_app.py

# 2) Or run pipeline via CLI
python -m src.cli --input data/raw/synthetic_feedback_data.csv --out outputs
```

## Run API (optional)
```bash
uvicorn api.main:app --reload
# open http://127.0.0.1:8000/docs
```
