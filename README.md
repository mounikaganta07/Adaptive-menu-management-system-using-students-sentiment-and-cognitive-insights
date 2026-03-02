# Adaptive Hostel Menu Management using Student Sentiments and Cognitive Insights

An internal-style analytics and automation platform for hostel menu decision support.

## What it does (implemented)

- Cleans feedback text and scores sentiment using VADER (Positive / Neutral / Negative)
- Trains an ML sentiment classifier (TF-IDF + Logistic Regression) using VADER pseudo-labels
- Evaluates ML model with accuracy, precision, recall, F1-score, and confusion matrix
- Generates trend reports (menu item, meal time, monthly, and monthly per item analysis)
- Computes a Menu Health Score per item and produces admin actions (Keep / Improve / Replace)
- Extracts categorized complaint keywords from negative feedback (temperature, taste, quality, service, etc.)
- Provides an interactive Streamlit dashboard for filtering, charts, and admin insights
- Exposes a FastAPI service with `/health` and `/run` endpoints (internal platform style)
- Includes unit tests (pytest) and GitHub Actions CI workflow

## Project Structure

- `data/raw/` input CSV (required columns: feedback_text, feedback_timestamp, menu_item, meal_time)
- `data/processed/` generated dataset with sentiment labels
- `outputs/` generated reports and ML metrics
- `src/` modular pipeline (preprocessing, sentiment, ML, trends, intelligence, CLI)
- `app/` Streamlit dashboard
- `api/` FastAPI service
- `tests/` unit tests
- `.github/workflows/` CI configuration

## Required Input Columns

CSV must contain:

- `feedback_text`
- `feedback_timestamp`
- `menu_item`
- `meal_time`

## Quick Start (VS Code)

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Run Pipeline via CLI

```bash
python -m src.cli --input data/raw/synthetic_feedback_data.csv --out outputs
```

## Run API (Optional)

```bash
uvicorn api.main:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

## Design Notes

- VADER is used for lightweight sentiment scoring without requiring labeled data.
- TF-IDF + Logistic Regression provides a supervised validation baseline.
- Complaint extraction is rule-based to ensure actionable business categories rather than noisy statistical n-grams.
- The system is designed as an internal analytics platform, focusing on reliability, modularity, and automation.

## ML Evaluation Strategy

The model is trained on synthetic feedback labeled using VADER.
To avoid inflated accuracy, final evaluation is performed on a
human-labeled gold test set (`data/raw/gold_test.csv`).

Since gold labels are higher quality but limited in size,
gold training samples are upsampled during training (10x)
to increase influence while keeping the gold test set untouched.

This ensures honest and realistic evaluation.
## Results (Gold Test Set)
- Accuracy: 0.65
- Precision (weighted): 0.6733
- Recall (weighted): 0.65
- F1 (weighted): 0.6315

> Note: Training labels are generated using VADER (weak supervision). Metrics are reported on a small gold-labeled test set.

