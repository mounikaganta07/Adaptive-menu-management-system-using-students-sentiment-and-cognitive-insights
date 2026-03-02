# Adaptive Hostel Menu Management using Student Sentiment's and Cognitive Insights

A data-driven system that analyzes student food feedback to help improve hostel menu planning.

The project processes feedback text, performs sentiment analysis, generates trend reports, and produces actionable insights for administrators.

## Features

- Sentiment analysis using VADER
- ML sentiment classifier (TF-IDF + Logistic Regression)
- Trend analysis by menu item, meal time, and month
- Menu Health Score with recommended actions (Keep / Improve / Replace)
- Complaint keyword extraction from negative feedback
- Interactive Streamlit dashboard
- FastAPI service for running the pipeline
- Automated CI checks and unit tests

## Tech Stack

Python, Pandas, Scikit-learn, NLTK (VADER), Streamlit, FastAPI, Pytest, GitHub Actions

## Project Structure

```
src/        Core pipeline modules
app/        Streamlit dashboard
api/        FastAPI service
data/       Input datasets
outputs/    Generated reports
tests/      Unit tests
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Pipeline

```bash
python -m src.cli --input data/raw/synthetic_feedback_data.csv --out outputs
```

## Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Model Evaluation

The ML classifier is trained on VADER pseudo-labels and evaluated on a human-labeled gold dataset.

Current results:

- Accuracy: 0.65  
- Precision (weighted): 0.67  
- Recall (weighted): 0.65  
- F1-score (weighted): 0.63

## Purpose

Designed as an internal analytics tool to help improve food quality, detect recurring complaints, and support better menu decisions.