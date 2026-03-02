import os
import json
import numpy as np
import pandas as pd

def save_df(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path

def save_ml_metrics(output_dir: str, metrics: dict, confusion: np.ndarray) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # --- Extract weighted averages from classification_report if available ---
    try:
        cr = metrics.get("classification_report")

        # case 1: sklearn report saved as dict
        if isinstance(cr, dict):
            wa = cr.get("weighted avg") or cr.get("weighted_avg") or {}
            metrics["precision_weighted"] = wa.get("precision")
            metrics["recall_weighted"] = wa.get("recall")
            metrics["f1_weighted"] = wa.get("f1-score") or wa.get("f1")

        # case 2: report saved as string
        elif isinstance(cr, str):
            # simple fallback: leave None but keep keys present
            metrics["precision_weighted"] = None
            metrics["recall_weighted"] = None
            metrics["f1_weighted"] = None

    except Exception:
        metrics["precision_weighted"] = None
        metrics["recall_weighted"] = None
        metrics["f1_weighted"] = None

    metrics_path = os.path.join(output_dir, "ml_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = os.path.join(output_dir, "ml_confusion_matrix.csv")
    np.savetxt(cm_path, confusion, delimiter=",", fmt="%d")

    return {"metrics_json": metrics_path, "confusion_csv": cm_path}