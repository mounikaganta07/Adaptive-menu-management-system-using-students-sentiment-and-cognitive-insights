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

    metrics_path = os.path.join(output_dir, "ml_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = os.path.join(output_dir, "ml_confusion_matrix.csv")
    np.savetxt(cm_path, confusion, delimiter=",", fmt="%d")

    return {"metrics_json": metrics_path, "confusion_csv": cm_path}
