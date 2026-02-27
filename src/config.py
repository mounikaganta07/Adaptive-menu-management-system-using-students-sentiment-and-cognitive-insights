from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    raw_data_path: str = "data/raw/synthetic_feedback_data.csv"
    processed_data_path: str = "data/processed/feedback_with_sentiment.csv"
    output_dir: str = "outputs"

    # VADER thresholds
    pos_threshold: float = 0.05
    neg_threshold: float = -0.05

    # Menu health scoring
    negative_penalty: float = 0.7   # penalize high negative ratio
    min_samples_per_item: int = 20  # avoid noisy actions
