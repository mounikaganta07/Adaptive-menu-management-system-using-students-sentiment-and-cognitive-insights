import argparse
from src.config import Config
from src.pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser(description="Run hostel menu management pipeline")
    p.add_argument("--input", default=Config().raw_data_path, help="Path to input CSV")
    p.add_argument("--out", default=Config().output_dir, help="Output directory")
    args = p.parse_args()

    cfg = Config(raw_data_path=args.input, output_dir=args.out)
    paths = run_pipeline(cfg)
    for k, v in paths.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
