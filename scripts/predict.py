"""
predict.py
----------
Entry point: load fold models → ensemble inference → save predictions + report.

Usage
-----
    python scripts/predict.py \\
        --config  configs/config.yaml \\
        --input   data/nli/test.jsonl \\
        --output  outputs/predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running as `python scripts/predict.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.metrics import (
    print_classification_report,
    run_ensemble_inference,
    save_results,
)
from src.utils.helpers import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble inference for Arabic NLI-RE.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to input JSONL file (default: test_path from config).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for output JSONL file.")
    return parser.parse_args()


def _load_jsonl(path: str | Path) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    device_obj  = get_device()
    cuda_device = 0 if str(device_obj) == "cuda" else -1

    data_cfg   = cfg["data"]
    model_cfg  = cfg["model"]
    output_cfg = cfg["output"]

    # 1. Load the test split (or a custom --input file)
    input_path = args.input or data_cfg["test_path"]
    test_df    = _load_jsonl(input_path)
    texts      = test_df[data_cfg["data_column"]].tolist()
    print(f"[predict] Input: {input_path}  ({len(texts)} rows)")

    # 2. Collect fold model directories
    base_dir        = Path(output_cfg["base_dir"])
    fold_model_dirs = sorted(base_dir.glob("cls_train_*/best_model"))
    if not fold_model_dirs:
        raise FileNotFoundError(
            f"No fold models found under '{base_dir}'. Run train.py first."
        )
    print(f"[predict] Found {len(fold_model_dirs)} fold model(s).")

    # 3. Ensemble inference
    cross_val_df = run_ensemble_inference(
        texts=texts,
        fold_model_dirs=fold_model_dirs,
        max_len=model_cfg["max_len"],
        device=cuda_device,
    )

    # 4. Report & save 
    label_column = data_cfg["label_column"]
    if label_column in test_df.columns:
        print_classification_report(test_df[label_column], cross_val_df["preds"])

    output_dir = Path(args.output).parent if args.output else base_dir
    pred_file  = Path(args.output).name   if args.output else output_cfg["predictions_file"]

    save_results(
        test_df=test_df,
        cross_val_df=cross_val_df,
        label_column=label_column,
        output_dir=output_dir,
        predictions_file=pred_file,
        false_predictions_file=output_cfg["false_predictions_file"],
    )


if __name__ == "__main__":
    main()
