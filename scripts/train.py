"""
train.py
--------
Entry point: load config → load data → run K-Fold training → report results.

Usage
-----
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path
from statistics import mean

# Ensure the repo root is on sys.path when running as `python scripts/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_nli_data
from src.training.cross_val import run_cross_validation
from src.utils.helpers import build_label_maps, get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Arabic NLI classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    get_device()  # informational print

    # ------------------------------------------------------------------ #
    # 1. Load data                                                        #
    # ------------------------------------------------------------------ #
    data_cfg = cfg["data"]
    dataset = load_nli_data(
        train_path=data_cfg["train_path"],
        dev_path=data_cfg["dev_path"],
        test_path=data_cfg["test_path"],
        label_column=data_cfg["label_column"],
    )

    label_map, inv_label_map = build_label_maps(dataset.label_list)
    print(f"[main] Label map: {label_map}")

    # ------------------------------------------------------------------ #
    # 2. Cross-validation training                                        #
    # ------------------------------------------------------------------ #
    model_cfg = cfg["model"]
    all_results, best_fold = run_cross_validation(
        dataset=dataset,
        label_map=label_map,
        inv_label_map=inv_label_map,
        model_name=model_cfg["name"],
        max_len=model_cfg["max_len"],
        num_folds=model_cfg["num_folds"],
        training_cfg=cfg["training"],
        loss_cfg=cfg["loss"],
        output_base_dir=cfg["output"]["base_dir"],
        data_column=data_cfg["data_column"],
        label_column=data_cfg["label_column"],
    )

    # ------------------------------------------------------------------ #
    # 3. Summary                                                          #
    # ------------------------------------------------------------------ #
    avg_f1 = mean(r["eval_micro_f1"] for r in all_results)
    print(f"\n[main] All fold results: {all_results}")
    print(f"[main] Average Micro-F1 across {model_cfg['num_folds']} folds: {avg_f1:.4f}")
    print(f"[main] Best fold: {best_fold}")


if __name__ == "__main__":
    main()
