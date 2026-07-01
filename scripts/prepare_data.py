"""
prepare_data.py
---------------
Step 0 of the NLI-RE pipeline: generate NLI sentence pairs from raw RE records.

Usage
-----
    python scripts/prepare_data.py --config configs/config.yaml
    python scripts/prepare_data.py --n_positive 500 --n_negative 500
"""

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running as `python scripts/prepare_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.nli_generator import (
    generate_nli_pairs,
    load_jsonl,
    sample_balanced,
    save_jsonl,
    split_data,
) 
from src.utils.helpers import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NLI-RE premise-hypothesis pairs from raw RE records."
    )
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--n_positive", type=int, default=None)
    parser.add_argument("--n_negative", type=int, default=None)
    parser.add_argument("--input",      type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    data_cfg = cfg["data"]

    raw_path   = Path(args.input or data_cfg["raw_train_path"])
    n_positive = args.n_positive or data_cfg.get("n_positive", 200)
    n_negative = args.n_negative or data_cfg.get("n_negative", 200)
    seed       = data_cfg.get("seed", 42)

    print(f"[prepare] Loading raw records from: {raw_path}")
    records = load_jsonl(raw_path)
    print(f"[prepare] Loaded {len(records):,} records.")

    print("[prepare] Generating NLI premise-hypothesis pairs ...")
    nli_pairs = generate_nli_pairs(records, seed=seed)
    n_true  = sum(1 for p in nli_pairs if p["Label"] == "True")
    n_false = sum(1 for p in nli_pairs if p["Label"] == "False")
    print(f"[prepare] Generated {len(nli_pairs):,} pairs  (True: {n_true:,} | False: {n_false:,})")

    print(f"[prepare] Sampling {n_positive} positive + {n_negative} negative pairs ...")
    sampled = sample_balanced(nli_pairs, n_positive, n_negative, seed=seed)

    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio   = data_cfg.get("val_ratio",   0.10)
    train, val, test = split_data(sampled, train_ratio, val_ratio, seed=seed)
    print(f"[prepare] Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    out_dir = Path("data/nli")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val,   out_dir / "val.jsonl")
    save_jsonl(test,  out_dir / "test.jsonl")

    print(f"\n[prepare] ✅  Saved to {out_dir}/")
    print(f"           train.jsonl  ({len(train)} records)")
    print(f"           val.jsonl    ({len(val)} records)")
    print(f"           test.jsonl   ({len(test)} records)")
    print(f"\n[prepare] Run training next:")
    print(f"           python scripts/train.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
