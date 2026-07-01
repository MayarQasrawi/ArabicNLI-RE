"""
helpers.py
----------
Shared utilities for the NLI-RE training pipeline.

Covers reproducibility (seed control), label map construction for the
binary entailment task {True, False}, configuration loading, and device
detection.
"""

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for full reproducibility across runs.

    Applied before each fold in the K-Fold loop to ensure that model
    initialisation and data sampling are identical across experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Label map utilities
# ---------------------------------------------------------------------------

def build_label_maps(label_list: List[str]) -> tuple[Dict[str, int], Dict[int, str]]:
    """
    Build forward and inverse label mappings for the binary entailment task.

    The NLI-RE Relation Inference layer outputs a score ŷ_i whose argmax 
    index is mapped back to a human-readable label via these dicts.

    Args:
        label_list: Ordered list of unique entailment labels,
                    e.g. ["True", "False"] or ["False", "True"].

    Returns:
        Tuple of:
          - label_map     : str → int  (e.g. {"True": 1, "False": 0})
          - inv_label_map : int → str  (e.g. {1: "True", 0: "False"})
    """
    label_map: Dict[str, int] = {v: int(i) for i, v in enumerate(label_list)}
    inv_label_map: Dict[int, str] = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """
    Load the NLI-RE pipeline configuration from a YAML file.

    The config controls the sentence encoder (model name, max_len),
    the training hyperparameters, the loss weights (w_p, w_n, τ),
    data paths, and output directories.

    Args:
        config_path: Path to ``configs/config.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available torch device and print a summary.

    GPU acceleration is strongly recommended for fine-tuning ARBERTv2
    on NLI-RE sentence pairs.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] GPU detected — using: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] No GPU available — using CPU.")
    return device
