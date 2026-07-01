"""
loader.py
---------
Loads and prepares the NLI-RE premise-hypothesis sentence pairs for training.

In the NLI-RE framework, each input sentence s containing two
named entity mentions n_i and n_j is treated as the premise. The corresponding
hypothesis is a verbalized candidate relation constructed from a template T_r.
This module loads the pre-built (premise, hypothesis, label) triples from
JSONL files and wraps them in a structured container for the training pipeline.

Data format (JSONL — one record per line):
    {
        "sentence_id":  "...",
        "sentence":     "raw Arabic sentence s",
        "subject":      "entity n_i",
        "object":       "entity n_j",
        "relation":     "Relation.type",
        "hypothesis":   "verbalized Arabic hypothesis h",
        "nli_sentence": "[CLS] s [SEP] h",
        "Label":        "True" | "False"
    }
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class NLIDataset:
    """
    Container for the NLI-RE sentence-pair splits and their metadata.

    Each row in the DataFrames represents one premise-hypothesis pair
    (nli_sentence) with its binary entailment label (True / False),
    as produced by the Template Construction step (Section 3).

    Attributes:
        name:        Human-readable dataset identifier.
        train:       Training split DataFrame.
        dev:         Validation split DataFrame.
        test:        Test split DataFrame.
        label_list:  Binary label set {True, False} derived from the training split.
    """

    name: str
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    label_list: List[str] = field(default_factory=list)


def _load_jsonl(path: str | Path, random_state: int) -> pd.DataFrame:
    """Load a JSONL file into a shuffled DataFrame."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_nli_data(
    train_path: str | Path,
    dev_path: str | Path,
    test_path: str | Path,
    label_column: str,
    random_state: int = 42,
    name: str = "NLI",
) -> NLIDataset:
    """
    Load the NLI-RE premise-hypothesis pairs from JSONL files.

    Each split is shuffled before use so that the Stratified K-Fold
    partitioning in the training loop is not biased by the original
    file ordering.

    Args:
        train_path:    Path to the training split JSONL file.
        dev_path:      Path to the validation split JSONL file.
        test_path:     Path to the test split JSONL file.
        label_column:  Column holding binary entailment labels (True / False).
        random_state:  Shuffle seed for reproducibility (default 42).
        name:          Dataset identifier string.

    Returns:
        Populated :class:`NLIDataset` instance ready for the training pipeline.
    """
    train = _load_jsonl(train_path, random_state)
    dev   = _load_jsonl(dev_path,   random_state)
    test  = _load_jsonl(test_path,  random_state)

    label_list: List[str] = train[label_column].unique().tolist()
    print(f"[data] Entailment labels detected: {label_list}")
    print(f"[data] Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")

    return NLIDataset(
        name=name,
        train=train,
        dev=dev,
        test=test,
        label_list=label_list,
    )
