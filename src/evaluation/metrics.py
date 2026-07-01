"""
metrics.py
----------
Evaluation utilities for the NLI-RE relation classifier.

After fine-tuning, the Relation Inference layer produces a predicted
entailment label ŷ_i ∈ {True, False} for each premise-hypothesis pair.
A True prediction indicates that relation r specified in the hypothesis h
exists between the entity pair (n_i, n_j) in the premise sentence s.

This module provides:
  - compute_metrics: Micro-F1 and accuracy callback for the HuggingFace Trainer.
  - run_ensemble_inference: averages softmax scores across all K fold models.
  - print_classification_report: per-class precision, recall, F1.
  - save_results: persists predictions and error analysis files as JSONL.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import more_itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
from transformers import EvalPrediction, pipeline


# ---------------------------------------------------------------------------
# HuggingFace Trainer evaluation callback
# ---------------------------------------------------------------------------

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Compute Micro-F1 and accuracy from Relation Inference logits.

    The argmax over logits gives the predicted entailment label
    (True = relation present, False = relation absent), which is then
    compared against gold labels to compute corpus-level Micro-F1.
    Micro-F1 is the primary model-selection criterion across folds.

    Compatible with :class:`transformers.Trainer`.
    """
    preds = np.argmax(p.predictions, axis=1)
    assert len(preds) == len(p.label_ids), "Prediction / label length mismatch."
    return {
        "micro_f1": f1_score(p.label_ids, preds, average="micro"),
        "accuracy": accuracy_score(p.label_ids, preds),
    }


# ---------------------------------------------------------------------------
# Ensemble inference across K fold models
# ---------------------------------------------------------------------------

def run_ensemble_inference(
    texts: List[str],
    fold_model_dirs: List[str | Path],
    max_len: int,
    batch_size: int = 32,
    device: int = 0,
) -> pd.DataFrame:
    """
    Perform ensemble inference by averaging softmax scores across all K folds.

    Each fold model independently scores every premise-hypothesis pair.
    The per-class softmax probabilities are averaged across folds, and the
    class with the highest average probability is taken as the final
    entailment prediction — True (relation present) or False (relation absent).

    Args:
        texts:            Pre-built premise-hypothesis sentence strings (nli_sentence).
        fold_model_dirs:  Paths to the best-checkpoint directory for each fold.
        max_len:          Maximum sub-word token length (must match training).
        batch_size:       Inference batch size for the HuggingFace pipeline.
        device:           CUDA device index (-1 for CPU).

    Returns:
        DataFrame with one column per fold model, plus ``preds`` (final
        entailment label) and ``confidence`` (winning class probability).
    """
    cross_val_df = pd.DataFrame()

    for fold_idx, model_dir in enumerate(fold_model_dirs):
        print(f"\n[inference] Fold {fold_idx}: {model_dir}")
        pipe = pipeline(
            "text-classification",
            model=str(model_dir),
            device=device,
            top_k=None,          # return scores for all classes
            max_length=max_len,
            truncation=True,
        )
        preds = []
        for batch in tqdm(more_itertools.chunked(texts, batch_size)):
            results = pipe(batch)
            # Normalise: each result is a list of {"label":..,"score":..} dicts
            for result in results:
                if isinstance(result, dict):
                    result = [result]
                preds.append(result)
        cross_val_df[f"model_{fold_idx}"] = preds

    # ==== Average softmax scores across K fold models ===============
    num_folds = len(fold_model_dirs)
    final_labels: List[str] = []
    final_scores: List[float] = []

    for _, row in cross_val_df.iterrows():
        total_score: Dict[str, float] = defaultdict(float)
        for pred in row:
            for cls in pred:
                total_score[cls["label"]] += cls["score"]

        avg_score = {k: v / num_folds for k, v in total_score.items()}
        best_label = max(avg_score, key=avg_score.get)
        final_labels.append(best_label)
        final_scores.append(avg_score[best_label])

    cross_val_df["preds"]      = final_labels
    cross_val_df["confidence"] = final_scores
    return cross_val_df


# ---------------------------------------------------------------------------
# Reporting and output persistence
# ---------------------------------------------------------------------------

def print_classification_report(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    digits: int = 4,
) -> None:
    """
    Print per-class precision, recall, and F1 for the binary entailment task.

    True  → relation r exists between (n_i, n_j) in sentence s  (positive).
    False → relation r does not exist between (n_i, n_j) in s   (negative).
    """
    print("\n[evaluation] Classification Report:")
    print(classification_report(true_labels, predicted_labels, digits=digits))


def save_results(
    test_df: pd.DataFrame,
    cross_val_df: pd.DataFrame,
    label_column: str,
    output_dir: str | Path,
    predictions_file: str = "predictions.jsonl",
    false_predictions_file: str = "false_predictions.jsonl",
) -> None:
    """
    Persist NLI-RE ensemble predictions and error-analysis artefacts to disk
    as JSONL files (one JSON object per line, UTF-8).

    Two files are written:
    - ``predictions_file``:       Full test-set predictions with confidence scores.
    - ``false_predictions_file``: Rows where the predicted entailment label
                                  disagrees with the gold label; useful for
                                  template and schema error analysis.

    Args:
        test_df:               Original test DataFrame (must contain ``label_column``).
        cross_val_df:          Output of :func:`run_ensemble_inference`.
        label_column:          Gold binary entailment label column name.
        output_dir:            Directory to write output files.
        predictions_file:      Filename for the full predictions JSONL.
        false_predictions_file: Filename for the misclassified-pair JSONL.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge test fields with predictions
    combined = test_df.copy().reset_index(drop=True)
    combined["pred_label"] = cross_val_df["preds"].values
    combined["confidence"] = cross_val_df["confidence"].values

    def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    pred_path = output_dir / predictions_file
    _write_jsonl(combined, pred_path)
    print(f"[output] Predictions  → {pred_path}")

    false_mask     = combined[label_column] != combined["pred_label"]
    false_pred_path = output_dir / false_predictions_file
    _write_jsonl(combined[false_mask], false_pred_path)
    print(f"[output] False predictions ({false_mask.sum()}) → {false_pred_path}")