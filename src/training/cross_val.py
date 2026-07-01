"""
cross_val.py
------------
Stratified K-Fold training loop for the NLI-RE relation classifier.

The NLI-RE framework fine-tunes a transformer sentence encoder (UBC-NLP/ARBERTv2) 
on premise-hypothesis pairs to predict binary entailment.
To obtain robust generalisation estimates and reduce sensitivity to a single
train/validation split, we use Stratified K-Fold cross-validation, preserving
the True/False class ratio in every fold.

Each fold:
  1. Partitions the training premise-hypothesis pairs into train and held-out
     subsets while keeping the class distribution balanced.
  2. Fine-tunes ARBERTv2 with the combined Loss = L_WCE + L_NCE objective
     (Section 3, Training Objective) via :class:`ContrastiveTrainer`.
  3. Evaluates on the fixed development set and saves the best checkpoint
     (selected by Micro-F1) for ensemble inference at test time.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, TrainingArguments

from src.data.dataset import ClassificationDataset
from src.data.loader import NLIDataset
from src.models.trainer import ContrastiveTrainer
from src.evaluation.metrics import compute_metrics
from src.utils.helpers import set_seed


def run_cross_validation(
    dataset: NLIDataset,
    label_map: Dict[str, int],
    inv_label_map: Dict[int, str],
    model_name: str,
    max_len: int,
    num_folds: int,
    training_cfg: dict,
    loss_cfg: dict,
    output_base_dir: str | Path,
    data_column: str,
    label_column: str,
) -> Tuple[List[dict], int]:
    """
    Run Stratified K-Fold fine-tuning of the NLI-RE sentence encoder.

    For each fold, ARBERTv2 is fine-tuned on a stratified subset of the
    premise-hypothesis training pairs. The Relation Inference layer
    predicts binary entailment, and the model is optimised with the combined
    Loss = L_WCE + L_NCE to handle positive/negative instance imbalance.  
    The best checkpoint per fold (highest Micro-F1 on the dev set)
    is saved for ensemble inference.

    Args:
        dataset:          :class:`NLIDataset` holding train/dev/test splits.
        label_map:        Entailment label → integer id  (e.g. {"True": 1, "False": 0}).
        inv_label_map:    Integer id → entailment label  (reverse of label_map).
        model_name:       HuggingFace identifier for the sentence encoder T.
        max_len:          Maximum sub-word token length for the encoder (Eq. 1).
        num_folds:        Number of stratified folds K.
        training_cfg:     Hyperparameters sub-dict from ``configs/config.yaml``.
        loss_cfg:         Loss hyperparameters (tau, class_weights) from config.
        output_base_dir:  Root directory; per-fold checkpoints written to
                          ``<base>/cls_train_{fold}/best_model/``.
        data_column:      Name of the premise-hypothesis sentence column.
        label_column:     Name of the binary entailment label column.

    Returns:
        Tuple of (all_fold_results, best_fold_index) where all_fold_results
        is a list of per-fold evaluation dicts and best_fold_index is the
        fold whose dev Micro-F1 was highest.
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Stratified split preserves the True/False ratio across all folds
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=123)

    all_results: List[dict] = []
    best_f1 = 0.0
    best_fold = 0

    for fold_num, (train_idx, _) in enumerate(
        kf.split(dataset.train, dataset.train[label_column])
    ):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_num + 1} / {num_folds}")
        print(f"{'='*60}")

        fold_output_dir = output_base_dir / f"cls_train_{fold_num}"

        # ======= Sentence Encoder input ==========================================
        # train_dataset: stratified fold subset of premise-hypothesis pairs
        # val_dataset:   fixed development set used across all folds
        train_dataset = ClassificationDataset(
            texts=list(dataset.train[data_column].iloc[train_idx]),
            targets=list(dataset.train[label_column].iloc[train_idx]),
            model_name=model_name,
            max_len=max_len,
            label_map=label_map,
        )
        val_dataset = ClassificationDataset(
            texts=dataset.dev[data_column].tolist(),
            targets=dataset.dev[label_column].tolist(),
            model_name=model_name,
            max_len=max_len,
            label_map=label_map,
        )

        # ======= Training arguments ==========================================
        training_args = TrainingArguments(
            output_dir=str(fold_output_dir),
            learning_rate=float(training_cfg["learning_rate"]),
            fp16=bool(training_cfg.get("fp16", False)),
            per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
            num_train_epochs=int(training_cfg["num_train_epochs"]),
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            seed=int(training_cfg["seed"]),
        )

        set_seed(training_args.seed)

        # ======= Sentence encoder T = ARBERTv2 with Relation Inference head ================
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            return_dict=True,
            num_labels=len(label_map),   # 2: True / False
        )

        # ======= ContrastiveTrainer: Loss = L_WCE + L_NCE ================
        trainer = ContrastiveTrainer(
            tau=loss_cfg.get("tau", 1.0),
            class_weights=loss_cfg.get("class_weights", [1.0, 1.0]),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Store label↔id mappings in the model config for inference
        trainer.model.config.label2id = label_map
        trainer.model.config.id2label = inv_label_map

        trainer.train()
        results = trainer.evaluate()
        all_results.append(results)
        print(f"[fold {fold_num}] Results: {results}")

        # ======= Persist best checkpoint for ensemble inference ===============
        best_model_dir = fold_output_dir / "best_model"
        trainer.save_model(str(best_model_dir))
        val_dataset.tokenizer.save_pretrained(str(best_model_dir))
        print(f"[fold {fold_num}] Saved best model → {best_model_dir}")

        if results["eval_micro_f1"] > best_f1:
            best_f1 = results["eval_micro_f1"]
            best_fold = fold_num
            print(f"[fold {fold_num}] *** New best model (F1={best_f1:.4f}) ***")

    print(f"\n[cv] Best fold: {best_fold}  |  Best Micro-F1: {best_f1:.4f}")
    return all_results, best_fold