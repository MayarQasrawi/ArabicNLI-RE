"""
trainer.py
----------
NLI-RE training objective: combined Loss = L_WCE + L_NCE.

A combined loss to address the inherent class imbalance between positive 
(entailment / relation present) and negative (non-entailment / relation absent) 
instances:

    Loss = L_WCE + L_NCE

L_WCE — Weighted Cross-Entropy loss:
    Penalises misclassification of positive instances (relation present)
    more heavily by assigning a higher weight w_p to the positive class
    relative to the negative weight w_n.

L_NCE — Noise Contrastive Estimation loss:
    Improves discrimination between positive and negative instances by
    pulling the score of the true class above all competing classes via
    a temperature-scaled softmax (temperature τ).
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer


class ContrastiveTrainer(Trainer):
    """
    Extends :class:`transformers.Trainer` with the NLI-RE combined loss:

        Loss = L_WCE + L_NCE

    The model head outputs logits of shape (N, 2), where:
        - index 0 corresponds to the *False* label (relation absent)
        - index 1 corresponds to the *True* label  (relation present)

    Args:
        tau:      Temperature τ in the NCE softmax denominator.
                  Lower values sharpen the distribution; default 1.0.
        class_weights: Per-class weights [w_n, w_p] for the cross-entropy
                  loss.  w_p > w_n penalises false negatives on
                  positive (entailment) instances more heavily.
        **kwargs: Forwarded to :class:`transformers.Trainer`.
    """

    def __init__(
        self,
        tau: float = 1.0,
        class_weights: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tau = tau
        self.class_weights = class_weights or [1.0, 1.0]

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, object]]:
        """
        Compute the combined NLI-RE training loss: Loss = L_WCE + L_NCE.
        --------------------------
        The feature vector H from the Sentence Encoder is passed through a
        fully connected layer to produce per-class logits:

            ŷ_i = σ(H W + b)

        The predicted label is True (relation present) when the positive-class
        logit dominates; False otherwise.

        L_WCE — Weighted Cross-Entropy
        ----------------------------------------
        Standard cross-entropy weighted by [w_n, w_p] so that
        misclassifying a positive instance (relation present) incurs a
        larger penalty, mitigating the natural class imbalance in RE corpora.

        L_NCE — Noise Contrastive Estimation
        ----------------------------------------------
        For each sample i, the NCE loss is the negative log-probability of
        the true class under a temperature-scaled softmax over all classes:

            L_NCE = −(1/N) Σ_i log [ exp(s(y_i)/τ) / Σ_j exp(s(y_j)/τ) ]

        where s(y_i) denotes the logit at the ground-truth label position,
        and τ is the temperature that controls distribution sharpness.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits: torch.Tensor = outputs.get("logits")   # (N, num_classes)

        num_classes = logits.size(1)

        # Discard any samples whose label index falls outside the class range
        non_zero_indices = labels < num_classes
        logits = logits[non_zero_indices]
        labels = labels[non_zero_indices]

        # ========== L_NCE ============================
        # s(y_i): logit at the ground-truth class for each sample i
        positive_scores = torch.gather(logits, 1, labels.view(-1, 1))

        # Σ_j exp(s(y_j)/τ): partition function over all classes
        negative_scores = torch.sum(torch.exp(logits / self.tau), dim=1)

        # −log [ exp(s(y_i)/τ) / Σ_j exp(s(y_j)/τ) ], averaged over batch
        loss_nce = -torch.log(
            torch.exp(positive_scores / self.tau) / (positive_scores.exp() + negative_scores)
        )
        loss_nce = torch.mean(loss_nce)

        # ========== L_WCE ============================
        # Class weights [w_n, w_p] penalise positive-instance errors more heavily
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights, device=model.device)
        )
        loss_cross = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )

        loss = loss_nce + loss_cross
        return (loss, outputs) if return_outputs else loss
