"""
dataset.py
----------
Sentence Encoder input preparation for the NLI-RE framework.

The input to the transformer T is a concatenated sequence of 
the premise s and the hypothesis h:

    H = T([CLS] s [SEP] h)

where [CLS] is the classification token, [SEP] separates the premise from
the hypothesis, and H ∈ R^d is the resulting contextual representation.

This module tokenises the pre-built nli_sentence strings — which already
encode the full [premise, hypothesis] pair — and maps binary entailment
labels {True, False} to integer ids for the Trainer.
"""

from typing import Dict, List

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.data.processors.utils import InputFeatures


class ClassificationDataset(Dataset):
    """
    Prepares premise-hypothesis pairs for the NLI-RE Sentence Encoder.

    Each nli_sentence is the full concatenated pair:
        [CLS] premise [SEP] hypothesis
    The transformer T processes this sequence to produce the feature vector H
    that feeds the Relation Inference layer.

    Args:
        texts:       Pre-built premise-hypothesis sentence strings (nli_sentence).
        targets:     Binary entailment labels — True (relation present) or
                     False (relation absent).
        model_name:  HuggingFace identifier for the sentence encoder T
                     (default: UBC-NLP/ARBERTv2).
        max_len:     Maximum sub-word token length for the encoder input.
                     Sequences exceeding this are truncated.
        label_map:   Mapping from entailment label string to integer id,
                     e.g. {"True": 1, "False": 0}.
    """

    def __init__(
        self,
        texts: List[str],
        targets: List[str],
        model_name: str,
        max_len: int,
        label_map: Dict[str, int],
    ) -> None:
        super().__init__()
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.label_map = label_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> InputFeatures:
        # Normalise whitespace in the premise-hypothesis string before encoding
        text = " ".join(str(self.texts[index]).split())

        # Produce [CLS] s [SEP] h token sequence as in Eq. 1
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        return InputFeatures(**inputs, label=self.label_map[self.targets[index]])

    # ------------------------------------------------------------------
    # Diagnostic helper
    # ------------------------------------------------------------------

    def report_truncation(self) -> int:
        """
        Report how many premise-hypothesis pairs exceed ``max_len`` tokens.

        Truncated sequences lose tail tokens of the hypothesis; a higher
        max_len reduces this at the cost of memory.
        """
        over = sum(
            len(self.tokenizer.tokenize(str(t))) > self.max_len for t in self.texts
        )
        print(f"[dataset] Truncated sequences: {over}/{len(self.texts)}")
        return over
