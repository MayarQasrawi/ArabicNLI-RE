# NLI-RE: Arabic Relation Extraction as Natural Language Inference

Implementation of **NLI-RE**, a framework that reformulates relation extraction (RE) as a Natural Language Inference (NLI) task. Given an input sentence *s* containing two named entity mentions *n_i* and *n_j*, NLI-RE treats *s* as the premise and automatically constructs a hypothesis by verbalizing a candidate relation *r* between *n_i* and *n_j* using a predefined Arabic template. A transformer encoder then predicts whether the premise **entails** the hypothesis — indicating the presence of relation *r* between the entity pair.

---

## Table of Contents

- [Framework Overview](#framework-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [License](#license)

---

## Framework Overview

NLI-RE proceeds through three steps (Section 3):

### 1. Template Construction
For each of the **40 relation types**, a relation-aware Arabic template `T_r(e1, e2)` verbalizes a candidate relation between subject entity `e1` and object entity `e2` into a hypothesis string `h`. For example:

| Relation | Subject | Object | Hypothesis h |
|---|---|---|---|
| `Location.located_in` | قبة الصخرة | مدينة القدس | قبة الصخرة يقع في مدينة القدس |
| `Personal.has_occupation` | حبيب بولس | مدير مال عكا | حبيب بولس يعمل مدير مال عكا |
| `Affiliation.member_of` | أطباء بلا حدود | الأمم المتحدة | أطباء بلا حدود عضو في الأمم المتحدة |

### 2. Sentence Encoder
A transformer encoder `T` (UBC-NLP/ARBERTv2) processes the concatenated premise-hypothesis pair:

```
H = T([CLS] s [SEP] h)
```

### 3. Relation Inference
The feature vector `H` is passed through a fully connected layer:

```
ŷ_i = σ(H W + b)
```

**True** → relation `r` exists between `(n_i, n_j)` in `s`.  
**False** → relation `r` does not exist.

### Training Objective

```
Loss = L_WCE + L_NCE
```

**L_WCE** (Eq. 3) — Weighted Cross-Entropy with `[w_n, w_p]` to address class imbalance.  
**L_NCE** (Eq. 4) — Noise Contrastive Estimation with temperature `τ` to improve discrimination.

---

## Project Structure

```
arabic-nli-classifier/
│
├── configs/
│   └── config.yaml                  # All hyperparameters and paths
│
├── data/
│   ├── nli/                         # ✅ Committed — NLI sample pairs (train/val/test)
│   │   ├── train.jsonl              #    280 premise-hypothesis pairs
│   │   ├── val.jsonl                #     60 pairs
│   │   └── test.jsonl               #     60 pairs
│   └── raw/                         # ❌ Not committed — place full dataset here
│       └── train.jsonl              #    Full RE records (sentence/subject/object/relation)
│
├── src/
│   ├── data/
│   │   ├── templates.py             # 40 Arabic relation verbalization templates T_r
│   │   ├── nli_generator.py         # Generates [CLS] s [SEP] h pairs from RE records
│   │   ├── dataset.py               # PyTorch Dataset — Sentence Encoder input (Eq. 1)
│   │   └── loader.py                # Loads NLI JSONL splits into DataFrames
│   │
│   ├── models/
│   │   └── trainer.py               # ContrastiveTrainer: Loss = L_WCE + L_NCE
│   │
│   ├── training/
│   │   └── cross_val.py             # Stratified K-Fold fine-tuning loop
│   │
│   ├── evaluation/
│   │   └── metrics.py               # Micro-F1, ensemble inference, JSONL reporting
│   │
│   └── utils/
│       └── helpers.py               # Seed control, label maps, config loader
│
├── scripts/
│   ├── prepare_data.py              # Step 0: generate NLI pairs from raw RE records
│   ├── train.py                     # Step 1: K-Fold training
│   └── predict.py                   # Step 2: ensemble inference
│
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_loss.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/aaljabari/arabic-nli-re.git
cd arabic-nli-re

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset

### Included sample (ready to use)

The repository includes a balanced sample of **400 NLI pairs** in `data/nli/`:

| File | Pairs | True | False |
|---|---|---|---|
| `train.jsonl` | 280 | 138 | 142 |
| `val.jsonl`   |  60 |  38 |  22 |
| `test.jsonl`  |  60 |  30 |  30 |

Each record contains exactly two fields:

```json
{
  "nli_sentence": "[CLS] رسالة من مدير مال عكا حبيب بولس ... [SEP] حبيب بولس يعمل كـ / مهنته مدير مال عكا",
  "Label": "True"
}
```

### Using your own full dataset

Place your raw RE records at `data/raw/train.jsonl`, then run:

```bash
python scripts/prepare_data.py --config configs/config.yaml

# Custom sample size:
python scripts/prepare_data.py --n_positive 500 --n_negative 500
```

Raw records must contain the fields: `sentence`, `subject`, `object`, `relation`.

---

## Configuration

```yaml
model:
  name: "UBC-NLP/ARBERTv2"   # Sentence encoder
  max_len: 128                 # Max tokens for [CLS] s [SEP] h
  num_folds: 2                 # K in Stratified K-Fold

loss:
  tau: 1.0                     # Temperature τ for L_NCE
  class_weights: [0.2, 1.0]   # [w_n, w_p] for L_WCE

data:
  train_path: "data/nli/train.jsonl"
  dev_path:   "data/nli/val.jsonl"
  test_path:  "data/nli/test.jsonl"
```

---

## Usage

### Run on the included sample (no setup needed)

```bash
# Train
python scripts/train.py --config configs/config.yaml

# Predict
python scripts/predict.py --config configs/config.yaml
```

### Run on the full dataset

```bash
# Step 0 — generate NLI pairs from raw RE records
python scripts/prepare_data.py --config configs/config.yaml

# Step 1 — train
python scripts/train.py --config configs/config.yaml

# Step 2 — predict
python scripts/predict.py --config configs/config.yaml
```
---

## Citation

If you use this code or the dataset in your research, please cite the following papers:

```bibtex
@inproceedings{aljabari-etal-2025-wojoodrelations,
    title = "$\mathrm{Wojood^{Relations}}$: {A}rabic Relation Extraction Corpus and Modeling",
    author = "Aljabari, Alaa  and
      Khalilia, Mohammed  and
      Jarrar, Mustafa",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1741/",
    doi = "10.18653/v1/2025.emnlp-main.1741",
    pages = "34342--34360",
    ISBN = "979-8-89176-332-6",
}

@inproceedings{aljabari-etal-2025-wojoodontology,
    title = "{W}ojood{O}ntology: Ontology-Driven {LLM} Prompting for Unified Information Extraction Tasks",
    author = "Aljabari, Alaa  and
      Hamad, Nagham  and
      Khalilia, Mohammed  and
      Jarrar, Mustafa",
    booktitle = "Proceedings of The Third Arabic Natural Language Processing Conference",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.arabicnlp-main.14/",
    doi = "10.18653/v1/2025.arabicnlp-main.14",
    pages = "179--193",
    ISBN = "979-8-89176-352-4",
}

@inproceedings{aljabari-etal-2024-event,
    title = "Event-Arguments Extraction Corpus and Modeling using {BERT} for {A}rabic",
    author = "Aljabari, Alaa  and
      Duaibes, Lina  and
      Jarrar, Mustafa  and
      Khalilia, Mohammed",
    booktitle = "Proceedings of the Second Arabic Natural Language Processing Conference",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.arabicnlp-1.26/",
    doi = "10.18653/v1/2024.arabicnlp-1.26",
    pages = "309--319",
}
```
---

## License

[MIT](LICENSE)
