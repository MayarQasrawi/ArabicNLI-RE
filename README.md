# NLI-RE: Arabic Relation Extraction as Natural Language Inference

**NLI-RE** reformulates relation extraction (RE) as a Natural Language Inference (NLI) task. Given an input sentence containing two named entity mentions, the sentence is treated as the premise and a hypothesis is constructed by verbalizing a candidate relation between the two entities using a predefined Arabic template. A transformer encoder then predicts whether the premise entails the hypothesis, indicating the presence of that relation between the entity pair.

---

## Table of Contents

- [Framework Overview](#framework-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Framework Overview

NLI-RE consists of three steps:

### 1. Template Construction
For each of the **40 relation types**, a relation-aware Arabic template maps a candidate relation between subject entity `e1` and object entity `e2` into a hypothesis string. For example:

| Relation | Subject | Object | Hypothesis |
|---|---|---|---|
| `Location.located_in` | قبة الصخرة | مدينة القدس | قبة الصخرة تقع في مدينة القدس |
| `Personal.has_occupation` | حبيب بولس | مدير مال عكا | حبيب بولس يعمل كـ / مهنته مدير مال عكا |
| `Affiliation.member_of` | أطباء بلا حدود | الأمم المتحدة | أطباء بلا حدود عضو في الأمم المتحدة |

### 2. Sentence Encoding
A transformer encoder (UBC-NLP/ARBERTv2) processes the concatenated premise-hypothesis pair:

```
[CLS] sentence [SEP] hypothesis
```

### 3. Relation Inference
The encoded representation is passed through a fully connected layer producing a binary prediction:

- **True** → the relation holds between the entity pair in the sentence
- **False** → the relation does not hold

### Training Objective

The model is trained with a combined loss to handle class imbalance between positive (relation present) and negative (relation absent) instances:

```
Loss = L_WCE + L_NCE
```

- **L_WCE** — Weighted Cross-Entropy with per-class weights `[w_n, w_p]`, where `w_p > w_n` penalises missed positive instances more heavily.
- **L_NCE** — Noise Contrastive Estimation with temperature `τ` to improve separation between positive and negative instances.

---

## Project Structure

```
arabic-nli-classifier/
│
├── configs/
│   └── config.yaml                  # Hyperparameters and paths
│
├── data/
│   ├── nli/                         # NLI pairs included in the repository
│   │   ├── train.jsonl              # 280 premise-hypothesis pairs
│   │   ├── val.jsonl                #  60 pairs
│   │   └── test.jsonl               #  60 pairs
│   └── raw/                         # Place full dataset here (not committed)
│       └── train.jsonl
│
├── src/
│   ├── data/
│   │   ├── templates.py             # 40 Arabic relation verbalization templates
│   │   ├── nli_generator.py         # Generates [CLS] s [SEP] h pairs from RE records
│   │   ├── dataset.py               # PyTorch Dataset for NLI sentence pairs
│   │   └── loader.py                # Loads NLI JSONL splits
│   │
│   ├── models/
│   │   └── trainer.py               # ContrastiveTrainer: Loss = L_WCE + L_NCE
│   │
│   ├── training/
│   │   └── cross_val.py             # Stratified K-Fold fine-tuning loop
│   │
│   ├── evaluation/
│   │   └── metrics.py               # Micro-F1, ensemble inference, reporting
│   │
│   └── utils/
│       └── helpers.py               # Seed control, label maps, config loader
│
├── scripts/
│   ├── prepare_data.py              # Prepare NLI pairs from raw RE records
│   ├── train.py                     # Run K-Fold training
│   └── predict.py                   # Run ensemble inference
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/alaa-aljabari/ArabicNLI-RE.git
cd ArabicNLI-RE

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset

### Included sample

The repository includes a sample of **400 NLI pairs** in `data/nli/`:

| File | Pairs | True | False |
|---|---|---|---|
| `train.jsonl` | 280 | 138 | 142 |
| `val.jsonl`   |  60 |  38 |  22 |
| `test.jsonl`  |  60 |  30 |  30 |

Each record contains:

```json
{
  "nli_sentence": "[CLS] رسالة من مدير مال عكا حبيب بولس ... [SEP] حبيب بولس يعمل كـ / مهنته مدير مال عكا",
  "Label": "True"
}
```

### Full dataset (WojoodRelations)

The full **WojoodRelations** dataset is available via registration:

> 📥 [Download WojoodRelations](https://docs.google.com/forms/d/e/1FAIpQLSdtN7OtVQBkRYktpnHd8xmyLi_KP2eZHc4T41IaWtIAptM4mw/viewform)

Tools, datasets, and models are also available through the online demo:

> 🔗 [Arabic Relation Extraction Tools, Datasets and Models](https://sina.birzeit.edu/relations/)

### Using your own data

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

### Google Colab

A ready-to-run Colab notebook is available. Click the badge below to open it directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZRRdWUaX_bNSiJ_61gV-Cp4oscdN_Lpn)

---

## Citation

If you use this code or the dataset in your research, please cite:

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
