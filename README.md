<p align="center">
  <h1 align="center">⚖️ UU ITE Violation Detection</h1>
  <p align="center">
    <strong>Deep Learning-based Classification of Indonesian Cyber Law (UU ITE) Violations in Social Media Comments</strong>
  </p>
  <p align="center">
    <a href="#models">Models</a> •
    <a href="#dataset">Dataset</a> •
    <a href="#preprocessing-pipeline">Preprocessing</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#demo">Demo</a> •
    <a href="#project-structure">Structure</a>
  </p>
</p>

---

## Overview

This project implements a multi-class text classification system to detect potential violations of **Indonesia's UU ITE (Undang-Undang Informasi dan Transaksi Elektronik)** in social media comments scraped from **YouTube** and **TikTok**. The system classifies comments into **6 categories** using two deep learning approaches: a **Hybrid CNN-BiLSTM** model with FastText embeddings and a fine-tuned **IndoBERT** transformer model.

### Classification Categories

| Label | Category | UU ITE Reference |
|:-----:|----------|------------------|
| `0` | **Netral** — No violation detected | — |
| `1` | **Judi Online** — Online gambling promotion | Pasal 27 ayat (2) |
| `2` | **Berita Hoaks** — Fake news / misinformation | Pasal 28 ayat (1) |
| `3` | **Pencemaran Nama Baik** — Defamation / cyberbullying | Pasal 27 ayat (3) |
| `4` | **Ujaran Kebencian (SARA)** — Hate speech (ethnicity, religion, race) | Pasal 28 ayat (2) |
| `5` | **Pornografi** — Obscene / sexually explicit content | Pasal 27 ayat (1) |

---

## Models

### 1. CNN-BiLSTM (Hybrid)

A hybrid architecture combining Convolutional Neural Networks for local feature extraction with Bidirectional LSTM for sequential context modeling.

| Component | Configuration |
|-----------|--------------|
| Embedding | FastText `cc.id.300.vec` (300d, trainable) |
| CNN | 2× Conv1D (64 filters, kernel size 7) + MaxPooling |
| BiLSTM | 64 units, bidirectional |
| Dense | 128 units → 6-class softmax |
| Dropout | 0.2 (embedding), 0.3 (conv/lstm), 0.5 (dense) |
| Optimizer | Adam (lr=1e-3) |
| Loss | Sparse Categorical Crossentropy |
| Max Sequence Length | 256 tokens |
| Early Stopping | Macro F1, patience=5 |

### 2. IndoBERT (Transformer)

Fine-tuned [`indobenchmark/indobert-base-p2`](https://huggingface.co/indobenchmark/indobert-base-p2) with custom special tokens for domain-specific vocabulary (gambling slang, obfuscated usernames, etc.).

| Component | Configuration |
|-----------|--------------|
| Base Model | IndoBERT-base-p2 (110M params) |
| Classifier | Linear (768 → 6) |
| Optimizer | AdamW (lr=1e-5, weight_decay=0.01) |
| Loss | CrossEntropyLoss |
| Batch Size | 16 |
| Max Sequence Length | 256 tokens |
| Early Stopping | Macro F1, patience=3 |

---

## Dataset

Comments were scraped from **YouTube** and **TikTok** using custom scrapers, then labeled using GPT-4o with human validation. The dataset was split into **train/validation/test** sets with multiple undersampling variants to address class imbalance.

| Split | Description |
|-------|-------------|
| `full/` | Complete dataset (original distribution) |
| `undersample_10k/` | Undersampled to ~10K per class |
| `undersample_20k/` | Undersampled to ~20K per class |
| `undersample_30k/` | Undersampled to ~30K per class |

Each split contains `train.csv`, `val.csv`, and `test.csv` with columns:
- `Indeks` — Unique comment identifier
- `normal_teks` — Preprocessed text
- `Label_Validator` — Validated label (0–5)

---

## Preprocessing Pipeline

The preprocessing pipeline is designed to handle the noisy, multilingual, and heavily obfuscated nature of Indonesian social media comments — especially gambling spam that uses Unicode tricks, Cyrillic lookalikes, and mathematical font characters.

```
Raw Comment
    │
    ├── 1. URL Normalization → replace with `link0` token
    ├── 2. Case Folding → lowercase
    ├── 3. Emoji Handling → demojize to `emoji_<name>` tokens
    ├── 4. Invisible Character Removal → ZWJ, word joiners, etc.
    ├── 5. Unicode Normalization → unidecode to ASCII
    ├── 6. Visual Homoglyph Mapping → Cyrillic → Latin (А→A, С→C, etc.)
    ├── 7. Mathematical Font Normalization → 𝗕𝗥𝗢𝗪𝗜𝗡 → BROWIN
    ├── 8. Split-letter Merging → "ᴡ ᴀ ᴋ ᴀ ᴛ ᴏ" → "wakato"
    ├── 9. Mention Handling → @user / @mention_<gambling_username>
    ├── 10. Hashtag Removal
    ├── 11. Slang Normalization → custom Indonesian slang dictionary
    ├── 12. Content Filtering → remove emoji-only, mention-only, etc.
    │
    ▼
Clean Text
```

### External Resources

| Resource | Description |
|----------|-------------|
| `kamus_slang.csv` | Indonesian slang → formal word mapping (~440K entries) |
| `judi_mentions.csv` | Known gambling-related usernames |
| `special_tokens.txt` | Domain-specific tokens added to IndoBERT tokenizer |
| `kbbi.txt` | Indonesian dictionary (KBBI) for word validation |
| `cc.id.300.vec` | Pre-trained FastText word vectors for Indonesian |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/uu-ite.git
cd uu-ite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train CNN-BiLSTM (TensorFlow)
python src/models/cnn_bilstm.py

# Train CNN-BiLSTM (PyTorch variant)
python src/models/cnn_bilstm_pytorch.py

# Train IndoBERT
python src/models/indobert.py
```

> **Note:** Training scripts reference Kaggle dataset paths by default. Update the `pd.read_csv()` paths to point to your local `data/processed/` directory before running.

### Running the Demo

```bash
cd demo
streamlit run app.py
```

The Streamlit demo supports both models and includes:
- Real-time text preprocessing visualization
- Confidence score bar charts
- Relevant UU ITE article citation for detected violations

---

## Demo

The interactive demo application allows users to input any Indonesian text comment and receive:

1. **Violation classification** with confidence scores
2. **Preprocessing visualization** showing each transformation step
3. **Legal reference** — the relevant UU ITE article, its text, and the criminal penalty

Select between **IndoBERT (Transformer)** or **CNN-BiLSTM (Hybrid)** via the sidebar.

---

## Project Structure

```
uu-ite/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── data/
│   ├── raw/                       # Original scraped data (immutable)
│   ├── interim/                   # Intermediate processed datasets
│   ├── processed/                 # Final train/val/test splits
│   │   ├── full/                  # Full dataset
│   │   ├── undersample_10k/       # Undersampled variants
│   │   ├── undersample_20k/
│   │   └── undersample_30k/
│   └── external/                  # Dictionaries, embeddings, resources
│
├── src/
│   ├── scraping/                  # YouTube & TikTok scrapers
│   ├── labeling/                  # GPT-based auto-labeling pipeline
│   ├── preprocessing/             # Text cleaning & normalization
│   ├── models/                    # Model architectures & training
│   │   ├── cnn_bilstm.py          # CNN-BiLSTM (TensorFlow) — Final
│   │   ├── cnn_bilstm_pytorch.py  # CNN-BiLSTM (PyTorch variant)
│   │   ├── indobert.py            # IndoBERT fine-tuning — Final
│   │   └── indobert_baseline.py   # IndoBERT baseline experiment
│   └── utils/                     # Data validation & utility scripts
│
├── notebooks/                     # Jupyter notebooks (EDA, analysis)
├── outputs/                       # Training logs, figures, analysis
│
├── demo/                          # Streamlit demo application
│   ├── app.py                     # Main demo app
│   ├── Cnn-BiLSTM_Model/         # Saved CNN-BiLSTM weights & tokenizer
│   └── Indobert_Model/           # Saved IndoBERT weights & tokenizer
│
└── docs/                          # Documentation & figures
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow/Keras, PyTorch |
| **NLP** | HuggingFace Transformers, IndoBERT, FastText |
| **Data Processing** | Pandas, NumPy, NLTK, Sastrawi |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | scikit-learn (classification report, confusion matrix, F1) |
| **Demo** | Streamlit |
| **Scraping** | Custom scrapers (YouTube API, TikTok) |
| **Labeling** | GPT-4o via g4f (with human validation) |

---

## License

This project is part of an undergraduate thesis (Tugas Akhir S1 Informatika).

---

<p align="center">
  Built with ❤️ for a safer Indonesian digital space
</p>
