# Module 0: Transformers from Scratch — Abstractive Text Summarization

A PyTorch implementation of a sequence-to-sequence **Encoder–Decoder Transformer** built from first principles (random initialization, no pretrained model weights) for **Abstractive Text Summarization**.

---

## 📌 Project Overview

The goal of this project is **Abstractive Text Summarization**: taking large, complex source texts (such as articles, documents, or narrative passages) and condensing them into concise, coherent, and fluent summaries.

Unlike *extractive* summarization—which merely selects and stitches together verbatim sentences from the source—*abstractive* summarization understands the underlying context and generates novel phrasing to capture the core essence of the input text.

```mermaid
flowchart LR
    A[Long / Huge Source Text] --> B[Custom Tokenizer]
    B --> C[Transformer Encoder]
    C -->|Encoder Hidden States| D[Transformer Decoder]
    D -->|Autoregressive Generation| E[Abstractive Summary]
```

---

## 🚀 Key Features & Highlights

- **Built from First Principles**: Implemented in pure PyTorch with zero reliance on pretrained weights or off-the-shelf summarization pipelines.
- **Classic Encoder–Decoder Architecture**:
  - **Encoder**: Bidirectional multi-head self-attention, positional encoding, layer normalization, and feed-forward networks.
  - **Decoder**: Causal masked multi-head self-attention, encoder–decoder cross-attention, and autoregressive next-token prediction.
- **End-to-End Pipeline**:
  - Text normalization and dataset curation.
  - Custom subword / BPE tokenizer trained strictly on target training corpora.
  - Teacher-forced training with cross-entropy loss (padding token masked).
  - Autoregressive inference with greedy decoding.
- **Evaluation & Benchmarking**:
  - Quantitative metrics including ROUGE-1, ROUGE-2, and ROUGE-L.
  - Comparison against baseline extractive heuristics (e.g., lead sentence).
- **Lightweight & Hardware Friendly**:
  - Optimized for local training and inference (e.g., NVIDIA RTX 4060 Laptop GPU / CUDA or CPU).

---

## 🏗️ Repository Structure

```text
Module-0-Transformers-from-Scratch/
├── README.md              # Project documentation and architecture overview
├── DATASET_CARD.md        # Dataset provenance, schema, and licensing notes
├── requirements.txt       # Pinned project dependencies
├── configs/               # Hyperparameter configurations (YAML)
│   └── default.yaml
├── data/
│   ├── raw/               # Raw downloaded datasets (Git ignored)
│   ├── processed/         # Tokenized / prepared datasets (Git ignored)
│   └── smoke_test/        # Minimal fixture for fast testing
├── src/
│   ├── __init__.py
│   ├── model.py           # Encoder, Decoder, Attention, Transformer
│   ├── tokenizer.py       # Custom subword tokenizer implementation/wrapper
│   ├── data.py            # Dataset loading, batching, and collation
│   ├── train.py           # Training loop, checkpointing, and validation
│   ├── evaluate.py        # Metrics computation (ROUGE, loss, perplexity)
│   ├── summarize.py       # CLI inference script for summarizing custom text
│   └── utils.py           # Device handling, seeding, and helper utilities
├── tests/                 # Unit tests (tensor shapes, causal masks, loss)
└── scripts/               # Standalone data prep & tokenizer training scripts
```

---

## ⚙️ Workflow

1. **Input**: A large input document or passage is supplied via text file or CLI prompt.
2. **Tokenization**: The input is mapped to token IDs using the custom-trained vocabulary.
3. **Encoding**: The bidirectional Transformer Encoder computes contextual representations across all input positions.
4. **Decoding**: The Transformer Decoder attends to the encoder representations via cross-attention and autoregressively generates the summary token-by-token until the `<EOS>` (End-of-Sequence) token is reached.
5. **Output**: The decoded tokens are converted back into clean, readable text.

---

## 🛠️ Getting Started

### 1. Environment Setup

Activate your Conda environment and install dependencies:

```bash
conda activate LLM
pip install -r requirements.txt
```

### 2. Verify Environment & Smoke Tests

```bash
pytest tests/
```

### 3. Run Inference / Summarization (Once Checkpoint is Available)

```bash
python -m src.summarize --text "Paste your large text here..."
```

Or pass a file directly:

```bash
python -m src.summarize --file path/to/document.txt
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
