# Project Brief for Antigravity IDE: Abstractive Text Summarization Transformer

## 1. Mission

Build a complete, runnable, GitHub-ready project that trains a **small encoder–decoder Transformer from random initialization** for abstractive text summarization.

The goal is to implement and run the project quickly, not to create a coding course. Generate clean, maintainable code, install/run instructions, tests, training logs, a saved checkpoint, and a simple demo.

**Do not use pretrained model weights or a pretrained summarization pipeline.** Implement the Transformer architecture in PyTorch, including encoder self-attention, decoder causal self-attention, encoder–decoder cross-attention, feed-forward networks, residual connections, normalization, positional embeddings/encoding, output projection, and autoregressive decoding. Standard PyTorch tensor operations and utilities are allowed.

## 2. My hardware and constraints

- Laptop CPU: Intel Core i7
- System RAM: 16 GB
- GPU: NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM
- Primary goal: finish a working training run locally
- Keep the initial model and data small enough for this hardware.
- Prefer Windows-compatible instructions unless the detected environment indicates otherwise.
- Detect whether CUDA is available and print the selected device at startup.
- Do not require Google Colab or cloud GPU.
- Avoid unnecessarily complicated frameworks and dependencies.

Start with conservative settings and allow configuration overrides:
- `d_model`: 128 or 256
- Encoder layers: 2–3
- Decoder layers: 2–3
- Attention heads: 4 (must divide `d_model`)
- Feed-forward dimension: 512–1024
- Maximum source length: 128 tokens
- Maximum target length: 64 tokens
- Batch size: start at 8 or 16; increase only if memory permits
- Use gradient clipping, AdamW, padding masks, and gradient-safe training.
- Start with FP32 for simplicity; optionally support CUDA mixed precision if tested.
- Set and record random seeds.
- Include a small quick-run mode for smoke testing and a configurable full training mode.

Do not promise a specific training duration before measuring throughput on this machine.

## 3. Dataset and licensing requirements — important

First inspect the dataset card, license field, data files, feature schema, and actual examples for:

**Candidate dataset:** https://huggingface.co/datasets/abhiprd2000/nlp-benchmark-suite

The listing has been described as Apache-2.0, but do not assume that this automatically clears the rights of every underlying source or every example. Verify that:
1. A summarization subset actually exists.
2. It contains source documents and corresponding human-written summaries in usable fields.
3. Its dataset card and upstream provenance support redistribution and intended use.
4. There are no non-commercial-only restrictions or unclear third-party rights for the selected data.

**If the candidate dataset does not meet these checks, do not silently use it.** Stop the dataset selection step, document what failed, and choose one of these paths:
- A small summarization dataset whose actual content and license permit the intended use and redistribution, after verifying its source and terms; or
- A clearly labeled educational fallback using verified public-domain source texts and programmatically generated training pairs.

For the fallback, make the pair-generation process transparent and deterministic. Do not claim these are human-written reference summaries. Clearly label results as synthetic/pseudo-summary training and explain the quality limitations. Do not scrape arbitrary websites or include copyrighted content with unclear terms.

Keep the dataset download/preparation script separate from training. Store raw and processed data under `data/`, keep large/raw data out of Git by default, and include exact download/preparation instructions and source/license attribution in `DATASET_CARD.md`. Do not commit a dataset until its license and redistribution terms have been verified. Include a small, clearly licensed or self-generated smoke-test fixture so tests run without downloading a large corpus.

## 4. Model requirements

Implement a conventional encoder–decoder Transformer in PyTorch, not a decoder-only GPT and not a wrapper around a pretrained model.

### Encoder
- Token embeddings and positional encoding/embeddings
- Multi-head bidirectional self-attention
- Padding mask for source tokens
- Position-wise feed-forward network
- Residual connections and layer normalization
- Configurable number of encoder layers

### Decoder
- Target token embeddings and positional encoding/embeddings
- Causal masked multi-head self-attention
- Cross-attention where decoder queries attend to encoder outputs as keys/values
- Target padding mask and source padding mask in the appropriate attention operations
- Position-wise feed-forward network
- Residual connections and layer normalization
- Configurable number of decoder layers

### Output and training
- Linear projection to target vocabulary logits
- Teacher-forced training
- Shift target sequence correctly:
  - decoder input: `<BOS> summary tokens ...`
  - labels: `summary tokens ... <EOS>`
- Cross-entropy loss that ignores `<PAD>` labels
- Correct causal mask: no decoder position can attend to future target positions
- Correct handling of `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>`
- Autoregressive inference that starts with `<BOS>` and stops at `<EOS>` or maximum length
- Greedy decoding first; optional beam search only after greedy decoding works

Use a simple tokenizer that can be trained from the selected data, such as SentencePiece BPE/unigram or a compact word/subword tokenizer. Do not download pretrained tokenizer assets. If using SentencePiece, include the tokenizer-training script and vocabulary-size configuration. Keep tokenizer training reproducible.

## 5. Repository structure

Create a simple repository with at least:

```text
transformer-summarizer/
├── README.md
├── DATASET_CARD.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                 # ignored by Git
│   ├── processed/           # ignored by Git
│   └── smoke_test/          # tiny permissible test fixture
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── summarize.py
│   └── utils.py
├── tests/
│   ├── test_shapes.py
│   ├── test_masks.py
│   ├── test_training_step.py
│   └── test_generation.py
├── scripts/
│   ├── prepare_data.py
│   └── train_tokenizer.py
├── artifacts/               # checkpoints ignored by Git by default
└── outputs/                 # logs/predictions ignored by Git by default
```

Adjust only if there is a clear simplification. Keep model, data, training, evaluation, and inference responsibilities separated.

## 6. Required implementation workflow

Implement and verify in this order. Do not generate a huge untested code dump and assume it works.

### Step 1 — Environment and data audit
- Check Python version, PyTorch installation, CUDA availability, and GPU name/memory.
- Inspect the proposed dataset's license, schema, examples, and size.
- Produce `DATASET_CARD.md` and explain the selected data path.
- If dataset licensing or schema is unclear, report that and use the documented fallback only after stating its limitations.

### Step 2 — Data pipeline and tokenizer
- Download or prepare the permitted dataset.
- Normalize text conservatively without destroying punctuation.
- Remove empty or malformed pairs and deduplicate exact pairs.
- Split train/validation/test deterministically, with no duplicate pair leakage.
- Train tokenizer only on training data.
- Add special tokens and implement truncation/padding and attention masks.
- Print several decoded examples to verify alignment.
- Never train on test data.

### Step 3 — Model and unit tests
- Implement the full encoder–decoder Transformer.
- Print and document tensor shapes at each major stage.
- Test source and target masks, including that future target positions are blocked.
- Verify logits shape is `[batch, target_length, target_vocab_size]`.
- Verify loss is finite and gradients reach both encoder and decoder parameters.
- Include an overfit-one-batch sanity test: the model should be able to reduce loss on a tiny batch. This is a debugging check, not a claim of generalization.

### Step 4 — Training
- Train from random initialization.
- Log training loss, validation loss, epoch/step, elapsed time, learning rate, and peak CUDA memory when available.
- Save checkpoints and configuration, tokenizer assets, optimizer state, and random seed.
- Save the best checkpoint based on validation loss.
- Support resuming training.
- Include early stopping or a configurable maximum number of epochs/steps.
- Use gradient clipping.
- Avoid claiming success if loss is NaN, gradients are missing, or validation does not run.

### Step 5 — Evaluation
Evaluate on a held-out test set and report:
- Test loss/perplexity where mathematically appropriate
- ROUGE-1, ROUGE-2, and ROUGE-L using a documented implementation
- Inference latency per example
- Peak GPU memory, parameter count, and tokens/examples processed per second
- Several source/reference/generated examples, including failures

Do not treat ROUGE as a complete measure of summary quality. If data is synthetic, explicitly say the metrics are only meaningful relative to that synthetic setup. Compare against at least one simple baseline, such as a lead-sentence or extractive baseline, when the dataset supports it.

### Step 6 — Inference demo
Provide a command-line interface such as:

```bash
python -m src.summarize --text "Paste a short document here"
```

Also support reading a UTF-8 text file. Load the saved tokenizer and trained checkpoint, run greedy autoregressive decoding, and print the summary. Provide useful errors if artifacts are missing.

Optional: add a lightweight Streamlit demo only after the CLI works; do not make it a prerequisite for training.

### Step 7 — Reproducibility and GitHub preparation
- Pin or bound tested dependency versions in `requirements.txt`.
- Provide exact setup, dataset preparation, training, evaluation, testing, and inference commands.
- Include a clear README architecture diagram (Mermaid is fine), tensor-shape explanation, model configuration, training curves, benchmark table, sample outputs, limitations, and known issues.
- Include dataset source/license attribution and model/dataset limitations.
- Add a suitable repository license for my original code, such as MIT, only after confirming dependencies and data terms do not conflict. A code license does not relicense third-party data.
- Ignore raw datasets, processed datasets, checkpoints, caches, logs, virtual environments, and secrets. Do not put private data or credentials in Git.
- Include a GitHub-ready commit checklist, but do not create or push a repository unless explicitly asked.

## 7. Acceptance criteria

The project is not done until all of these are true:

1. A clean environment can install the dependencies using the documented steps.
2. The data audit records the selected dataset, exact license, source, schema, and limitations.
3. A quick-run training job completes locally on CPU or CUDA with the smoke-test data.
4. A full small training run can execute on the laptop GPU within available memory.
5. The model is initialized from random weights; no pretrained Transformer weights are loaded.
6. Encoder self-attention, decoder causal self-attention, and decoder cross-attention are all implemented and exercised.
7. At least four automated tests pass: shapes, causal masking, one backward/update step, and generation/EOS handling.
8. A checkpoint can be loaded in a fresh process and generate a summary from text input.
9. Evaluation outputs and example predictions are saved.
10. README documents the real results and limitations. Do not fabricate metrics, claim tests passed without running them, or claim the model produces high-quality summaries without evidence.

## 8. Engineering rules

- Prefer readable, modular PyTorch over clever abstractions.
- Use a fixed seed and record software/hardware configuration.
- Keep paths and hyperparameters configurable; do not hard-code machine-specific absolute paths.
- Use a small initial run to validate the entire pipeline before scaling up.
- Check GPU memory and reduce batch size or sequence lengths if CUDA out-of-memory occurs.
- Do not add pretrained model weights, hosted inference APIs, or paid services.
- Do not make unsupported claims about dataset licensing or model quality.
- Make a distinction in the README between an educational from-scratch model and production-grade summarization systems.

## 9. Expected final response from Antigravity

After implementation, report:
1. Files created and each file's purpose.
2. Dataset selected, exact license evidence, and any unresolved rights caveats.
3. Environment and GPU detection results.
4. Commands to install, prepare data, train, evaluate, run tests, and summarize text.
5. Whether each command was actually run and its result.
6. Training configuration, parameter count, runtime, peak VRAM, train/validation curves, and test metrics if measured.
7. Example outputs and known failures.
8. Any remaining tasks before publishing to GitHub.

Start by auditing the environment and dataset. Then implement the smallest end-to-end version, test it, and iterate. Prioritize a working, honest, reproducible project over a large amount of generated code.




## 10. Transformer Project — Step-by-Step Execution Workflow

1. **Setup Environment**
   - Install required libraries and dependencies.
   - Configure the environment and GPU/device if available.

2. **Import Required Packages**
   - Import libraries such as PyTorch, NumPy, pandas, and other required packages.

3. **Load the Dataset**
   - Load the dataset from the relevant source.
   - Inspect the dataset structure, size, and sample entries.

4. **Preprocess the Dataset**
   - Clean and normalize the data if necessary.
   - Handle missing or invalid values.
   - Split the dataset into training, validation, and test sets.
   - Prepare the data for tokenization.

5. **Tokenization and Numerical Encoding**
   - Create or load a tokenizer.
   - Build or load the vocabulary if required.
   - Convert tokens into numerical token IDs.
   - Add special tokens where necessary.
   - Apply padding, truncation, and sequence-length handling as required.

6. **Create Dataset and DataLoader**
   - Create a custom Dataset class or use an appropriate existing dataset implementation.
   - Define how individual training examples and their targets are retrieved.
   - Create DataLoaders for training, validation, and testing.
   - Configure batch size, shuffling, and other loading parameters.
   - Verify batch shapes and tensor contents.

7. **Define Model Configuration and Hyperparameters**
   - Set the vocabulary size.
   - Define the embedding dimension (d_model).
   - Set the number of attention heads.
   - Set the number of Transformer layers.
   - Define the maximum sequence length or context window.
   - Configure dropout and other model parameters.

8. **Build the Transformer Architecture**
   - Implement token embeddings.
   - Implement positional embeddings or positional encoding.
   - Implement scaled dot-product attention.
   - Implement multi-head attention.
   - Implement feed-forward neural networks.
   - Add residual connections and layer normalization.
   - Construct Transformer blocks.
   - Stack the required number of Transformer blocks.
   - Add the final normalization and output layer.
   - Implement the appropriate attention masks for the chosen architecture.

9. **Initialize the Model**
   - Instantiate the Transformer model using the defined configuration.
   - Move the model to the appropriate device (CPU or GPU).
   - Verify the model architecture and output dimensions.
   - Run a test forward pass to check tensor shapes.

10. **Define the Loss Function and Optimizer**
    - Select an appropriate loss function, such as cross-entropy loss for token prediction.
    - Configure an optimizer, such as AdamW.
    - Set the learning rate and other optimization parameters.
    - Configure learning-rate scheduling or gradient clipping if needed.

11. **Implement the Training Loop**
    - Set the model to training mode.
    - Iterate over batches from the training DataLoader.
    - Move input and target tensors to the appropriate device.
    - Perform the forward pass.
    - Calculate the loss.
    - Clear the previous gradients.
    - Perform backpropagation.
    - Update model parameters using the optimizer.
    - Track and log training loss.
    - Repeat for the required number of epochs.

12. **Implement Validation**
    - Set the model to evaluation mode.
    - Disable gradient calculations during validation.
    - Evaluate the model using the validation DataLoader.
    - Calculate validation loss and relevant evaluation metrics.
    - Monitor overfitting and adjust hyperparameters if necessary.
    - Save the best-performing model checkpoint when appropriate.

13. **Evaluate the Model**
    - Evaluate the trained model on the test dataset.
    - Calculate task-specific metrics.
    - Analyze model performance and prediction quality.
    - Compare training, validation, and test results.

14. **Test Inference**
    - Prepare new, unseen input data.
    - Apply the same preprocessing and tokenization pipeline.
    - Generate predictions or text using the trained model.
    - Convert predicted token IDs back into readable output where applicable.
    - Inspect the quality and correctness of the outputs.

15. **Save the Model and Artifacts**
    - Save the trained model weights.
    - Save the model configuration and hyperparameters.
    - Save the tokenizer and vocabulary when applicable.
    - Save optimizer and scheduler states if training may be resumed.
    - Save the best checkpoint and relevant evaluation results.

16. **Final Testing and Documentation**
    - Load the saved model and verify that it works correctly.
    - Test it on representative unseen examples.
    - Document the architecture, training settings, evaluation results, and limitations.
    - Organize the code and artifacts for reproducibility.

**Important:** The exact implementation depends on the Transformer architecture and task. A decoder-only language model, an encoder-decoder translation model, and an encoder-only model require different input-target structures, attention masks, and training objectives. Ensure the Dataset, DataLoader, model outputs, and loss function are consistent with the chosen task.

**Important:** Do not run code yourself, just give the user all code and commands inline. They will copy-paste the code manually into each cell and run it on their own. You do not have to run the code and commands automatically (backend or frontend) for them. Just make sure the code is correct and will work when copied and pasted.
