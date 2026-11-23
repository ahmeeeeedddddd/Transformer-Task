# Transformer Summarization: From Scratch vs BART-large-cnn

**Task:** Text Summarization  
**Dataset:** [CNN/DailyMail 3.0.0](https://huggingface.co/datasets/cnn_dailymail) — `See et al., 2017` — Apache 2.0  
**Splits used:** 50k train / 2k val / 2k test (subset of full dataset)  
**Random seed:** 42

---

## Project Structure

```
├── model_a/
│   ├── transformer.py     # Encoder-decoder Transformer from scratch
│   └── train.py           # Training script
├── model_b/
│   └── inference.py       # facebook/bart-large-cnn inference
├── data/
│   └── dataset.py         # CNN/DailyMail loading + Dataset class
├── evaluate/
│   ├── metrics.py         # ROUGE computation + latency utils
│   └── run_eval.py        # Full evaluation on held-out test set
├── checkpoints/           # Saved Model A weights
├── app.py                 # Streamlit demo
├── requirements.txt
└── README.md
```

---

## Models

### Model A — Transformer from Scratch
- Architecture: Encoder-Decoder
- d_model=512, 8 heads, 6 encoder + 6 decoder layers, d_ff=2048
- Sinusoidal positional encoding, residual connections, LayerNorm
- Weight-tied input/output embeddings
- Trained with label smoothing (0.1), mixed precision (fp16), AdamW, warmup-cosine LR
- No pretrained weights, no `nn.MultiheadAttention`, no Hugging Face Transformer blocks

### Model B — Pretrained (not fine-tuned further)
- Checkpoint: `facebook/bart-large-cnn`
- ~400M parameters, pre-fine-tuned on CNN/DailyMail
- Run inference only; no additional fine-tuning

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train Model A
```bash
python -m model_a.train
```
Trains for 15 epochs on 50k CNN/DailyMail articles. Saves best checkpoint to `checkpoints/model_a_best.pt`.

### 2. Evaluate both models on the test set
```bash
python evaluate/run_eval.py
```
Outputs ROUGE-1, ROUGE-2, ROUGE-L, latency, parameter count, and 3 sample predictions. Saves `eval_results.json`.

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| ROUGE-1 | Unigram overlap between generated and reference summary |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence |
| Avg latency | Per-sample inference time (seconds) |
| Parameters | Trainable parameter count |

---

## Hardware Used
- GPU: NVIDIA RTX 4060 (8 GB VRAM)
- RAM: 64 GB
- Mixed precision: fp16 (torch.cuda.amp)
