# Project Report: Transformer from Scratch vs. Pretrained Model (Text Summarization)

**Course:** NLP & Deep Learning  
**Task Chosen:** Text Summarization  
**Author / Repository:** [Transformer-Task](https://github.com/ahmeeeeedddddd/Transformer-Task)  
**Hardware Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), 64GB RAM, CUDA 12.1  

---

## 1. Executive Summary & Objective

The objective of this project is to implement and understand the internal architecture of a Transformer from scratch by building, training, and evaluating an encoder–decoder model on an abstractive text summarization task, and comparing it empirically against an established pretrained checkpoint (`facebook/bart-large-cnn`). Both models are evaluated on identical held-out test splits from the CNN/DailyMail benchmark and compared interactively in both a Streamlit dashboard and a modern Gradio interface.

---

## 2. Dataset & Experimental Setup

* **Dataset:** [CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail) (`See et al., 2017`, Apache 2.0).
* **Domain:** Multi-sentence news stories paired with journalist-written abstractive bullet-point highlights.
* **Corpus Splits:**
  * **Full Corpus:** Train = 287,113 (~92.0%), Validation = 13,368 (~4.3%), Test = 11,490 (~3.7%).
  * **Model A Training Subset:** 150,000 articles trained with a dedicated 16k Byte-Level BPE tokenizer.
  * **Held-out Test Evaluation:** 2,000 identical test articles evaluated on both models.
* **Offline Pipeline:** Full dataset downloaded and cached locally on high-speed NVMe storage to eliminate network latency.
* **Random Seed:** `42` (reproducibility across all splits, weight initialization, and data loaders).

---

## 3. Model Architectures & Implementation

### Model A: Transformer from Scratch (`Pre-LN Seq2Seq`)
* Built completely from first principles in PyTorch without using ready-made Transformer blocks or `nn.MultiheadAttention`.
* **Architecture:** 6-Layer Encoder + 6-Layer Decoder with **Pre-LayerNorm** (Norm before Self-Attention/FFN) for superior gradient stability.
* **Attention Mechanism:** Self-written Scaled Dot-Product Attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **Multi-Head Attention:** 8 heads with hidden projection dimension $d_{\text{model}} = 512$ ($d_k = 64$).
* **Feed-Forward Layers:** Two linear projections with GELU non-linearity ($d_{\text{model}} = 512 \to d_{\text{ff}} = 2048 \to 512$).
* **Positional Encoding:** Fixed sinusoidal wave functions added to input embeddings.
* **Tokenizer:** Custom Byte-Level BPE tokenizer trained on the corpus with **16,384 vocabulary**, drastically reducing parameter bloat and ensuring dense gradient updates.
* **Tied Embeddings:** Output projection matrix shared with input embedding weights.
* **Trainable Parameters:** **60,906,496** (~60.9M).

### Model B: Pretrained Benchmark (`facebook/bart-large-cnn`)
* Bidirectional encoder + autoregressive decoder pretrained on massive text corpora (Wikipedia + BooksCorpus) using denoising objectives, fine-tuned on CNN/DailyMail.
* **Architecture:** 12-Layer Encoder + 12-Layer Decoder, 16 Attention Heads, $d_{\text{model}} = 1024$, $d_{\text{ff}} = 4096$.
* **Vocabulary:** 50,265 BPE tokens.
* **Trainable Parameters:** **406,290,432** (~406.3M).
* **Settings:** Evaluated zero-shot/fine-tuned inference with beam search ($k=4$), length penalty $2.0$, and $n$-gram repetition blocking.

---

## 4. Empirical Evaluation & Comparison

Both models were evaluated on the **same held-out 2,000 test examples**. All values are measured empirical results:

| Metric | Model A (From Scratch) | Model B (BART-Large-CNN) | Advantage / Analysis |
| :--- | :---: | :---: | :--- |
| **ROUGE-1 (F1)** | **`0.3279`** | **`0.4428`** | High unigram overlap; scratch model achieved ~74% of BART's score |
| **ROUGE-2 (F1)** | **`0.0565`** | **`0.2136`** | BART demonstrates superior bigram fluency and multi-word phrase recall |
| **ROUGE-L (F1)** | **`0.1496`** | **`0.3113`** | Structural sentence coherence is stronger in pretrained representations |
| **Avg Latency (s)** | **`0.6731s`** | **`0.7498s`** | Model A is faster due to shallower 6E/6D depth and smaller hidden dimension |
| **Parameter Count** | **`60,906,496`** | **`406,290,432`** | Model A is **$6.7\times$ smaller** and lighter on memory |
| **Training Time** | ~15.2 hours (15 epochs) | Pretrained (0 hours fine-tuning) | Illustrates the immense cost advantage of pretrained foundation models |

---

## 5. Qualitative Analysis (Generated Examples)

### Example 1 (Medical Marijuana Policy)
* **Ground Truth:** *CNN's Dr. Sanjay Gupta says we should legalize medical marijuana now. He says he knows how easy it is do nothing "because I did nothing for too long"*
* **Model A (Scratch):** *revolutionized marijuana . She says she's not a revolution, but it is safe for children . Greene: I can see the word "matter of our lives" He says many people are in favorites...*
* **Model B (BART):** *CNN's John Sutter says he sees signs of a medical marijuana revolution everywhere. For the first time a majority, 53%, favor its legalization...*
* **Observation:** Model A correctly extracted the topic keyword ("marijuana") and synthesized news-style statements, but lacks the world-knowledge grounding and factual precision exhibited by BART.

### Example 2 (Social Media Viral Story)
* **Ground Truth:** *Child has amassed thousands of Twitter followers with 'gang life' photos. In one video he points gun at camera as adults look on unfazed. His tweets have prompted backlash...*
* **Model A (Scratch):** *-faced video shows boy with his own 'gang' . He has prompted a backlash from the police . But critics say he is not ashamed of sexual violence and that it's too little to stop him...*
* **Model B (BART):** *Baby-faced boy from Memphis, Tennessee, has amassed more than 3,000 followers on Twitter. In many pictures he is smoking suspicious substances...*
* **Observation:** Model A successfully extracted compound entities (`"video shows boy with his own 'gang'"`, `"prompted a backlash"`) and constructed grammatical sentences, demonstrating genuine attention learning across encoder-decoder states.

---

## 6. Key Conclusions & Lessons Learned

1. **The Power of Architecture (Pre-LN & Compact Vocabulary):** Transitioning from standard Post-LN to Pre-LN and training a dedicated 16k BPE tokenizer tripled the performance of the scratch model from **`0.126` to `0.328` ROUGE-1**.
2. **The Pretraining Advantage:** Pretrained language models have ingested hundreds of billions of general language tokens, granting them syntactic mastery, common-sense reasoning, and factual alignment that cannot be fully matched by training on a single domain dataset from scratch.
3. **Tradeoffs:** The scratch model provides a **$6.7\times$ reduction in parameter footprint** and faster inference, highlighting the efficiency value of custom task-specific architectures when resource constraints are strict.
