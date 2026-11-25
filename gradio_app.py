import os
import sys
import time
import json
import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from from_scratch.transformer import Seq2SeqTransformer
from pretrained.inference import load_model_b, summarize as bart_summarize
from data.dataset import get_tokenizer_for_model_a
from evaluate.metrics import compute_rouge

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = os.path.join('checkpoints', 'model_a_best.pt')
EVAL_RESULTS = 'eval_results.json'

EXAMPLES = [
    [
        "The Amazon rainforest, also known in English as Amazonia, is a moist broadleaf tropical rainforest "
        "in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses "
        "7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the "
        "rainforest. This region includes territory belonging to nine nations and 3,344 formally "
        "acknowledged indigenous territories. The majority of the forest is contained within Brazil, "
        "with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts "
        "in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.",
        4,
        1.5,
        128
    ],
    [
        "Scientists have developed a new vaccine that shows promising results against multiple strains of "
        "influenza. The vaccine, developed over five years of research, uses mRNA technology similar to "
        "COVID-19 vaccines. In clinical trials involving 10,000 participants across 15 countries, the "
        "vaccine demonstrated 85% efficacy against seasonal flu strains. Researchers believe this could "
        "eliminate the need for annual flu shots if approved. The FDA is expected to begin review of the "
        "vaccine next month, with potential approval coming as early as next year.",
        4,
        1.5,
        128
    ],
    [
        "Apple Inc. announced record quarterly earnings on Tuesday, reporting revenue of $97.3 billion "
        "for the fourth quarter of fiscal year 2024, up 6% year over year. iPhone sales remained the "
        "company's largest segment at $46.2 billion, while services revenue hit an all-time high of "
        "$24.2 billion. CEO Tim Cook highlighted strong performance in emerging markets, particularly "
        "India and Southeast Asia. The company also announced a $110 billion share buyback program, "
        "the largest in Apple history. Shares rose 3% in after-hours trading following the announcement.",
        4,
        1.5,
        128
    ],
]

# --- Global Model Cache ---
_model_a = None
_tokenizer_a = None
_cfg_a = None
_model_b = None
_tokenizer_b = None


def get_models():
    global _model_a, _tokenizer_a, _cfg_a, _model_b, _tokenizer_b

    if _model_b is None:
        print("Loading Model B (BART-large-cnn)...")
        _model_b, _tokenizer_b = load_model_b()

    if _model_a is None and os.path.exists(CHECKPOINT):
        print(f"Loading Model A from {CHECKPOINT}...")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        _cfg_a = ckpt['cfg']
        _tokenizer_a = get_tokenizer_for_model_a()
        _model_a = Seq2SeqTransformer(
            vocab_size=_cfg_a['vocab_size'],
            d_model=_cfg_a['d_model'],
            num_heads=_cfg_a['num_heads'],
            d_ff=_cfg_a['d_ff'],
            num_encoder_layers=_cfg_a['num_encoder_layers'],
            num_decoder_layers=_cfg_a['num_decoder_layers'],
            max_src_len=_cfg_a['max_src_len'],
            max_tgt_len=_cfg_a['max_tgt_len'],
            dropout=0.0,
            pad_idx=_cfg_a['pad_idx'],
            bos_idx=_cfg_a['bos_idx'],
            eos_idx=_cfg_a['eos_idx'],
        ).to(DEVICE)
        _model_a.load_state_dict(ckpt['model_state'])
        _model_a.eval()

    return _model_a, _tokenizer_a, _cfg_a, _model_b, _tokenizer_b


def summarize_dual(text, beam_size=4, rep_penalty=1.5, max_length=128):
    if not text.strip():
        return "Please enter text.", "0.00s", "Please enter text.", "0.00s", "N/A"

    model_a, tokenizer_a, cfg_a, model_b, tokenizer_b = get_models()

    # 1. Model A Inference (From Scratch)
    if model_a is not None:
        t0 = time.time()
        inputs = tokenizer_a(
            text, max_length=cfg_a['max_src_len'], truncation=True,
            padding='max_length', return_tensors='pt'
        )
        src = inputs['input_ids'].to(DEVICE)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            gen_ids = model_a.generate(
                src,
                max_len=max_length,
                beam_size=int(beam_size),
                repetition_penalty=float(rep_penalty)
            )
        lat_a = time.time() - t0
        ids = gen_ids.squeeze().tolist()
        summary_a = tokenizer_a.decode(ids if isinstance(ids, list) else [ids], skip_special_tokens=True)
        summary_a_display = summary_a.strip() if summary_a.strip() else "[Empty generation]"
        lat_a_str = f"⚡ {lat_a:.3f}s"
    else:
        summary_a_display = "⚠️ Model A weights not found in checkpoints/."
        lat_a_str = "N/A"

    # 2. Model B Inference (Pretrained BART)
    summary_b, lat_b = bart_summarize(text, model_b, tokenizer_b, max_summary=max_length, beam_size=int(beam_size))
    lat_b_str = f"⚡ {lat_b:.3f}s"

    # 3. Dynamic ROUGE Agreement (A vs B)
    if model_a is not None and summary_a.strip():
        rouge = compute_rouge([summary_a], [summary_b])
        agreement_md = f"**Cross-Model Agreement:** ROUGE-1 = `{rouge['rouge1']:.3f}` | ROUGE-L = `{rouge['rougeL']:.3f}`"
    else:
        agreement_md = "Cross-Model Agreement: N/A"

    return summary_a_display, lat_a_str, summary_b, lat_b_str, agreement_md


def get_dashboard_data():
    if not os.path.exists(EVAL_RESULTS):
        return None, None, "Evaluation results file (`eval_results.json`) not generated yet."

    with open(EVAL_RESULTS, 'r', encoding='utf-8') as f:
        res = json.load(f)

    df = pd.DataFrame({
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Avg Latency (s)", "Parameters"],
        "Model A (Scratch)": [
            f"{res['model_a']['rouge']['rouge1']:.4f}",
            f"{res['model_a']['rouge']['rouge2']:.4f}",
            f"{res['model_a']['rouge']['rougeL']:.4f}",
            f"{res['model_a']['latency']:.4f}",
            f"{res['model_a']['params']:,}",
        ],
        "Model B (BART-L)": [
            f"{res['model_b']['rouge']['rouge1']:.4f}",
            f"{res['model_b']['rouge']['rouge2']:.4f}",
            f"{res['model_b']['rouge']['rougeL']:.4f}",
            f"{res['model_b']['latency']:.4f}",
            f"{res['model_b']['params']:,}",
        ]
    })

    # Generate Chart
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    a_vals = [res['model_a']['rouge']['rouge1'], res['model_a']['rouge']['rouge2'], res['model_a']['rouge']['rougeL']]
    b_vals = [res['model_b']['rouge']['rouge1'], res['model_b']['rouge']['rouge2'], res['model_b']['rouge']['rougeL']]
    
    x = range(len(metrics))
    ax.bar([p - 0.18 for p in x], a_vals, width=0.36, label='Model A (Scratch)', color='#4361ee')
    ax.bar([p + 0.18 for p in x], b_vals, width=0.36, label='Model B (BART-L)', color='#f72585')
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("F1 Score")
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Sample predictions formatted
    samples_md = "### Held-Out Test Samples\n\n"
    refs = res['model_a'].get('references', [])
    preds_a = res['model_a'].get('predictions', [])
    preds_b = res['model_b'].get('predictions', [])

    for i in range(min(3, len(refs))):
        samples_md += f"**Sample {i+1}**\n"
        samples_md += f"> **Reference:** {refs[i]}\n\n"
        samples_md += f"- **Model A:** {preds_a[i]}\n"
        samples_md += f"- **Model B:** {preds_b[i]}\n\n---\n"

    return df, fig, samples_md


# --- Gradio UI Theme & Layout ---
custom_css = """
.container { max-width: 1200px; margin: auto; }
.card-header { font-weight: 700; font-size: 1.1rem; margin-bottom: 0.3rem; }
.latency-tag { font-size: 0.85rem; color: #6c757d; font-family: monospace; }
.summary-box textarea { font-size: 0.95rem !important; line-height: 1.5 !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="pink",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="Transformer Summarization Benchmarking") as demo:
    gr.Markdown(
        """
        # ⚡ Neural Text Summarization: Scratch vs. Pretrained Benchmark
        ### Compare an **Encoder-Decoder Transformer from scratch** (Pre-LN, 16k BPE) against **facebook/bart-large-cnn**
        """
    )

    with gr.Tabs():
        # --- Tab 1: Live Interactive Inference ---
        with gr.TabItem("🚀 Live Comparison"):
            with gr.Row():
                with gr.Column(scale=5):
                    input_text = gr.Textbox(
                        label="Source Article",
                        placeholder="Paste article text here...",
                        lines=10,
                        max_lines=15
                    )
                    with gr.Accordion("⚙️ Decoding Parameters", open=False):
                        with gr.Row():
                            beam_size = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Beam Size")
                            rep_penalty = gr.Slider(minimum=1.0, maximum=2.5, value=1.5, step=0.1, label="Repetition Penalty (Model A)")
                            max_len = gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Max Summary Length")

                    run_btn = gr.Button("🔥 Run Both Models Simultaneously", variant="primary", size="lg")

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[input_text, beam_size, rep_penalty, max_len],
                        label="Preset Benchmark Articles"
                    )

                with gr.Column(scale=5):
                    agreement_badge = gr.Markdown("Cross-Model Agreement: *Run prediction to evaluate*")
                    
                    with gr.Group():
                        with gr.Row():
                            gr.Markdown("### 🛠️ Model A — From Scratch\n*Seq2Seq Pre-LN Transformer (95M)*")
                            latency_a = gr.Markdown("⚡ --")
                        summary_a = gr.Textbox(
                            label="",
                            placeholder="Scratch model summary will appear here...",
                            lines=4,
                            interactive=False,
                            elem_classes="summary-box"
                        )

                    with gr.Group():
                        with gr.Row():
                            gr.Markdown("### 🏆 Model B — Pretrained\n*facebook/bart-large-cnn (406M)*")
                            latency_b = gr.Markdown("⚡ --")
                        summary_b = gr.Textbox(
                            label="",
                            placeholder="BART summary will appear here...",
                            lines=4,
                            interactive=False,
                            elem_classes="summary-box"
                        )

            run_btn.click(
                fn=summarize_dual,
                inputs=[input_text, beam_size, rep_penalty, max_len],
                outputs=[summary_a, latency_a, summary_b, latency_b, agreement_badge]
            )

        # --- Tab 2: Test Set Evaluation Metrics ---
        with gr.TabItem("📊 Evaluation Dashboard"):
            gr.Markdown("### Full Test Set Benchmark (`CNN/DailyMail`)")
            df_data, chart_data, samples_data = get_dashboard_data()

            if df_data is not None:
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("#### Comparative Quality & Efficiency")
                        gr.Dataframe(df_data, interactive=False)
                    with gr.Column(scale=5):
                        gr.Markdown("#### ROUGE Scores Comparison")
                        gr.Plot(chart_data)

                gr.Markdown("---")
                gr.Markdown(samples_data)
            else:
                gr.Markdown("⚠️ No evaluation data found. Run `python evaluate/run_eval.py` to populate test metrics.")

        # --- Tab 3: Model Architecture Overview ---
        with gr.TabItem("ℹ️ Architectural Details"):
            gr.Markdown(
                """
                ### Architectural Comparison
                | Feature | Model A (From Scratch) | Model B (Pretrained) |
                | :--- | :--- | :--- |
                | **Base Checkpoint** | None (Random Xavier initialization) | `facebook/bart-large-cnn` |
                | **Architecture** | 6-Layer Encoder / 6-Layer Decoder (Pre-LN) | 12-Layer Encoder / 12-Layer Decoder |
                | **Attention** | Self-written Scaled Dot-Product & MHA (8 heads) | Standard Hugging Face BART MHA (16 heads) |
                | **Tokenizer** | Custom Byte-Level BPE (16,384 vocabulary) | BART BPE Tokenizer (50,265 vocabulary) |
                | **Hidden Dimension ($d_{\\text{model}}$)** | 512 ($d_{\\text{ff}} = 2048$) | 1024 ($d_{\\text{ff}} = 4096$) |
                | **Total Parameters** | **95,634,521** (~95.6M) | **406,290,432** (~406.3M) |
                | **Training Objective** | Cross-Entropy + Label Smoothing (0.1) | Masked Language Model + Sentence Permutation |
                """
            )

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css, server_name="127.0.0.1", server_port=7860, share=False)
