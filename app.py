import os
import sys
import time
import json
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from from_scratch.transformer import Seq2SeqTransformer
from pretrained.inference import load_model_b, summarize as bart_summarize
from evaluate.metrics import compute_rouge
from data.dataset import get_tokenizer_for_model_a

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = os.path.join('checkpoints', 'model_a_best.pt')
EVAL_RESULTS = 'eval_results.json'

EXAMPLES = [
    (
        "The Amazon rainforest, also known in English as Amazonia, is a moist broadleaf tropical rainforest "
        "in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses "
        "7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the "
        "rainforest. This region includes territory belonging to nine nations and 3,344 formally "
        "acknowledged indigenous territories. The majority of the forest is contained within Brazil, "
        "with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts "
        "in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela."
    ),
    (
        "Scientists have developed a new vaccine that shows promising results against multiple strains of "
        "influenza. The vaccine, developed over five years of research, uses mRNA technology similar to "
        "COVID-19 vaccines. In clinical trials involving 10,000 participants across 15 countries, the "
        "vaccine demonstrated 85% efficacy against seasonal flu strains. Researchers believe this could "
        "eliminate the need for annual flu shots if approved. The FDA is expected to begin review of the "
        "vaccine next month, with potential approval coming as early as next year."
    ),
    (
        "Apple Inc. announced record quarterly earnings on Tuesday, reporting revenue of $97.3 billion "
        "for the fourth quarter of fiscal year 2024, up 6% year over year. iPhone sales remained the "
        "company's largest segment at $46.2 billion, while services revenue hit an all-time high of "
        "$24.2 billion. CEO Tim Cook highlighted strong performance in emerging markets, particularly "
        "India and Southeast Asia. The company also announced a $110 billion share buyback program, "
        "the largest in Apple history. Shares rose 3% in after-hours trading following the announcement."
    ),
]


@st.cache_resource(show_spinner="Loading Model A (Transformer from Scratch)...")
def get_model_a():
    if not os.path.exists(CHECKPOINT):
        return None, None, None
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    cfg  = ckpt['cfg']
    tokenizer = get_tokenizer_for_model_a()
    model = Seq2SeqTransformer(
        vocab_size=cfg['vocab_size'], d_model=cfg['d_model'], num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'], num_encoder_layers=cfg['num_encoder_layers'],
        num_decoder_layers=cfg['num_decoder_layers'], max_src_len=cfg['max_src_len'],
        max_tgt_len=cfg['max_tgt_len'], dropout=0.0,
        pad_idx=cfg['pad_idx'], bos_idx=cfg['bos_idx'], eos_idx=cfg['eos_idx'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, tokenizer, cfg


@st.cache_resource(show_spinner="Loading Model B (BART-large-cnn)...")
def get_model_b():
    return load_model_b()


def summarize_a(text, model, tokenizer, cfg):
    inputs = tokenizer(
        text, max_length=cfg['max_src_len'], truncation=True,
        padding='max_length', return_tensors='pt',
    )
    src = inputs['input_ids'].to(DEVICE)
    t0  = time.time()
    with torch.no_grad():
        gen = model.generate(
            src, 
            max_len=cfg['max_tgt_len'], 
            beam_size=cfg.get('beam_size', 4),
            repetition_penalty=cfg.get('repetition_penalty', 1.5)
        )
    latency = time.time() - t0
    ids  = gen.squeeze().tolist()
    text = tokenizer.decode(ids if isinstance(ids, list) else [ids], skip_special_tokens=True)
    return text, latency


def main():
    st.set_page_config(page_title="Transformer Summarization", layout="wide", page_icon="📝")

    st.title("📝 Text Summarization: Scratch Transformer vs BART-large-cnn")
    st.markdown(
        "Compare a **Transformer built from scratch** against **facebook/bart-large-cnn** "
        "on the CNN/DailyMail summarization task."
    )

    model_a, tokenizer_a, cfg_a = get_model_a()
    model_b, tokenizer_b        = get_model_b()

    if model_a is None:
        st.warning(
            "⚠️ Model A checkpoint not found. Train Model A first:\n"
            "```bash\npython -m model_a.train\n```"
        )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        beam_size = st.slider("Beam size (Model A)", 1, 8, 4)
        rep_penalty = st.slider("Repetition penalty (Model A)", 1.0, 2.5, 1.5, step=0.1)
        if cfg_a:
            cfg_a['beam_size'] = beam_size
            cfg_a['repetition_penalty'] = rep_penalty
        st.markdown("---")
        st.markdown("**Model A:** Encoder-Decoder Transformer (from scratch)")
        if cfg_a:
            st.markdown(f"- d_model: {cfg_a['d_model']}")
            st.markdown(f"- Layers: {cfg_a['num_encoder_layers']}E / {cfg_a['num_decoder_layers']}D")
            st.markdown(f"- Heads: {cfg_a['num_heads']}")
        st.markdown("**Model B:** facebook/bart-large-cnn")
        st.markdown(f"- Device: `{DEVICE}`")

    # --- Input ---
    st.subheader("Input")
    ex_choice = st.selectbox("Load an example", ["(custom input)", "Example 1 – Amazon", "Example 2 – Vaccine", "Example 3 – Apple"])
    if ex_choice != "(custom input)":
        default = EXAMPLES[["Example 1 – Amazon", "Example 2 – Vaccine", "Example 3 – Apple"].index(ex_choice)]
    else:
        default = ""
    text_input = st.text_area("Article text", value=default, height=200, placeholder="Paste article here...")

    run_btn = st.button("🚀 Summarize with both models", type="primary", disabled=(not text_input.strip()))

    if run_btn and text_input.strip():
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Model A — From Scratch")
            if model_a is not None:
                with st.spinner("Generating..."):
                    summary_a, lat_a = summarize_a(text_input, model_a, tokenizer_a, cfg_a)
                st.success(summary_a)
                st.caption(f"⏱ Latency: {lat_a:.3f}s")
            else:
                st.error("Model A not loaded (not trained yet).")
                summary_a, lat_a = None, None

        with col_b:
            st.subheader("Model B — BART-large-cnn")
            with st.spinner("Generating..."):
                summary_b, lat_b = bart_summarize(text_input, model_b, tokenizer_b)
            st.success(summary_b)
            st.caption(f"⏱ Latency: {lat_b:.3f}s")

        if summary_a and summary_b:
            rouge = compute_rouge([summary_a], [summary_b])
            st.info(
                f"ROUGE-1 (A vs B): `{rouge['rouge1']:.4f}` | "
                f"ROUGE-L: `{rouge['rougeL']:.4f}`"
            )

    st.divider()

    # --- Test Results Dashboard ---
    st.subheader("📊 Test Set Results")

    if os.path.exists(EVAL_RESULTS):
        with open(EVAL_RESULTS) as f:
            results = json.load(f)

        metrics = {
            'Metric':          ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Avg Latency (s)', 'Parameters'],
            'Model A (Scratch)': [
                f"{results['model_a']['rouge']['rouge1']:.4f}",
                f"{results['model_a']['rouge']['rouge2']:.4f}",
                f"{results['model_a']['rouge']['rougeL']:.4f}",
                f"{results['model_a']['latency']:.4f}",
                f"{results['model_a']['params']:,}",
            ],
            'Model B (BART-L)': [
                f"{results['model_b']['rouge']['rouge1']:.4f}",
                f"{results['model_b']['rouge']['rouge2']:.4f}",
                f"{results['model_b']['rouge']['rougeL']:.4f}",
                f"{results['model_b']['latency']:.4f}",
                f"{results['model_b']['params']:,}",
            ],
        }
        st.dataframe(pd.DataFrame(metrics).set_index('Metric'), use_container_width=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        a_vals = [results['model_a']['rouge'][k] for k in ['rouge1', 'rouge2', 'rougeL']]
        b_vals = [results['model_b']['rouge'][k] for k in ['rouge1', 'rouge2', 'rougeL']]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], a_vals, width=0.4, label='Model A (Scratch)', color='steelblue')
        ax.bar([i + 0.2 for i in x], b_vals, width=0.4, label='Model B (BART-L)',  color='darkorange')
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylabel('Score'); ax.legend(); ax.set_ylim(0, 1)
        ax.set_title('ROUGE Scores — Test Set')
        st.pyplot(fig)

        # Sample predictions
        st.subheader("Sample Predictions")
        preds_a = results['model_a']['predictions']
        preds_b = results['model_b']['predictions']
        refs    = results['model_a']['references']
        for i in range(min(3, len(refs))):
            with st.expander(f"Example {i+1}"):
                st.markdown(f"**Reference:** {refs[i]}")
                st.markdown(f"**Model A:** {preds_a[i]}")
                st.markdown(f"**Model B:** {preds_b[i]}")
    else:
        st.info("Run `python evaluate/run_eval.py` to generate test results, then refresh.")

    st.divider()
    st.caption("Dataset: CNN/DailyMail 3.0.0 · Model B: facebook/bart-large-cnn (not fine-tuned further)")


if __name__ == '__main__':
    main()
