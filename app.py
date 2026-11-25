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

EXAMPLES = {
    "🌿 Example 1 – Amazon Rainforest": (
        "The Amazon rainforest, also known in English as Amazonia, is a moist broadleaf tropical rainforest "
        "in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses "
        "7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the "
        "rainforest. This region includes territory belonging to nine nations and 3,344 formally "
        "acknowledged indigenous territories. The majority of the forest is contained within Brazil, "
        "with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts "
        "in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela."
    ),
    "💉 Example 2 – mRNA Flu Vaccine": (
        "Scientists have developed a new vaccine that shows promising results against multiple strains of "
        "influenza. The vaccine, developed over five years of research, uses mRNA technology similar to "
        "COVID-19 vaccines. In clinical trials involving 10,000 participants across 15 countries, the "
        "vaccine demonstrated 85% efficacy against seasonal flu strains. Researchers believe this could "
        "eliminate the need for annual flu shots if approved. The FDA is expected to begin review of the "
        "vaccine next month, with potential approval coming as early as next year."
    ),
    "📈 Example 3 – Apple Quarterly Earnings": (
        "Apple Inc. announced record quarterly earnings on Tuesday, reporting revenue of $97.3 billion "
        "for the fourth quarter of fiscal year 2024, up 6% year over year. iPhone sales remained the "
        "company's largest segment at $46.2 billion, while services revenue hit an all-time high of "
        "$24.2 billion. CEO Tim Cook highlighted strong performance in emerging markets, particularly "
        "India and Southeast Asia. The company also announced a $110 billion share buyback program, "
        "the largest in Apple history. Shares rose 3% in after-hours trading following the announcement."
    ),
}


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
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
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
    st.set_page_config(
        page_title="Neural Summarization | Scratch vs BART",
        layout="wide",
        page_icon="⚡",
        initial_sidebar_state="expanded"
    )

    # Custom Clean Styling
    st.markdown("""
        <style>
        .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.5rem; }
        .model-card { border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
        .latency-badge { font-family: monospace; font-size: 0.85rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; background-color: #f1f5f9; color: #334155; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">⚡ Neural Text Summarization Benchmark</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluating a <b>From-Scratch Pre-LN Transformer (95M)</b> against <b>facebook/bart-large-cnn (406M)</b> on the CNN/DailyMail benchmark.</div>',
        unsafe_allow_html=True
    )

    model_a, tokenizer_a, cfg_a = get_model_a()
    model_b, tokenizer_b        = get_model_b()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        beam_size = st.slider("Beam Search Size", 1, 8, 4)
        rep_penalty = st.slider("Repetition Penalty (Model A)", 1.0, 2.5, 1.5, step=0.1)
        if cfg_a:
            cfg_a['beam_size'] = beam_size
            cfg_a['repetition_penalty'] = rep_penalty

        st.markdown("---")
        st.markdown("### 🖥️ Hardware & Models")
        st.caption(f"Device: `{DEVICE}`")
        if torch.cuda.is_available():
            st.caption(f"GPU: `{torch.cuda.get_device_name(0)}`")
        
        st.markdown("---")
        st.markdown("**Model A (Scratch):**")
        st.caption("• Architecture: 6E / 6D Pre-LN\n• Heads: 8 | d_model: 512\n• Vocab: 16k Byte-Level BPE\n• Params: ~95.6M")
        st.markdown("**Model B (Pretrained):**")
        st.caption("• Checkpoint: `facebook/bart-large-cnn`\n• 12E / 12D Standard\n• Params: ~406.3M")

    # --- Preset Selector & Input ---
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        preset_choice = st.selectbox("Select Preset Benchmark Article:", ["(Write / Paste custom text)"] + list(EXAMPLES.keys()))
    
    default_text = EXAMPLES[preset_choice] if preset_choice != "(Write / Paste custom text)" else ""
    text_input = st.text_area("Source Article Text:", value=default_text, height=180, placeholder="Type or paste text here...")

    run_col1, run_col2 = st.columns([1, 4])
    with run_col1:
        run_btn = st.button("🔥 Run Comparison", type="primary", use_container_width=True, disabled=(not text_input.strip()))

    if run_btn and text_input.strip():
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🛠️ Model A — From Scratch")
            st.caption("Pre-LN Transformer (95.6M parameters)")
            if model_a is not None:
                with st.spinner("Generating summary with Model A..."):
                    summary_a, lat_a = summarize_a(text_input, model_a, tokenizer_a, cfg_a)
                st.success(summary_a)
                st.markdown(f'<span class="latency-badge">⚡ Inference Latency: {lat_a:.3f}s</span>', unsafe_allow_html=True)
            else:
                st.error("⚠️ Model A checkpoint not loaded.")
                summary_a, lat_a = None, None

        with col_b:
            st.markdown("### 🏆 Model B — Pretrained")
            st.caption("facebook/bart-large-cnn (406.3M parameters)")
            with st.spinner("Generating summary with BART..."):
                summary_b, lat_b = bart_summarize(text_input, model_b, tokenizer_b, beam_size=beam_size)
            st.success(summary_b)
            st.markdown(f'<span class="latency-badge">⚡ Inference Latency: {lat_b:.3f}s</span>', unsafe_allow_html=True)

        if summary_a and summary_b:
            rouge = compute_rouge([summary_a], [summary_b])
            st.divider()
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Cross-Model ROUGE-1", f"{rouge['rouge1']:.3f}")
            mcol2.metric("Cross-Model ROUGE-L", f"{rouge['rougeL']:.3f}")
            speedup = (lat_b / max(1e-4, lat_a)) if lat_a else 1.0
            mcol3.metric("Model A Speedup", f"{speedup:.1f}x Faster")

    st.divider()

    # --- Evaluation Dashboard Section ---
    st.subheader("📊 Full Held-Out Test Set Results")
    if os.path.exists(EVAL_RESULTS):
        with open(EVAL_RESULTS, 'r', encoding='utf-8') as f:
            res = json.load(f)

        dcol1, dcol2 = st.columns([1, 1])
        with dcol1:
            metrics_table = {
                'Quality & Efficiency Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Avg Latency (s)', 'Parameter Count'],
                'Model A (From Scratch)': [
                    f"{res['model_a']['rouge']['rouge1']:.4f}",
                    f"{res['model_a']['rouge']['rouge2']:.4f}",
                    f"{res['model_a']['rouge']['rougeL']:.4f}",
                    f"{res['model_a']['latency']:.4f}",
                    f"{res['model_a']['params']:,}",
                ],
                'Model B (BART-Large)': [
                    f"{res['model_b']['rouge']['rouge1']:.4f}",
                    f"{res['model_b']['rouge']['rouge2']:.4f}",
                    f"{res['model_b']['rouge']['rougeL']:.4f}",
                    f"{res['model_b']['latency']:.4f}",
                    f"{res['model_b']['params']:,}",
                ]
            }
            st.dataframe(pd.DataFrame(metrics_table).set_index('Quality & Efficiency Metric'), use_container_width=True)

        with dcol2:
            fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
            labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            a_vals = [res['model_a']['rouge']['rouge1'], res['model_a']['rouge']['rouge2'], res['model_a']['rouge']['rougeL']]
            b_vals = [res['model_b']['rouge']['rouge1'], res['model_b']['rouge']['rouge2'], res['model_b']['rouge']['rougeL']]
            x = range(len(labels))
            ax.bar([i - 0.18 for i in x], a_vals, width=0.36, label='Model A (Scratch)', color='#4361ee')
            ax.bar([i + 0.18 for i in x], b_vals, width=0.36, label='Model B (BART-L)',  color='#f72585')
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 0.6)
            ax.set_ylabel('F1 Score')
            ax.legend(frameon=True)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        # Sample predictions
        st.subheader("🔍 Sample Predictions vs Ground Truth")
        refs = res['model_a'].get('references', [])
        preds_a = res['model_a'].get('predictions', [])
        preds_b = res['model_b'].get('predictions', [])
        for i in range(min(3, len(refs))):
            with st.expander(f"📌 Held-Out Example {i+1}"):
                st.markdown(f"**Reference Ground Truth:** {refs[i]}")
                st.markdown(f"**Model A Output:** {preds_a[i]}")
                st.markdown(f"**Model B Output:** {preds_b[i]}")
    else:
        st.info("Run `python evaluate/run_eval.py` to populate test set metrics.")


if __name__ == '__main__':
    main()
