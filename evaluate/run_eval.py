import os
import time
import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm

from from_scratch.transformer import Seq2SeqTransformer
from pretrained.inference import load_model_b, summarize as bart_summarize
from data.dataset import load_cnn_dailymail, SummarizationDataset
from evaluate.metrics import compute_rouge, count_parameters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = os.path.join('checkpoints', 'model_a_best.pt')
N_EXAMPLES  = 3
TEST_SIZE   = 2000
SEED        = 42


def load_model_a(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    cfg  = ckpt['cfg']
    model = Seq2SeqTransformer(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'],
        num_encoder_layers=cfg['num_encoder_layers'],
        num_decoder_layers=cfg['num_decoder_layers'],
        max_src_len=cfg['max_src_len'],
        max_tgt_len=cfg['max_tgt_len'],
        dropout=0.0,
        pad_idx=cfg['pad_idx'],
        bos_idx=cfg['bos_idx'],
        eos_idx=cfg['eos_idx'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, cfg


def run_model_a_on_test(model, tokenizer, test_ds, cfg):
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    preds, refs = [], []
    latencies = []

    for batch in tqdm(loader, desc='Eval Model A'):
        src = batch['src'].to(DEVICE)
        t0  = time.time()
        with torch.no_grad():
            gen = model.generate(src, max_len=cfg['max_tgt_len'], beam_size=cfg.get('beam_size', 4))
        latencies.append((time.time() - t0) / src.size(0))

        for g, r in zip(gen, batch['tgt_text']):
            ids  = g.squeeze().tolist()
            text = tokenizer.decode(ids if isinstance(ids, list) else [ids], skip_special_tokens=True)
            preds.append(text)
            refs.append(r)

    return preds, refs, sum(latencies) / len(latencies)


def run_model_b_on_test(model_b, tokenizer_b, test_data):
    preds, refs, latencies = [], [], []

    for item in tqdm(test_data, desc='Eval Model B'):
        summary, lat = bart_summarize(item['article'], model_b, tokenizer_b)
        preds.append(summary)
        refs.append(item['highlights'])
        latencies.append(lat)

    return preds, refs, sum(latencies) / len(latencies)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model A vs Model B on test set")
    parser.add_argument('--test_size', type=int, default=-1, help="Number of test samples to evaluate (-1 for full test set, default: -1)")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    sample_str = "full test set (11,490)" if args.test_size <= 0 else f"{args.test_size} samples"
    print(f"Loading test data ({sample_str})...")
    _, _, test_data = load_cnn_dailymail(train_size=1, val_size=1, test_size=args.test_size, seed=SEED)
    test_ds = SummarizationDataset(test_data, tokenizer, max_src_len=512, max_tgt_len=128)

    # --- Model A ---
    print(f"\nLoading Model A from {CHECKPOINT}...")
    model_a, cfg_a = load_model_a(CHECKPOINT)
    preds_a, refs_a, lat_a = run_model_a_on_test(model_a, tokenizer, test_ds, cfg_a)
    rouge_a = compute_rouge(preds_a, refs_a)
    params_a = count_parameters(model_a)

    # --- Model B ---
    print("\nLoading Model B (facebook/bart-large-cnn)...")
    model_b, tokenizer_b = load_model_b()
    preds_b, refs_b, lat_b = run_model_b_on_test(model_b, tokenizer_b, test_data)
    rouge_b = compute_rouge(preds_b, refs_b)
    params_b = count_parameters(model_b)

    # --- Results table ---
    print("\n" + "=" * 65)
    print(f"{'Metric':<20} {'Model A (Scratch)':>20} {'Model B (BART-L)':>20}")
    print("=" * 65)
    print(f"{'ROUGE-1':<20} {rouge_a['rouge1']:>20.4f} {rouge_b['rouge1']:>20.4f}")
    print(f"{'ROUGE-2':<20} {rouge_a['rouge2']:>20.4f} {rouge_b['rouge2']:>20.4f}")
    print(f"{'ROUGE-L':<20} {rouge_a['rougeL']:>20.4f} {rouge_b['rougeL']:>20.4f}")
    print(f"{'Avg latency (s)':<20} {lat_a:>20.4f} {lat_b:>20.4f}")
    print(f"{'Parameters':<20} {params_a:>20,} {params_b:>20,}")
    print("=" * 65)

    # --- Examples ---
    print(f"\n--- {N_EXAMPLES} Generated Examples ---")
    for i in range(N_EXAMPLES):
        print(f"\n[{i+1}] Reference : {refs_a[i]}")
        print(f"    Model A   : {preds_a[i]}")
        print(f"    Model B   : {preds_b[i]}")

    results = {
        'model_a': {'rouge': rouge_a, 'latency': lat_a, 'params': params_a,
                    'predictions': preds_a[:20], 'references': refs_a[:20]},
        'model_b': {'rouge': rouge_b, 'latency': lat_b, 'params': params_b,
                    'predictions': preds_b[:20], 'references': refs_b[:20]},
    }
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to eval_results.json")


if __name__ == '__main__':
    main()
