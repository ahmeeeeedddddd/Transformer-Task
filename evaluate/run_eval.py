import os
import sys
import time
import json
import argparse
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm

from from_scratch.transformer import Seq2SeqTransformer
from pretrained.inference import load_model_b
from data.dataset import load_cnn_dailymail, SummarizationDataset, get_tokenizer_for_model_a
from evaluate.metrics import compute_rouge, count_parameters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = os.path.join('checkpoints', 'model_a_best.pt')
CACHE_A = os.path.join('checkpoints', 'cache_model_a.json')
CACHE_B = os.path.join('checkpoints', 'cache_model_b.json')
N_EXAMPLES = 3
SEED = 42


def load_model_a(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
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


def run_model_a_on_test(model, tokenizer, test_ds, cfg, batch_size=8):
    if os.path.exists(CACHE_A):
        with open(CACHE_A, 'r', encoding='utf-8') as f:
            cached = json.load(f)
            if len(cached.get('preds', [])) == len(test_ds):
                print(f"Loaded cached Model A predictions ({len(test_ds)} samples). Skipping Model A eval.")
                return cached['preds'], cached['refs'], cached['latency']

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    preds, refs = [], []
    latencies = []
    rep_penalty = cfg.get('repetition_penalty', 1.5)

    for batch in tqdm(loader, desc='Eval Model A (Batched FP16)'):
        src = batch['src'].to(DEVICE)
        t0  = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            gen = model.generate(src, max_len=cfg['max_tgt_len'], beam_size=1, repetition_penalty=rep_penalty)
        latencies.append((time.time() - t0) / src.size(0))

        for g, r in zip(gen, batch['tgt_text']):
            ids  = g.squeeze().tolist()
            text = tokenizer.decode(ids if isinstance(ids, list) else [ids], skip_special_tokens=True)
            preds.append(text)
            refs.append(r)

    avg_lat = sum(latencies) / len(latencies)
    with open(CACHE_A, 'w', encoding='utf-8') as f:
        json.dump({'preds': preds, 'refs': refs, 'latency': avg_lat}, f)

    return preds, refs, avg_lat


def run_model_b_on_test(model_b, tokenizer_b, test_data, batch_size=8):
    preds, refs, latencies = [], [], []
    start_idx = 0

    if os.path.exists(CACHE_B):
        with open(CACHE_B, 'r', encoding='utf-8') as f:
            saved = json.load(f)
            if len(saved.get('preds', [])) == len(test_data):
                print(f"Loaded cached Model B predictions ({len(test_data)} samples).")
                return saved['preds'], saved['refs'], saved['latency']
            elif len(saved.get('preds', [])) > 0:
                preds = saved['preds']
                refs = saved['refs']
                latencies = saved['latencies']
                start_idx = len(preds)
                print(f"Resuming Model B from sample {start_idx}/{len(test_data)}...")

    articles = [test_data[i]['article'] for i in range(start_idx, len(test_data))]
    targets  = [test_data[i]['highlights'] for i in range(start_idx, len(test_data))]

    for i in tqdm(range(0, len(articles), batch_size), desc='Eval Model B (Batched FP16)'):
        batch_articles = articles[i:i + batch_size]
        batch_targets  = targets[i:i + batch_size]

        inputs = tokenizer_b(
            batch_articles,
            return_tensors='pt',
            max_length=1024,
            truncation=True,
            padding=True,
        ).to(DEVICE)

        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            ids = model_b.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        elapsed = time.time() - t0
        latencies.extend([elapsed / len(batch_articles)] * len(batch_articles))

        decoded = tokenizer_b.batch_decode(ids, skip_special_tokens=True)
        preds.extend(decoded)
        refs.extend(batch_targets)

        if (i // batch_size) % 25 == 0:
            with open(CACHE_B, 'w', encoding='utf-8') as f:
                json.dump({'preds': preds, 'refs': refs, 'latencies': latencies, 'latency': sum(latencies)/len(latencies)}, f)

    avg_lat = sum(latencies) / len(latencies)
    with open(CACHE_B, 'w', encoding='utf-8') as f:
        json.dump({'preds': preds, 'refs': refs, 'latencies': latencies, 'latency': avg_lat}, f)

    return preds, refs, avg_lat


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model A vs Model B on test set")
    parser.add_argument('--test_size', type=int, default=-1, help="Number of test samples (-1 for full set, default: -1)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for evaluation (default: 8)")
    parser.add_argument('--force_reeval_a', action='store_true', help="Force re-evaluation of Model A even if cache exists")
    args = parser.parse_args()

    if args.force_reeval_a and os.path.exists(CACHE_A):
        os.remove(CACHE_A)

    print("Loading tokenizers...")
    tokenizer_a = get_tokenizer_for_model_a()
    tokenizer_b = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    sample_str = "full test set (11,490)" if args.test_size <= 0 else f"{args.test_size} samples"
    print(f"Loading test data ({sample_str})...")
    _, _, test_data = load_cnn_dailymail(train_size=1, val_size=1, test_size=args.test_size, seed=SEED)
    test_ds_a = SummarizationDataset(test_data, tokenizer_a, max_src_len=512, max_tgt_len=128)

    # --- Model A ---
    print(f"\nLoading Model A from {CHECKPOINT}...")
    model_a, cfg_a = load_model_a(CHECKPOINT)
    preds_a, refs_a, lat_a = run_model_a_on_test(model_a, tokenizer_a, test_ds_a, cfg_a, batch_size=args.batch_size)
    rouge_a = compute_rouge(preds_a, refs_a)
    params_a = count_parameters(model_a)

    # --- Model B ---
    print("\nLoading Model B (facebook/bart-large-cnn)...")
    model_b, _ = load_model_b()
    preds_b, refs_b, lat_b = run_model_b_on_test(model_b, tokenizer_b, test_data, batch_size=args.batch_size)
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
