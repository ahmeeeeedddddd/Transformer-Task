import time
import torch
from rouge_score import rouge_scorer


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    agg = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k] += scores[k].fmeasure

    n = len(predictions)
    return {k: v / n for k, v in agg.items()}


def measure_latency(fn, *args, n_runs=5, warmup=1, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(n_runs):
        fn(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    return (time.time() - t0) / n_runs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
