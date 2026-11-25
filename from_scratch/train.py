import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from from_scratch.transformer import Seq2SeqTransformer
from data.dataset import SummarizationDataset, load_cnn_dailymail, get_tokenizer_for_model_a
from evaluate.metrics import compute_rouge

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device('cuda' if USE_CUDA else 'cpu')

CFG = {
    'd_model':             512,
    'num_heads':           8,
    'd_ff':                2048,
    'num_encoder_layers':  6,
    'num_decoder_layers':  6,
    'max_src_len':         512,
    'max_tgt_len':         128,
    'dropout':             0.1,
    'pad_idx':             1,
    'bos_idx':             0,
    'eos_idx':             2,
    'batch_size':          8,
    'grad_accum':          4,      # effective batch size = 32
    'epochs':              15,
    'lr':                  4e-4,   # boosted LR for Pre-LN architecture
    'warmup_steps':        3000,
    'label_smoothing':     0.1,
    'clip_grad':           1.0,
    'train_size':          150000, # 3x more training data
    'val_size':            2000,
    'test_size':           2000,
    'seed':                42,
    'checkpoint_dir':      'checkpoints',
    'checkpoint_name':     'model_a_best.pt',
    'beam_size':           4,
    'repetition_penalty':  1.5,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-5):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_model(vocab_size):
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=CFG['d_model'],
        num_heads=CFG['num_heads'],
        d_ff=CFG['d_ff'],
        num_encoder_layers=CFG['num_encoder_layers'],
        num_decoder_layers=CFG['num_decoder_layers'],
        max_src_len=CFG['max_src_len'],
        max_tgt_len=CFG['max_tgt_len'],
        dropout=CFG['dropout'],
        pad_idx=CFG['pad_idx'],
        bos_idx=CFG['bos_idx'],
        eos_idx=CFG['eos_idx'],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model A (Pre-LN) parameters: {n_params:,} (vocab_size={vocab_size})")
    return model


def train_epoch(model, loader, optimizer, total_steps, scaler, criterion, global_step):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        src = batch['src'].to(DEVICE)
        tgt = batch['tgt'].to(DEVICE)

        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.amp.autocast('cuda', enabled=USE_CUDA):
            logits = model(src, tgt_in)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))
            loss = loss / CFG['grad_accum']

        scaler.scale(loss).backward()
        total_loss += loss.item() * CFG['grad_accum']

        if (i + 1) % CFG['grad_accum'] == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            lr_mult = warmup_cosine_lr(global_step, CFG['warmup_steps'], total_steps, CFG['lr'])
            for pg in optimizer.param_groups:
                pg['lr'] = CFG['lr'] * lr_mult

    return total_loss / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, tokenizer):
    model.eval()
    total_loss = 0.0
    preds, refs = [], []

    for batch in tqdm(loader, desc='Val', leave=False):
        src = batch['src'].to(DEVICE)
        tgt = batch['tgt'].to(DEVICE)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.amp.autocast('cuda', enabled=USE_CUDA):
            logits = model(src, tgt_in)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        total_loss += loss.item()

        gen = model.generate(src, max_len=CFG['max_tgt_len'], beam_size=1, repetition_penalty=CFG['repetition_penalty'])
        for g, r in zip(gen, batch['tgt_text']):
            token_ids = g.squeeze().tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            preds.append(text)
            refs.append(r)

    rouge = compute_rouge(preds, refs)
    return total_loss / len(loader), rouge


def train():
    set_seed(CFG['seed'])
    os.makedirs(CFG['checkpoint_dir'], exist_ok=True)

    print("Loading Model A tokenizer...")
    tokenizer = get_tokenizer_for_model_a()
    vocab_size = len(tokenizer)
    CFG['vocab_size'] = vocab_size

    print(f"Loading dataset ({CFG['train_size']} train samples)...")
    train_data, val_data, _ = load_cnn_dailymail(
        train_size=CFG['train_size'],
        val_size=CFG['val_size'],
        test_size=1,
        seed=CFG['seed'],
    )

    train_ds = SummarizationDataset(train_data, tokenizer, CFG['max_src_len'], CFG['max_tgt_len'])
    val_ds   = SummarizationDataset(val_data,   tokenizer, CFG['max_src_len'], CFG['max_tgt_len'])

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    total_steps = (len(train_loader) // CFG['grad_accum']) * CFG['epochs']
    print(f"Total optimization steps: {total_steps:,}")

    model     = build_model(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=CFG['pad_idx'], label_smoothing=CFG['label_smoothing'])
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_CUDA)

    best_rouge = 0.0
    global_step = 0

    for epoch in range(1, CFG['epochs'] + 1):
        t0 = time.time()
        train_loss, global_step = train_epoch(model, train_loader, optimizer, total_steps, scaler, criterion, global_step)
        val_loss, rouge = evaluate(model, val_loader, criterion, tokenizer)
        elapsed = time.time() - t0

        r1 = rouge['rouge1']
        rl = rouge['rougeL']
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"R1={r1:.4f} | RL={rl:.4f} | {elapsed:.0f}s")

        if r1 > best_rouge:
            best_rouge = r1
            path = os.path.join(CFG['checkpoint_dir'], CFG['checkpoint_name'])
            torch.save({'model_state': model.state_dict(), 'cfg': CFG, 'epoch': epoch, 'rouge1': r1}, path)
            print(f"  → Saved best checkpoint (ROUGE-1={r1:.4f})")

    print(f"\nTraining complete. Best ROUGE-1: {best_rouge:.4f}")


if __name__ == '__main__':
    train()
