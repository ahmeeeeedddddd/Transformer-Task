import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast, BartTokenizer

OFFLINE_DIR = os.path.join(os.path.dirname(__file__), 'cnn_dailymail_offline')
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'tokenizer_16k')


def load_cnn_dailymail(train_size=150000, val_size=2000, test_size=2000, seed=42):
    if os.path.exists(OFFLINE_DIR):
        print(f"Loading CNN/DailyMail from offline cache: {OFFLINE_DIR}")
        ds = load_from_disk(OFFLINE_DIR)
    else:
        print("Offline cache not found; loading CNN/DailyMail via Hugging Face...")
        ds = load_dataset('abisee/cnn_dailymail', '3.0.0')

    train = ds['train'].shuffle(seed=seed).select(range(min(train_size, len(ds['train'])))) if train_size and train_size > 0 else ds['train']
    val   = ds['validation'].shuffle(seed=seed).select(range(min(val_size, len(ds['validation'])))) if val_size and val_size > 0 else ds['validation']
    test  = ds['test'].shuffle(seed=seed).select(range(min(test_size, len(ds['test'])))) if test_size and test_size > 0 else ds['test']

    return train, val, test


def get_tokenizer_for_model_a():
    if os.path.exists(os.path.join(TOKENIZER_DIR, 'tokenizer.json')) or os.path.exists(os.path.join(TOKENIZER_DIR, 'vocab.json')):
        print(f"Loading custom 16k tokenizer from {TOKENIZER_DIR}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            TOKENIZER_DIR,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        return tokenizer
    print("Custom 16k tokenizer not found; using BartTokenizer fallback.")
    return BartTokenizer.from_pretrained('facebook/bart-large-cnn')


class SummarizationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_src_len=512, max_tgt_len=128):
        self.data        = hf_dataset
        self.tokenizer   = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        src_enc = self.tokenizer(
            item['article'],
            max_length=self.max_src_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        tgt_enc = self.tokenizer(
            item['highlights'],
            max_length=self.max_tgt_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'src':      src_enc['input_ids'].squeeze(0),
            'tgt':      tgt_enc['input_ids'].squeeze(0),
            'tgt_text': item['highlights'],
        }
