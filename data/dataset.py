import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def load_cnn_dailymail(train_size=50000, val_size=2000, test_size=2000, seed=42):
    ds = load_dataset('cnn_dailymail', '3.0.0')

    train = ds['train'].shuffle(seed=seed).select(range(min(train_size, len(ds['train']))))
    val   = ds['validation'].shuffle(seed=seed).select(range(min(val_size, len(ds['validation']))))
    test  = ds['test'].shuffle(seed=seed).select(range(min(test_size, len(ds['test']))))

    return train, val, test


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

        with self.tokenizer.as_target_tokenizer():
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
