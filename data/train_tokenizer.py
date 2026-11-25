import os
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from data.dataset import load_cnn_dailymail

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'tokenizer_16k')
VOCAB_SIZE = 16384

def train_tokenizer():
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    print("Loading training texts for tokenizer...")
    train_data, _, _ = load_cnn_dailymail(train_size=50000, val_size=1, test_size=1)

    def text_iterator():
        for item in train_data:
            yield item['article']
            yield item['highlights']

    print(f"Training Byte-Level BPE tokenizer with vocab size {VOCAB_SIZE}...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
    )

    tokenizer.save_model(TOKENIZER_DIR)

    # Wrap in HuggingFace fast tokenizer for convenient encode/decode
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(TOKENIZER_DIR, "vocab.json"),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"Custom 16k tokenizer saved to: {TOKENIZER_DIR}")

if __name__ == '__main__':
    train_tokenizer()
