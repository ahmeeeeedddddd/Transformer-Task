import time
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_NAME = 'facebook/bart-large-cnn'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_b():
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer


def summarize(text, model, tokenizer, max_input=1024, max_summary=128, min_summary=30, beam_size=4):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_input,
        truncation=True,
        padding=False,
    ).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_summary,
            min_length=min_summary,
            num_beams=beam_size,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    latency = time.time() - t0

    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    return summary, latency
