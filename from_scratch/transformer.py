import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _scaled_dot_product(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, V)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        out = self._scaled_dot_product(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos   = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.pos(self.embed(x) * self.scale)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos   = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.pos(self.embed(x) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_encoder_layers=6, num_decoder_layers=6,
                 max_src_len=512, max_tgt_len=128, dropout=0.1,
                 pad_idx=1, bos_idx=0, eos_idx=2):
        super().__init__()
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_encoder_layers, max_src_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_decoder_layers, max_tgt_len, dropout)
        self.proj = nn.Linear(d_model, vocab_size)

        self.proj.weight = self.encoder.embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'embed' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _tgt_mask(self, tgt):
        T = tgt.size(1)
        pad_mask    = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device, dtype=torch.bool))
        return pad_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self._src_mask(src)
        tgt_mask = self._tgt_mask(tgt)
        enc_out  = self.encoder(src, src_mask)
        dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.proj(dec_out)

    @torch.no_grad()
    def generate(self, src, max_len=128, beam_size=4):
        self.eval()
        device = src.device
        B = src.size(0)

        src_mask = self._src_mask(src)
        enc_out  = self.encoder(src, src_mask)

        if beam_size <= 1:
            return self._greedy(enc_out, src_mask, B, max_len, device)
        return self._beam(enc_out, src_mask, B, max_len, beam_size, device)

    def _greedy(self, enc_out, src_mask, B, max_len, device):
        tgt = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tgt_mask = self._tgt_mask(tgt)
            logits   = self.proj(self.decoder(tgt, enc_out, src_mask, tgt_mask))
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
            done |= (next_tok.squeeze(-1) == self.eos_idx)
            if done.all():
                break
        return tgt

    def _beam(self, enc_out, src_mask, B, max_len, beam_size, device):
        # Only supports B=1 beam search; falls back to greedy for B>1
        if B > 1:
            return self._greedy(enc_out, src_mask, B, max_len, device)

        sequences = [(torch.full((1, 1), self.bos_idx, dtype=torch.long, device=device), 0.0)]
        completed = []

        for _ in range(max_len - 1):
            all_cands = []
            for seq, score in sequences:
                if seq[0, -1].item() == self.eos_idx:
                    completed.append((seq, score))
                    continue
                tgt_mask = self._tgt_mask(seq)
                logits   = self.proj(self.decoder(seq, enc_out, src_mask, tgt_mask))
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                top_lp, top_idx = log_probs.topk(beam_size)
                for lp, idx in zip(top_lp, top_idx):
                    new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                    all_cands.append((new_seq, score + lp.item()))

            if not all_cands:
                break
            all_cands.sort(key=lambda x: x[1] / (x[0].size(1) ** 0.6), reverse=True)
            sequences = all_cands[:beam_size]

        completed += sequences
        completed.sort(key=lambda x: x[1] / (x[0].size(1) ** 0.6), reverse=True)
        return completed[0][0]
