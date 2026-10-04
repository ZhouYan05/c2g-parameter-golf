"""Official parameter-golf FineWeb shard IO + BPB metric.

Faithful re-implementation of the pieces used by the challenge's train_gpt.py:
  * shard header: 256 x int32, magic=20240520, version=1, header[2]=num_tokens
  * payload:     uint16 tokens, little endian
  * BPB:         bits_per_token * tokens_per_byte, with per-token byte counts
                 derived from the SentencePiece LUTs.
"""
import glob
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def load_data_shard(path):
    path = Path(path)
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {path}")
    n = int(header[2])
    tok = np.fromfile(path, dtype="<u2", count=n, offset=HEADER_INTS * 4)
    if tok.size != n:
        raise ValueError(f"Short read for {path}: got {tok.size}, expected {n}")
    return tok


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files for pattern: {pattern}")
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError("validation split too short")
    return tokens[: usable + 1]


def build_sentencepiece_luts(sp, vocab_size):
    """Verbatim logic from train_gpt.py build_sentencepiece_luts()."""
    sp_vocab = int(sp.vocab_size())
    table = max(sp_vocab, vocab_size)
    base = np.zeros((table,), dtype=np.int16)
    lead = np.zeros((table,), dtype=np.bool_)
    bnd = np.ones((table,), dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        bnd[tid] = False
        if sp.is_byte(tid):
            base[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            lead[tid] = True
            piece = piece[1:]
        base[tid] = len(piece.encode("utf-8"))
    return base, lead, bnd


@torch.no_grad()
def eval_bpb(model, tokens, seq_len, batch, n_batches, luts, device):
    """Returns (mean_xent_nats, bpb, tokens_per_byte, tokens_used)."""
    base, lead, bnd = luts
    base_t = torch.from_numpy(base).to(device)
    lead_t = torch.from_numpy(lead).to(device)
    bnd_t = torch.from_numpy(bnd).to(device)
    model.eval()
    loss_sum = 0.0
    tok_n = 0
    byte_n = 0.0
    span = batch * seq_len
    for i in range(n_batches):
        start = i * span
        chunk = tokens[start : start + span + 1]
        if chunk.numel() < span + 1:
            break
        x = chunk[:-1].reshape(batch, seq_len).long()
        y = chunk[1:].reshape(batch, seq_len).long()
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
        prev = x.reshape(-1)
        tgt = y.reshape(-1)
        tb = base_t[tgt].to(torch.int32) + (lead_t[tgt] & ~bnd_t[prev]).to(torch.int32)
        tok_n += int(tgt.numel())
        byte_n += float(tb.sum())
    model.train()
    mean_nats = loss_sum / tok_n
    bits_per_token = mean_nats / math.log(2)
    tokens_per_byte = tok_n / byte_n
    return mean_nats, bits_per_token * tokens_per_byte, tokens_per_byte, tok_n


def uniform_bpb_anchor(tokens, seq_len, batch, n_batches, vocab_size, luts):
    """Analytic check: a uniform-over-vocab predictor gives bits_per_token=log2(V)."""
    base, lead, bnd = luts
    span = batch * seq_len
    tok_n = 0
    byte_n = 0
    for i in range(n_batches):
        start = i * span
        chunk = tokens[start : start + span + 1]
        if chunk.size < span + 1:
            break
        prev = chunk[:-1].reshape(-1).astype(np.int64)
        tgt = chunk[1:].reshape(-1).astype(np.int64)
        tb = base[tgt].astype(np.int32) + (lead[tgt] & ~bnd[prev]).astype(np.int32)
        tok_n += int(tgt.size)
        byte_n += float(tb.sum())
    tpb = tok_n / byte_n
    return math.log2(vocab_size) * tpb, tpb, tok_n
