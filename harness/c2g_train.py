"""mu-scale parameter-golf training run.

Usage:
  python c2g_train.py --data-dir <repo>/data --tok <repo>/data/tokenizers/fineweb_1024_bpe.model \
      --tag base --opt adamw --out runs/base_seed1337.json
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2g_data import (build_sentencepiece_luts, eval_bpb, load_data_shard,
                      load_validation_tokens, uniform_bpb_anchor)
from c2g_model import GPT
from c2g_optim import build_optimizers, lr_at
from c2g_quant import dequantize_int8, roundtrip_state_dict

import sentencepiece as spm


def ascii_safe_path(p):
    """sentencepiece uses a narrow std::ifstream on Windows -> non-ASCII paths fail."""
    p = Path(p).resolve()
    if all(ord(c) < 128 for c in str(p)):
        return p
    import hashlib
    import shutil
    import tempfile
    tag = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    dst = Path(tempfile.gettempdir()) / f"c2g_{tag}_{p.name}"
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        shutil.copy2(p, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--variant", default="sp1024")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tie", type=int, default=0)
    ap.add_argument("--depth-share", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--max-seconds", type=float, default=480.0)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--opt", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-batches", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    ds_dir = data_dir / "datasets" / f"fineweb10B_{args.variant}"
    train_file = sorted(ds_dir.glob("fineweb_train_*.bin"))[0]
    val_glob = str(ds_dir / "fineweb_val_*.bin")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")
    log = open(log_path, "w", encoding="utf-8")

    def log0(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log.write(line + "\n")
        log.flush()
        print(line, flush=True)

    log0(f"TASK tag={args.tag} seed={args.seed} opt={args.opt} depth_share={args.depth_share} tie={args.tie}")

    sp = spm.SentencePieceProcessor()
    tok_path = ascii_safe_path(args.tok)
    log0(f"tokenizer={tok_path}")
    sp.load(str(tok_path))
    vocab_size = 1024
    luts = build_sentencepiece_luts(sp, vocab_size)

    val_np = load_validation_tokens(val_glob, args.seq)
    span = args.batch * args.seq
    need = min(val_np.size, args.eval_batches * span + 1)
    val_t = torch.from_numpy(val_np[:need].astype(np.int64))
    log0(f"data train_shard={train_file.name} val_tokens_available={val_np.size} val_eval_tokens={need}")

    anchor_bpb, anchor_tpb, anchor_tok = uniform_bpb_anchor(
        val_np, args.seq, args.batch, args.eval_batches, vocab_size, luts)
    log0(f"anchor uniform_over_vocab bpb={anchor_bpb:.6f} tokens_per_byte={anchor_tpb:.6f} tokens={anchor_tok}")

    model = GPT(vocab=vocab_size, dim=args.dim, n_layers=args.layers, n_heads=args.heads,
                tie=bool(args.tie), depth_share=args.depth_share).to(device)
    n_params = model.n_params()
    log0(f"model params={n_params} effective_depth={args.layers + args.depth_share - 1}")

    optimizers, opt_names = build_optimizers(model, lr=args.lr, opt=args.opt, muon_lr=args.muon_lr)
    log0(f"optimizers={opt_names} lr={args.lr} muon_lr={args.muon_lr}")

    train_np = load_data_shard(train_file)
    cursor = 0
    t0 = time.time()
    losses = []
    steps_done = 0
    tokens_seen = 0

    for step in range(args.steps):
        elapsed = time.time() - t0
        if elapsed > args.max_seconds:
            log0(f"stop: wall-clock budget reached at step {step} ({elapsed:.1f}s)")
            break
        if cursor + span + 1 > train_np.size:
            cursor = 0
        chunk = train_np[cursor : cursor + span + 1]
        cursor += span
        arr = torch.from_numpy(chunk.astype(np.int64))
        x = arr[:-1].reshape(args.batch, args.seq)
        y = arr[1:].reshape(args.batch, args.seq)

        lr = lr_at(step, int(args.max_seconds * 1.0) if args.steps > 100000 else args.steps,
                   args.lr, args.warmup)
        for opt, name in zip(optimizers, opt_names):
            for g in opt.param_groups:
                g["lr"] = (args.muon_lr if name == "muon" else lr)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        losses.append(float(loss.detach()))
        steps_done = step + 1
        tokens_seen += span
        if step % 25 == 0 or step == 1:
            log0(f"step {step} loss {float(loss):.4f} lr {lr:.3e} elapsed {elapsed:.1f}s tok {tokens_seen}")

    wall = time.time() - t0
    train_loss = float(np.mean(losses[-20:])) if losses else float("nan")
    log0(f"train done steps={steps_done} tokens={tokens_seen} wall={wall:.1f}s "
         f"s/step={wall / max(1, steps_done):.3f} tail_loss={train_loss:.4f}")

    val_loss, val_bpb, tpb, tok_used = eval_bpb(
        model, val_t, args.seq, args.batch, args.eval_batches, luts, device)
    log0(f"final val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} tokens_per_byte={tpb:.6f} eval_tokens={tok_used}")

    comp_bytes, raw_bytes, qsd, kinds = roundtrip_state_dict(model)
    art_path = out_path.with_suffix(".int8.pt")
    torch.save(qsd, art_path)
    log0(f"int8+zlib artifact bytes={comp_bytes} raw={raw_bytes} file={art_path.name} "
         f"under_16MB={comp_bytes <= 16_000_000}")

    template = model.state_dict()
    model.load_state_dict(dequantize_int8(qsd, kinds, template))
    q_val_loss, q_val_bpb, q_tpb, _ = eval_bpb(
        model, val_t, args.seq, args.batch, args.eval_batches, luts, device)
    log0(f"final_int8_zlib_roundtrip val_loss={q_val_loss:.4f} val_bpb={q_val_bpb:.4f}")

    result = dict(
        tag=args.tag, seed=args.seed, opt=args.opt, tie=bool(args.tie),
        depth_share=args.depth_share, dim=args.dim, layers=args.layers, heads=args.heads,
        effective_depth=args.layers + args.depth_share - 1,
        seq=args.seq, batch=args.batch, steps=steps_done, tokens_seen=tokens_seen,
        wall_seconds=round(wall, 2), seconds_per_step=round(wall / max(1, steps_done), 4),
        train_tail_loss=round(train_loss, 4), n_params=n_params,
        lr=args.lr, muon_lr=args.muon_lr if args.opt == "muon" else None,
        val_loss=round(val_loss, 4), val_bpb=round(val_bpb, 6),
        tokens_per_byte=round(tpb, 6), eval_tokens=tok_used,
        artifact_bytes_zlib_int8=comp_bytes, artifact_bytes_raw=raw_bytes,
        artifact_under_16mb=bool(comp_bytes <= 16_000_000),
        int8_val_loss=round(q_val_loss, 4), int8_val_bpb=round(q_val_bpb, 6),
        quant_bpb_delta=round(q_val_bpb - val_bpb, 6),
        anchor_bpb_uniform=round(anchor_bpb, 6), anchor_tokens_per_byte=round(anchor_tpb, 6),
        threads=args.threads, torch=torch.__version__, device="cpu",
    )
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log0("RESULT " + json.dumps(result, ensure_ascii=False))
    log.close()


if __name__ == "__main__":
    main()
