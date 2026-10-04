"""int8 + zlib round-trip, mirroring the challenge's artifact-size constraint.

The official train_gpt.py reports `final_int8_zlib_roundtrip` (size + val_bpb).
This module reproduces the *shape* of that check: per-row symmetric int8 for 2D
tensors, per-tensor int8 for vectors, scales kept in fp16, payload zlib-compressed.
"""
import io
import zlib

import torch


def quantize_state_dict_int8(sd):
    qsd = {}
    kinds = {}
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            qsd[k] = v.detach().clone()
            kinds[k] = "raw"
            continue
        v32 = v.detach().to(torch.float32)
        if v32.ndim >= 2:
            rows = v32.reshape(v32.shape[0], -1)
            amax = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            scale = amax / 127.0
            q = torch.round(rows / scale).clamp_(-127, 127).to(torch.int8).reshape(v32.shape)
            qsd[k] = q
            qsd[k + ".__scale"] = scale.reshape(-1).to(torch.float16)
            kinds[k] = "int8_row"
        else:
            amax = v32.abs().max().clamp_min(1e-8)
            scale = amax / 127.0
            q = torch.round(v32 / scale).clamp_(-127, 127).to(torch.int8)
            qsd[k] = q
            qsd[k + ".__scale"] = scale.to(torch.float16)
            kinds[k] = "int8_vec"
    return qsd, kinds


def zlib_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    comp = zlib.compress(raw, 9)
    return len(raw), len(comp)


def dequantize_int8(qsd, kinds, template):
    out = {}
    for k, ref in template.items():
        kind = kinds.get(k)
        if kind == "int8_row":
            sc = qsd[k + ".__scale"].to(torch.float32).view(-1, *([1] * (qsd[k].ndim - 1)))
            out[k] = (qsd[k].to(torch.float32) * sc).to(ref.dtype)
        elif kind == "int8_vec":
            out[k] = (qsd[k].to(torch.float32) * qsd[k + ".__scale"].to(torch.float32)).to(ref.dtype)
        else:
            out[k] = qsd[k] if k in qsd else ref
    return out


def roundtrip_state_dict(model):
    """Returns (compressed_bytes, raw_bytes, quantized_payload, kinds)."""
    sd = model.state_dict()
    qsd, kinds = quantize_state_dict_int8(sd)
    raw, comp = zlib_bytes(qsd)
    return comp, raw, qsd, kinds
