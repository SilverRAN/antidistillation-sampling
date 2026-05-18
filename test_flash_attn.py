#!/usr/bin/env python
"""Smoke test for flash-attn in the local antidistillation-sampling environment.

Run from this directory after activating the environment:

    source .venv/bin/activate
    python test_flash_attn.py

The test checks:
1. flash-attn can be imported.
2. CUDA is visible to PyTorch.
3. flash_attn_func runs a forward pass.
4. Gradients are finite after a backward pass.
5. The output is close to PyTorch scaled_dot_product_attention on a small case.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate flash-attn forward/backward on CUDA.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Only check imports and versions; does not require CUDA.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except Exception as exc:
        print(f"[FAIL] flash-attn import failed: {exc}", file=sys.stderr)
        return 1

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"flash_attn: {flash_attn.__version__}")

    if args.import_only:
        print("[OK] import-only check passed")
        return 0

    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"compute capability: sm_{capability[0]}{capability[1]}")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    shape = (args.batch_size, args.seq_len, args.num_heads, args.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)
    if out.shape != q.shape:
        print(f"[FAIL] unexpected output shape: got {tuple(out.shape)}, expected {shape}", file=sys.stderr)
        return 3
    if not torch.isfinite(out).all():
        print("[FAIL] flash-attn output contains non-finite values", file=sys.stderr)
        return 4

    loss = out.float().square().mean()
    loss.backward()
    for name, tensor in (("q.grad", q.grad), ("k.grad", k.grad), ("v.grad", v.grad)):
        if tensor is None or not torch.isfinite(tensor).all():
            print(f"[FAIL] {name} is missing or contains non-finite values", file=sys.stderr)
            return 5

    with torch.no_grad():
        sdpa_out = F.scaled_dot_product_attention(
            q.detach().transpose(1, 2),
            k.detach().transpose(1, 2),
            v.detach().transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,
        ).transpose(1, 2)
        max_abs = (out.detach().float() - sdpa_out.float()).abs().max().item()
        close = torch.allclose(out.detach().float(), sdpa_out.float(), atol=args.atol, rtol=args.rtol)

    print(f"shape: {shape}")
    print(f"dtype: {dtype}")
    print(f"output mean/std: {out.float().mean().item():.6f} / {out.float().std().item():.6f}")
    print(f"max abs diff vs torch SDPA: {max_abs:.6e}")

    if not close:
        print(
            f"[FAIL] flash-attn output is not close to torch SDPA "
            f"(atol={args.atol}, rtol={args.rtol})",
            file=sys.stderr,
        )
        return 6

    print("[OK] flash-attn forward/backward smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
