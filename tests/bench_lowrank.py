"""Micro-benchmark of the randomized low-rank Muon orthogonalization on CUDA.

Compares, on the NanoGPT parameter-bank shapes:
  * householder : eager orthogonalize_lowrank_eager (torch.linalg.qr, original path)
  * cholesky    : compiled orthogonalize_lowrank (shifted CholeskyQR3, fullgraph)
  * full NS     : orthogonalize on the full matrix (no projection), for reference
  * QR only     : eager torch.linalg.qr vs compiled _cholesky_qr on Y (..., max(m,n), k+p)
and runs torch._dynamo.explain on the real functions to count graph breaks.

The optimizer code is extracted from nanogpt/train_gpt.py with `ast` (importing the
script would initialize NCCL and start training).

Run (needs CUDA + triton):  python tests/bench_lowrank.py
"""
import ast
import os
import statistics
import sys
import time

import torch
import torch._dynamo as dynamo

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanogpt")
SRC = os.path.join(ROOT, "train_gpt.py")
FIRST, LAST = "polar_express_coeffs", "orthogonalize_lowrank_eager"


def load_optimizer_code():
    sys.path.insert(0, ROOT)
    import triton_kernels

    tree = ast.parse(open(SRC, encoding="utf-8").read())
    names = [getattr(n, "name", None) or ast.unparse(n.targets[0]) if isinstance(n, (ast.FunctionDef, ast.Assign)) else None
             for n in tree.body]
    lo, hi = names.index(FIRST), names.index(LAST)
    body = [n for n in tree.body[lo:hi + 1] if isinstance(n, (ast.FunctionDef, ast.Assign))]
    ns = {"torch": torch, "os": os, **{k: getattr(triton_kernels, k) for k in ("XXT", "XTX", "ba_plus_cAA")}}
    exec(compile(ast.Module(body=body, type_ignores=[]), SRC, "exec"), ns)
    return ns


def bench(fn, iters=50, warmup=10):
    """Host wall-clock per call (includes launch overhead and any host<->device syncs)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(times)


def main():
    assert torch.cuda.is_available()
    ns = load_optimizer_code()
    k, p, h = int(os.environ.get("RANK", 200)), 10, int(os.environ.get("POWER_ITER", 1))
    coeffs = ns["build_coeffs"](os.environ.get("SOLVER", "quintic_theoretical"), int(os.environ.get("NS_STEPS", 7)))
    print(f"{torch.cuda.get_device_name()} torch={torch.__version__} k={k} p={p} h={h} "
          f"q={len(coeffs)} CholeskyQR passes={ns['CHOLQR_PASSES']}")

    # --- graph breaks on the real functions -----------------------------------------
    M = torch.randn(24, 3072, 768, device="cuda", dtype=torch.bfloat16)
    raw = ns["orthogonalize_lowrank"]._torchdynamo_orig_callable
    ex = dynamo.explain(raw)(M, k, p, h, coeffs)
    print(f"explain(orthogonalize_lowrank, CholeskyQR3): graphs={ex.graph_count} breaks={ex.graph_break_count}")
    assert ex.graph_break_count == 0, ex.break_reasons
    dynamo.reset()
    hh = lambda M_: ns["_lift"](*ns["_randomized_project_impl"](M_, k, p, h, ns["_qr_reduced"]), coeffs)
    ex = dynamo.explain(hh)(M)
    print(f"explain(same pipeline, torch.linalg.qr):    graphs={ex.graph_count} breaks={ex.graph_break_count}"
          + "".join(f"\n    break: {r.reason.splitlines()[0][:110]}" for r in ex.break_reasons))
    dynamo.reset()

    # --- latency -----------------------------------------------------------------------
    shapes = [  # (label, batch, m, n)
        ("attn 1xGPU", 40, 768, 768), ("mlp 1xGPU", 24, 3072, 768),
        ("attn 4xGPU", 10, 768, 768), ("mlp 4xGPU", 6, 3072, 768),
    ]
    cqr = torch.compile(ns["_cholesky_qr"], dynamic=False, fullgraph=True)
    print(f"\n{'shape':>24} | {'householder':>11} {'cholesky':>9} {'speedup':>7} | {'full NS':>8} | "
          f"{'QR: HH':>7} {'CQR':>6}   (ms/call)")
    for label, b, m, n in shapes:
        M = torch.randn(b, m, n, device="cuda", dtype=torch.bfloat16)
        Y = torch.randn(b, max(m, n), k + p, device="cuda", dtype=torch.bfloat16)
        t_h = bench(lambda: ns["orthogonalize_lowrank_eager"](M, k, p, h, coeffs, "gaussian"))
        t_c = bench(lambda: ns["orthogonalize_lowrank"](M, k, p, h, coeffs))
        t_f = bench(lambda: ns["orthogonalize"](M, coeffs, split_baddbmm=m > 1024))
        q_h = bench(lambda: ns["_qr_reduced"](Y))
        q_c = bench(lambda: cqr(Y))
        print(f"{label + f' ({b},{m},{n})':>24} | {t_h:11.3f} {t_c:9.3f} {t_h / t_c:6.2f}x | {t_f:8.3f} | "
              f"{q_h:7.3f} {q_c:6.3f}")


if __name__ == "__main__":
    main()
