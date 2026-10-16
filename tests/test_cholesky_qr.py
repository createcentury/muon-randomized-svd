"""Numerical check: regularized CholeskyQR vs Householder QR for the randomized Muon projector.

train_gpt.py initializes CUDA/NCCL at import time, so the QR helpers are extracted from
its source with `ast` and exec'd here; the test exercises the exact code that trains.

Metrics (same Gaussian sketch Omega for both QR variants):
  * orthogonality      ||Q^T Q - I||_F
  * subspace distance  ||(I - Q_h Q_h^T) Q_c||_2  (= sin of largest principal angle)
  * alignment          <M, Q Z_q> / ||M||_*      (Assumption 2.1 / Prop. 2.3 quantity),
                       Z_q = polar factor of B = Q^T M  (exact, isolates the QR effect)
                       and Z_q = q-step NS (quintic) to match the training solver.

Run:  python tests/test_cholesky_qr.py            (CPU or CUDA; also checks torch.compile on CUDA)
"""
import ast
import os
import sys

import torch

SRC = os.path.join(os.path.dirname(__file__), "..", "nanogpt", "train_gpt.py")
WANTED = {"_qr_reduced", "_cholesky_qr", "CHOLQR_SHIFT", "CHOLQR_PASSES"}


def load_helpers():
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in WANTED for t in node.targets
        ):
            nodes.append(node)
    ns = {"torch": torch, "os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SRC, "exec"), ns)
    missing = WANTED - ns.keys()
    assert not missing, f"not found in train_gpt.py: {missing}"
    return ns


H = load_helpers()
_qr_reduced, _cholesky_qr = H["_qr_reduced"], H["_cholesky_qr"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUINTIC = (3.4445, -4.7750, 2.0315)


def make_M(m, n, decay, batch=(), seed=0):
    """Random matrix with power-law spectrum sigma_i = i^-decay, scaled like an unnormalized
    momentum buffer (~1/(1-beta) = 30x), in bf16."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(*batch, m, k, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(*batch, n, k, generator=g, dtype=torch.float64))
    s = torch.arange(1, k + 1, dtype=torch.float64).pow(-decay) * 30.0
    M = (U * s) @ V.mT
    return M.to(DEVICE, torch.bfloat16)


def sketch(M, r, h, seed):
    """Same Y as randomized_project (bf16 matmuls); returns Y and the tall flag."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    tall = M.size(-2) > M.size(-1)
    if tall:
        Om = torch.randn((*M.shape[:-2], M.size(-1), r), generator=g, device=DEVICE, dtype=M.dtype)
        Y = M @ Om
        for _ in range(h):
            Y = M @ (M.mT @ Y)
    else:
        Om = torch.randn((*M.shape[:-2], M.size(-2), r), generator=g, device=DEVICE, dtype=M.dtype)
        Y = M.mT @ Om
        for _ in range(h):
            Y = M.mT @ (M @ Y)
    return Y, tall


def polar(B):
    U, _, Vh = torch.linalg.svd(B.double(), full_matrices=False)
    return U @ Vh


def ns_quintic(B, steps=5):
    X = B.double()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    wide = X.size(-2) <= X.size(-1)
    if not wide:
        X = X.mT
    a, b, c = QUINTIC
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X if wide else X.mT


def lowrank_output(M, Q, tall, zfn):
    Md, Qd = M.double(), Q.double()
    if tall:
        return Qd @ zfn(Qd.mT @ Md)
    return zfn(Md @ Qd) @ Qd.mT


def metrics(M, r, h, seed):
    Y, tall = sketch(M, r, h, seed)
    Qh, Qc = _qr_reduced(Y), _cholesky_qr(Y)
    assert Qc.dtype == Y.dtype == torch.bfloat16, Qc.dtype
    assert torch.isfinite(Qc).all(), "CholeskyQR produced non-finite values"
    eye = torch.eye(r, device=DEVICE, dtype=torch.float64)
    orth = lambda Q: torch.linalg.matrix_norm(Q.double().mT @ Q.double() - eye).max().item()
    Qh64, Qc64 = Qh.double(), Qc.double()
    # orthonormalize in fp64 before measuring principal angles (removes bf16 rounding of Q)
    Ph, _ = torch.linalg.qr(Qh64)
    Pc, _ = torch.linalg.qr(Qc64)
    sub = torch.linalg.matrix_norm(Pc - Ph @ (Ph.mT @ Pc), ord=2).max().item()
    nuc = torch.linalg.matrix_norm(M.double(), ord="nuc")
    out = {"orth_h": orth(Qh), "orth_c": orth(Qc), "subspace": sub}
    for name, zfn in (("polar", polar), ("ns5", ns_quintic)):
        ah = ((M.double() * lowrank_output(M, Qh, tall, zfn)).sum((-2, -1)) / nuc)
        ac = ((M.double() * lowrank_output(M, Qc, tall, zfn)).sum((-2, -1)) / nuc)
        out[f"align_{name}_h"] = ah.min().item()
        out[f"align_{name}_c"] = ac.min().item()
        out[f"align_{name}_reldiff"] = ((ac - ah).abs() / ah.abs()).max().item()
    return out


CASES = [
    # (m, n, batch, decay)     shapes from the NanoGPT banks (d=768, mlp 4d), plus stress spectra
    (768, 768, (), 0.5),
    (3072, 768, (), 0.5),
    (768, 3072, (), 0.5),
    (3072, 768, (4,), 1.0),
    (768, 768, (4,), 2.0),       # fast decay: Y very ill-conditioned after power iteration
    (3072, 768, (), 3.0),        # extreme decay: sigma_200/sigma_1 ~ 1e-7
]
TOL = {"orth_c": 5e-2, "subspace_by_h": {0: 5e-2, 1: 5e-2, 2: 5e-2}, "align_reldiff": 1e-2}


def main():
    k, p = 200, 10
    r = k + p
    failures = []
    print(f"device={DEVICE} torch={torch.__version__} shift={H['CHOLQR_SHIFT']} passes={H['CHOLQR_PASSES']}")
    hdr = f"{'shape':>18} {'dec':>4} {'h':>2} | {'orth_HH':>8} {'orth_CQR':>8} | {'subspace':>8} | " \
          f"{'align_pol H/C':>15} {'rel':>8} | {'align_NS5 H/C':>15} {'rel':>8}"
    print(hdr); print("-" * len(hdr))
    for m, n, batch, decay in CASES:
        M = make_M(m, n, decay, batch)
        for h in (0, 1, 2):
            o = metrics(M, r, h, seed=1234 + h)
            shape = "x".join(map(str, (*batch, m, n)))
            print(f"{shape:>18} {decay:>4} {h:>2} | {o['orth_h']:8.1e} {o['orth_c']:8.1e} | {o['subspace']:8.1e} | "
                  f"{o['align_polar_h']:.4f}/{o['align_polar_c']:.4f} {o['align_polar_reldiff']:8.1e} | "
                  f"{o['align_ns5_h']:.4f}/{o['align_ns5_c']:.4f} {o['align_ns5_reldiff']:8.1e}")
            # Orthogonality is limited by the bf16 cast of Q (both variants), so compare to Householder.
            if o["orth_c"] > max(TOL["orth_c"], 2 * o["orth_h"]):
                failures.append((shape, decay, h, "orth", o["orth_c"]))
            # Subspace is only well defined when the r-th singular gap is resolvable in bf16;
            # alignment is the quantity the convergence theory needs, so it is the hard check.
            for z in ("polar", "ns5"):
                if o[f"align_{z}_reldiff"] > TOL["align_reldiff"]:
                    failures.append((shape, decay, h, f"align_{z}", o[f"align_{z}_reldiff"]))

    # Degenerate inputs: zero matrix and exactly rank-deficient Y must stay finite.
    Z = torch.zeros(1024, 210, device=DEVICE, dtype=torch.bfloat16)
    assert torch.isfinite(_cholesky_qr(Z)).all(), "zero Y -> non-finite"
    Y = make_M(1024, 64, 0.0)                     # rank 64 < r
    Yd = torch.cat([Y, Y, Y, Y[:, :18]], dim=-1)  # (1024, 210), rank 64
    assert torch.isfinite(_cholesky_qr(Yd)).all(), "rank-deficient Y -> non-finite"
    print("degenerate inputs (zero, rank-deficient): finite OK")

    if DEVICE == "cuda":
        check_compile()

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  ", f)
        sys.exit(1)
    print("\nPASS")


def check_compile():
    """fullgraph compile of the CholeskyQR projector must have zero graph breaks."""
    import torch._dynamo as dynamo

    def proj(M, Om):
        Y = M @ Om
        Y = M @ (M.mT @ Y)
        Q = _cholesky_qr(Y)
        return Q, Q.mT @ M

    M = make_M(3072, 768, 1.0)
    Om = torch.randn(768, 210, device=DEVICE, dtype=torch.bfloat16)
    ex = dynamo.explain(proj)(M, Om)
    print(f"dynamo.explain(_cholesky_qr projector): graphs={ex.graph_count} breaks={ex.graph_break_count}")
    assert ex.graph_break_count == 0 and ex.graph_count == 1, ex.break_reasons
    dynamo.reset()
    Qc, _ = torch.compile(proj, dynamic=False, fullgraph=True)(M, Om)
    Qe, _ = proj(M, Om)
    err = (Qc.float() - Qe.float()).abs().max().item()
    print(f"compiled vs eager CholeskyQR max|dQ| = {err:.2e}")
    assert err < 5e-2, err


if __name__ == "__main__":
    main()
