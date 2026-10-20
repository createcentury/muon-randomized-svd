# Randomized Low-Rank Muon on NVIDIA A100: Profiling and a Compiled Shifted CholeskyQR3 Kernel

This report covers the orthonormalization step `Q = orth(Y)` inside the randomized low-rank
polar decomposition (Algorithm 1 of [arXiv:2605.06884](https://arxiv.org/abs/2605.06884)). We
profile it on an NVIDIA A100 and replace it with a compiled, numerically regularized
CholeskyQR. All numbers below were measured on branch `cholesky-qr`
(commit `88c2f2d`).

## 1. Background

### 1.1 Inexact polar condition (Assumption 2.1)

The convergence analysis allows the exact polar factor to be replaced by an inexact operator
$\mathcal{T}$, provided there exist $`\gamma_k \in [0,1)`$ and $`\nu_k \ge 0`$ such that

```math
\mathbb{E}\left[\langle M_k, \mathcal{T}(M_k)\rangle \mid M_k\right] \ge (1-\gamma_k)\,\|M_k\|_*,
\qquad
\mathbb{E}\left[\|\mathcal{T}(M_k)\|_{\mathrm{op}}^2 \mid M_k\right] \le (1+\nu_k)^2 .
```

### 1.2 Randomized lifted polar factor (Algorithm 1, Proposition 2.3)

Algorithm 1 takes a Gaussian sketch $\Omega \in \mathbb{R}^{n\times \ell}$, $\ell = s + p$,
and proceeds in five steps:

1. Form $Y = (MM^\top)^h M\Omega$.
2. Orthonormalize: $Q = \mathrm{orth}(Y)$.
3. Compress: $B = Q^\top M$.
4. Run $q$ Newton–Schulz steps from $`Z_0 = B/\delta`$.
5. Lift back: return $`\mathcal{T}_{h,q}(M;\Omega) = Q Z_q`$.

For any $`\delta \ge \|M\|_{\mathrm{op}}`$, Proposition 2.3 gives

```math
\mathbb{E}_\Omega\left[\langle M, \mathcal{T}_{h,q}(M;\Omega)\rangle\right] \ge
\frac{1}{\delta}\left[\sum_{j=1}^{s}\sigma_j^2 - \frac{s}{p-1}\,\varrho_s^{4h}\sum_{j>s}\sigma_j^2\right]_+,
\qquad \varrho_s = \sigma_{s+1}/\sigma_s,
```

and $`\|\mathcal{T}_{h,q}(M;\Omega)\|_{\mathrm{op}} \le 1`$ almost surely.

The operator-norm bound uses the fact that $Q$ has orthonormal columns:
$`\|QZ_q\|_{\mathrm{op}} \le \|Q\|_{\mathrm{op}}\|Z_q\|_{\mathrm{op}}`$. Any replacement for
`orth` therefore has to (i) span the same subspace as $Y$ and (ii) produce columns that are
orthonormal to working precision. If $`\|Q^\top Q - I\|_2 = \epsilon`$, then
$`\|Q\|_{\mathrm{op}} \le \sqrt{1+\epsilon}`$, and the bound becomes $`\|\mathcal{T}\|_{\mathrm{op}} \le \sqrt{1+\epsilon}`$.

In the repository, $s$ = `--rank`, $p$ = `--oversampling`, $h$ = `--power-iters`, and
$q$ = `--orth-steps`. All experiments here use $s=200$, $p=10$ ($\ell = 210$), $h=1$, $q=7$,
and the `quintic_theoretical` solver.

## 2. Bottleneck analysis

The original implementation computes `orth` with
`torch.linalg.qr(Y.float(), mode="reduced")` (Householder QR via cuSOLVER) and runs the
projector eagerly.

1. **Householder QR dominates the step.** On the NanoGPT parameter banks, batched QR of
   $Y \in \mathbb{R}^{b\times m\times 210}$ takes **50–61 ms** for the single-GPU shapes and
   **12–15 ms** for the 4-GPU shapes (torch 2.7.1). That is more than the entire full-rank
   Newton–Schulz orthogonalization it is meant to accelerate. §2.1 analyzes where this time
   goes.
2. **No graph break was observed.** The original motivation for this work assumed that
   `torch.linalg.qr` forces a graph break under `torch.compile(fullgraph=True)`. We did not
   observe this. `torch._dynamo.explain` on the original projector pipeline reports
   `graphs=1, breaks=0` on both torch 2.7.1 (the version pinned in `requirements.txt`) and
   torch 2.11. Compiling the Householder pipeline does not change its latency either: 53.3 ms
   vs. 53.4 ms eager for (40, 768, 768) on torch 2.11. The speedup reported below therefore
   comes from replacing the QR algorithm, not from removing a graph break.

### 2.1 Root cause analysis of `torch.linalg.qr`

Script: `tests/profile_qr_root_cause.py`. Setup: A100-SXM4-40GB, torch 2.7.1+cu126, FP32
$Y$, default linalg backend (cuSOLVER). The script measures three things:

- **Timing** (10 warm-up and 50 timed iterations, medians):
  - host time until `torch.linalg.qr` returns;
  - GPU time between `torch.cuda.Event`s recorded around the call;
  - wall-clock time including synchronization.
- **Profiler capture** of one call with `torch.profiler` (CPU and CUDA activities).
- **Backend comparison** between cuSOLVER and MAGMA.

| | $Y \in \mathbb{R}^{24\times 3072\times 210}$ | $Y \in \mathbb{R}^{40\times 768\times 210}$ |
| :--- | :--- | :--- |
| Host time until return | 4.3 ms | 44.0 ms |
| GPU time (CUDA events) | 60.7 ms | 50.4 ms |
| Wall-clock incl. sync | 60.8 ms | 50.4 ms |
| Kernels launched per QR call | 748 (31.2 per matrix) | 1,241 (31.0 per matrix) |
| Dominant kernel | `geqr2_gmem_domino` ×24, 2.28 ms each | `geqr2_smem_domino_fast` ×40, 1.07 ms each |
| Share of GPU time in `geqr2` | 91.1% | 86.5% |
| Sync / D2H copies issued by QR | 0 | 0 |
| `Command Buffer Full` stalls | none | 34, 33.2 ms total |
| MAGMA backend (GPU time) | 415.5 ms | 238.4 ms |

1. **GPU-kernel-bound, not host-synchronization-bound.** For both shapes, wall-clock time
   equals GPU time, and the call issues no blocking runtime API. The only
   `cudaDeviceSynchronize` calls in the capture also appear in a capture that contains only
   the script's own `torch.cuda.synchronize()`. For (24, 3072, 210), the host gets control
   back after 4.3 ms, and the remaining 56 ms is GPU execution.

   For (40, 768, 210), the host is held for 44 ms, but this is launch-queue back-pressure, not
   synchronization. A single QR call enqueues 1,241 kernels, about 31 per matrix. That fills
   the CUDA command buffer 34 times, and `cudaLaunchKernel` blocks for 33.2 ms in total
   waiting for the GPU to drain. The host stall is a consequence of slow GPU execution, not
   its cause.
2. **Sequential per-matrix panel factorization.** Although `torch.linalg.qr` accepts a batched
   input, the cuSOLVER path does not parallelize across the batch. The profile shows one
   Householder panel-factorization kernel (`geqr2_*`) per matrix (24 and 40 invocations),
   each surrounded by about 30 small kernels: blocked-reflector setup (`larft`), small SGEMMs,
   copies, and info checks. All of these run one matrix after another. The `geqr2` kernels
   alone account for 86–91% of GPU time, and the SGEMM-based trailing updates take only a few
   percent.
3. **Hardware under-utilization (estimate).** Counting $\approx 2mn^2 - \tfrac{2}{3}n^3$ flops
   for `geqrf` plus a comparable amount for forming $Q$ (`orgqr`) gives $\approx 0.5$ GFLOP per
   $3072\times 210$ matrix. That is executed in $\approx 2.5$ ms, an effective throughput of
   about 0.2 TFLOP/s, or about 1% of the A100's 19.5 TFLOP/s FP32 peak. This figure is derived
   from the flop count and the measured time, not from hardware counters.

   CholeskyQR3 avoids the per-matrix panel factorization. Its work consists of batched GEMMs
   over the whole bank (Gram matrix and application of $L^{-1}$) plus a $210\times 210$
   Cholesky factorization and triangular inverse per matrix. This reduces QR time from
   50–61 ms to 3.6–5.5 ms (§5). CholeskyQR3 is not free of overheads either: its remaining
   time is dominated by FP32 SIMT GEMMs, and at small batch sizes by per-matrix
   cuSOLVER/cuBLAS launches (§5).
4. **Backend comparison.** Forcing the MAGMA backend
   (`torch.backends.cuda.preferred_linalg_library("magma")`) makes the QR 4.7–6.9× slower
   (415.5 ms and 238.4 ms), and the host is blocked for nearly the entire duration. The
   slowdown is therefore not specific to cuSOLVER: both Householder QR backends available
   through PyTorch perform poorly on this tall-skinny batched workload. We did not evaluate
   dedicated batched Householder implementations outside PyTorch, so this does not establish
   a limit of the Householder algorithm itself.

## 3. Method: compiled Shifted CholeskyQR3

`_cholesky_qr` in `nanogpt/train_gpt.py` replaces the Householder QR in the Gaussian projector.
Each of three passes does the following, in FP32:

```math
Q \leftarrow Q D^{-1}\ (\text{column scaling}),\quad
G = Q^\top Q + sI,\quad
L = \mathrm{chol}(G),\quad
Q \leftarrow Q\,(L^{-1})^\top .
```

- **Column scaling.** Each column is normalized before the Gram product. This leaves
  $\mathrm{span}(Q)$ unchanged, makes the shift scale-invariant, and prevents FP32 overflow
  in $G$. Overflow is a real risk because the momentum buffer is unnormalized and $Y$ scales
  like $\sigma^{2h+1}$.
- **Diagonal shift.** Pass 1 uses $s = 10^{-6}\,\mathrm{tr}(G)$, which keeps $G$ positive
  definite for $\kappa(Y)$ up to about $5\times10^6$ in our tests. Passes 2–3 are plain
  CholeskyQR (CholeskyQR2), with only an $O(u)$ safeguard shift $s = (u/\ell)\,\mathrm{tr}(G)$.
  This restores orthogonality to $O(u)$.
- **No TRSM.** Because $L$ is only $210\times 210$, $L^{-1}$ is formed with a small
  triangular solve against $I$ and then applied as a single batched GEMM. This avoids a
  triangular solve over all $m$ rows.
- **Sync-free failure handling.** `torch.linalg.cholesky_ex` is used instead of
  `torch.linalg.cholesky`, because the latter checks `info` on the host. If a pass fails,
  for example when $Y$ is exactly rank-deficient, that pass is skipped per matrix via
  `torch.where`.
- **Compilation.** `randomized_project` and `orthogonalize_lowrank` are wrapped in
  `torch.compile(dynamic=False, fullgraph=True)`. `torch._dynamo.explain` on the real
  `orthogonalize_lowrank`, including the Triton Newton–Schulz kernels, reports
  `graphs=1, breaks=0`.

Two paths still use the original eager Householder implementation: the Kaczmarz projector,
and runs with `RANDOMIZED_QR=householder` (kept as a baseline).

A single shifted pass followed by only one CholeskyQR pass reached a worst-case
$`\|Q^\top Q - I\|_F`$ of $1.8\times10^{-4}$. We use three passes because that brings the error
below $10^{-4}$.

## 4. Numerical validation

`tests/test_cholesky_qr.py` compares Shifted CholeskyQR3 with Householder QR. Both receive
the same sketch $\Omega$, and the test covers 18 cases:

- **Shapes:** (768, 768), (3072, 768), (768, 3072), and batched (4, 3072, 768) and (4, 768, 768).
- **Spectra:** power-law, $`\sigma_i = i^{-d}`$ with $`d \in \{0.5, 1, 2, 3\}`$.
- **Power iterations:** $`h \in \{0, 1, 2\}`$.
- **Condition numbers:** $\kappa(Y)$ up to about $5\times10^6$.

**Results** (A100, torch 2.7.1, maxima over all 18 cases):

| Quantity | Result | Criterion |
| :--- | :--- | :--- |
| $`\lVert Q^\top Q - I\rVert_F`$, CholeskyQR3, FP32 factor | $1.2\times10^{-5}$ | $\le 10^{-4}$ |
| $`\lVert Q^\top Q - I\rVert_F`$, bf16 output (Householder / CholeskyQR3) | $2.3\times10^{-2}$ / $2.3\times10^{-2}$ | CholeskyQR3 $\le 1.5\times$ Householder |
| Relative difference of $`\langle M, QZ_q\rangle/\lVert M\rVert_*`$ vs. Householder ($`Z_q`$ = exact polar factor and 5-step NS) | $6.5\times10^{-6}$ | $\le 10^{-2}$ |
| Degenerate inputs (zero matrix, exactly rank-deficient $Y$) | finite output | finite |

The orthogonality of the bf16 $Q$ consumed by the optimizer is limited by bf16 rounding,
which is about $10^{-2}$. This limit applies equally to the original Householder path, so
CholeskyQR3 does not loosen the effective $`(1+\nu_k)`$ bound relative to the original
implementation.

The alignment check is a comparison against the original implementation, not a direct
verification of Assumption 2.1. Proposition 2.3 notes that establishing Assumption 2.1
additionally requires $s, h$ such that the bracketed term is positive. Within this tolerance,
CholeskyQR3 preserves whatever alignment the Householder path achieves.

## 5. Micro-benchmark

Script: `tests/bench_lowrank.py`. Setup: A100-SXM4-40GB, torch 2.7.1+cu126, $s=200$, $p=10$,
$h=1$, $q=7$. Latency is measured as host wall-clock per call (the median of 5 × 50 calls
after warm-up), so launch overhead and any host–device synchronization are included.

| Shape | eager Householder | compiled CholeskyQR3 | Speedup | Full NS (no projection) |
| :--- | :--- | :--- | :--- | :--- |
| attn (40, 768, 768) | 54.4 ms | 7.7 ms | 7.1x | 5.5 ms |
| mlp (24, 3072, 768) | 62.7 ms | 7.4 ms | 8.5x | 11.5 ms |
| attn (10, 768, 768) | 14.0 ms | 3.8 ms | 3.7x | 1.5 ms |
| mlp (6, 3072, 768) | 16.3 ms | 5.8 ms | 2.8x | 4.0 ms |

The QR step alone takes 50.3 / 60.8 / 12.5 / 15.2 ms with Householder and
3.6 / 5.5 / 2.3 / 4.8 ms with CholeskyQR3, in the same row order.

- **mlp (24, 3072, 768): the low-rank path now beats full-rank Newton–Schulz** (7.4 ms vs.
  11.5 ms). With Householder QR, the low-rank path was 5.5× slower than full NS on this shape.
- **For the 768 × 768 attention banks and the small 4-GPU batches, full NS is still faster.**
  With $\ell = 210$ against $n = 768$, the FLOP reduction is modest. Profiling (torch 2.11)
  shows that the remaining CholeskyQR3 time is mostly FP32 GEMMs, i.e. forming $G$ and
  applying $L^{-1}$; TF32 cannot be used here without breaking the $10^{-4}$ orthogonality
  target. At small batch sizes, much of the time also goes to per-matrix
  cuSOLVER/cuBLAS launches.
- **Replacing TRSM with GEMM helped only partly.** It reduced the QR time for
  (24, 3072, 210) from 6.7 ms to 5.5 ms and for (40, 768, 210) from 3.8 ms to 3.6 ms
  (torch 2.11).

## 6. End-to-end NanoGPT smoke test

**Setup:**
- A100-SXM4-40GB, single GPU, torch 2.7.1+cu126, xformers 0.0.31.post1 attention backend.
- 100 steps (`NUM_ITERATIONS=90`, `NUM_EXT_ITERATIONS=10`), `--randomized True --rank 200`.
- The other hyperparameters are those of the README's "Muon + Nesterov + randomized Gaussian
  projection" command.
- `DISABLE_FP8=1` (the A100 has no FP8) and `GRAD_ACCUM_STEPS=16` (the non-FP8 loss
  materializes FP32 logits). The total batch size is unchanged.
- Both runs use the same seed, data, and configuration and differ only in `RANDOMIZED_QR`.

| Metric | Householder (Original) | CholeskyQR3 (Optimized) | Delta |
| :--- | :--- | :--- | :--- |
| step_avg (step 100) | 2886.4 ms | 2790.4 ms | -96.0 ms (-3.3%) |
| val_loss @ 25 | 6.4760 | 6.4763 | +0.0003 |
| val_loss @ 50 | 5.6248 | 5.6233 | -0.0015 |
| val_loss @ 75 | 5.0763 | 5.0752 | -0.0011 |
| val_loss @ 100 | 4.8200 | 4.8189 | -0.0011 |
| peak memory | 32.4 GB | 32.4 GB | 0.0 GB |

- **The per-step saving is constant.** The gap in `step_avg` is 95.6–96.4 ms at every
  checkpoint (steps 10, 25, 50, 75, and 100), even though the batch size grows across the
  three training stages and `step_avg` grows from 1.75 s to 2.89 s. This matches the
  micro-benchmark difference for the two single-GPU banks: (54.4 + 62.7) − (7.7 + 7.4) ≈ 102 ms.
  The saving therefore comes from the optimizer step.
- **Validation loss is equivalent.** The largest difference is 0.0015. The trajectories are
  not bitwise identical because the Gaussian sketch $\Omega$ is drawn by different RNG paths
  in eager and compiled code.
- **Training loss was not compared.** It is not printed to stdout by default.

**Outlook for H100 (not measured).** On A100 without FP8 or FlashAttention-3, an average step
takes about 2.8–2.9 s and is dominated by the forward and backward passes, so the 96 ms saving
is only 3.3% of the step. The
optimizer saving does not depend on the forward/backward pass. On an H100 cluster with FP8
matmuls and FlashAttention-3, forward and backward shrink, so the saving should account for
a larger fraction of each step. Two things change the absolute saving on that setup: with 4
GPUs, each rank holds the smaller 4-GPU shapes (rows 3–4 of §5), and the H100 has a different
cuSOLVER and GEMM throughput. We have not measured it there.

## 7. Reproducibility

All commands are run from the repository root. `tests/` requires CUDA and Triton.

```bash
# 1. Numerical equivalence vs. Householder QR; on CUDA this also checks
#    fullgraph compilation and zero graph breaks.
python tests/test_cholesky_qr.py

# 2. Micro-benchmark plus dynamo.explain on the real orthogonalize_lowrank.
python tests/bench_lowrank.py

# 2b. Root cause of the Householder QR cost (§2.1).
python tests/profile_qr_root_cause.py

# 3. NanoGPT smoke test on a single A100 (run once per QR variant).
python nanogpt/data/cached_fineweb10B.py 2
cd nanogpt
for QR in householder cholesky; do
  RANDOMIZED_QR=$QR GRAD_ACCUM_STEPS=16 VAL_BATCH_SIZE=262144 VAL_TOKENS=2097152 \
  DISABLE_FA3=1 DISABLE_FP8=1 NUM_ITERATIONS=90 NUM_EXT_ITERATIONS=10 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --optimizer-mode muon --num-trials 1 --log-every 25 \
    --muon-nesterov True --muon-lr 0.032463575100400426 --muon-momentum 0.9664542176931918 \
    --inexact-solver quintic_theoretical --orth-steps 7 \
    --randomized True --rank 200 --oversampling 10 --power-iters 1 --seq-len 2048 \
    --sgd-lr 0.001 --sgd-momentum 0.95 --sgd-nesterov True --adamw-lr 0.001 \
    > smoke_$QR.log 2>&1
done
```

**Environment variables added on this branch:**

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `RANDOMIZED_QR` | `cholesky` | `cholesky`: compiled CholeskyQR3 path. `householder`: original eager path. |
| `CHOLQR_SHIFT` | `1e-6` | Relative shift for pass 1 ($`s = \text{shift}\cdot\mathrm{tr}(G)`$). |
| `CHOLQR_PASSES` | `3` | Total number of CholeskyQR passes. |

**Software environment:** A100-SXM4-40GB (Google Colab), driver 580.82, Python 3.13,
torch 2.7.1+cu126, triton 3.3.1, xformers 0.0.31.post1. `requirements.txt` pins
torch 2.7.1+cu118; cu126 was used because it is the CUDA variant for which a matching
xformers wheel was available.
