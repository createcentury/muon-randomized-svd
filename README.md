# Muon with Nesterov Momentum: Heavy-Tailed Noise and (Randomized) Inexact Polar Decomposition

> **Hardware Profiling & Optimization**: For GPU bottleneck analysis on NVIDIA A100 and benchmarks of the numerically stable Shifted CholeskyQR3 kernel, see [BENCHMARKS_A100.md](./BENCHMARKS_A100.md).

Code for the paper *Muon with Nesterov Momentum: Heavy-Tailed Noise and (Randomized) Inexact Polar Decomposition*, which develops a convergence theory for Muon with Nesterov momentum and inexact polar decomposition in non-convex matrix optimization under heavy-tailed noise. Before each Newton–Schulz iteration, the momentum buffer is projected onto a low-rank sketch, so the orthogonalization runs on a rank-`s` compressed matrix and is then lifted back to the original space. We validate this scheme empirically on CIFAR-10 and NanoGPT.

## Layout

- [cifar10/](cifar10/) — CIFAR-10 experiments, built on Keller Jordan's `airbench94` 94% speedrun. The main trainer is [cifar10/airbench94_muon.py](cifar10/airbench94_muon.py), which embeds `randomized_project` into the Muon update and supports four Newton–Schulz solvers (`cubic_ns_theoretical`, `quintic_ns_theoretical`, `quintic_ns_empirical`, `polar_express`).
- [nanogpt/](nanogpt/) — Modded-NanoGPT speedrun fork with the same randomized Muon plugged into [nanogpt/train_gpt.py](nanogpt/train_gpt.py). 

## Running

See the per-subproject READMEs for setup and launch instructions:

- [cifar10/README.md](cifar10/README.md)
- [nanogpt/README.md](nanogpt/README.md)
