"""Root-cause profiling of batched torch.linalg.qr (Householder) on CUDA.

Question: why does reduced QR of Y in R^(b, m, 210) (FP32) take 50-60 ms on A100?
  (a) the GPU kernels themselves are slow / poorly batched, or
  (b) the host blocks on CPU-GPU synchronization (cudaStreamSynchronize, D2H memcpy, ...).

Measurements per shape:
  1. Timing split (10 warm-up, 50 timed iterations, medians):
       host_return : perf_counter around the call only (no sync) -> time until the CPU gets control back
       gpu_event   : cuda.Event elapsed time between records placed around the call
       wall_sync   : perf_counter around call + synchronize
     If host_return ~ gpu_event, the call blocks the host (synchronous); if host_return << gpu_event,
     it is asynchronous and the cost is GPU execution.
  2. torch.profiler capture of one call: top CUDA kernels, kernel launches per call, and
     blocking runtime APIs (cudaStreamSynchronize / cudaDeviceSynchronize / cudaMemcpy*).
  3. Backend comparison: torch.backends.cuda.preferred_linalg_library cusolver vs magma.

Run:  python tests/profile_qr_root_cause.py
"""
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

SHAPES = [(24, 3072, 210), (40, 768, 210)]
WARMUP, ITERS = 10, 50
BLOCKING_APIS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
                 "cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpy2DAsync", "cudaStreamWaitEvent")


def qr(Y):
    return torch.linalg.qr(Y, mode="reduced")


def timing_split(Y):
    for _ in range(WARMUP):
        qr(Y)
    torch.cuda.synchronize()
    host, gpu, wall = [], [], []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        qr(Y)
        end.record()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        host.append((t1 - t0) * 1e3)
        wall.append((t2 - t0) * 1e3)
        gpu.append(start.elapsed_time(end))
    return statistics.median(host), statistics.median(gpu), statistics.median(wall)


PSEUDO_EVENTS = ("Unrecognized", "Command Buffer Full")  # profiler annotations, not kernels


def profile_once(Y, run_qr=True):
    qr(Y)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        if run_qr:
            qr(Y)
        torch.cuda.synchronize()  # the script's own sync; counted by the baseline capture
    return prof


def api_counts(prof):
    return {e.key: e.count for e in prof.key_averages() if e.key.startswith(BLOCKING_APIS)}


def summarize(prof, batch, baseline):
    events = prof.key_averages()
    print(events.table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=60))

    def dev_time(e):  # attribute name differs across torch versions
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)

    kernels = [e for e in events if dev_time(e) > 0 and e.key not in PSEUDO_EVENTS
               and not e.key.startswith(("aten::", "cuda", "Memcpy", "Memset"))]
    gpu_total = sum(dev_time(e) for e in kernels)
    n_kernels = sum(e.count for e in kernels)
    print(f"GPU kernels: {n_kernels} launches for one QR call of batch {batch} "
          f"({n_kernels / batch:.1f} per matrix), total kernel time {gpu_total / 1e3:.2f} ms")
    for e in sorted(kernels, key=dev_time, reverse=True)[:6]:
        print(f"   {dev_time(e) / 1e3:8.3f} ms  {100 * dev_time(e) / max(gpu_total, 1):5.1f}%  x{e.count:<5d} {e.key[:90]}")
    copies = [e for e in events if e.key.startswith(("Memcpy", "Memset")) and dev_time(e) > 0]
    for e in copies:
        print(f"   copy/memset: {e.key[:60]:60s} x{e.count:<5d} {dev_time(e) / 1e3:.3f} ms")

    print("Host-side blocking / runtime APIs (calls inside QR = capture - baseline sync-only capture):")
    found = False
    for e in events:
        if e.key.startswith(BLOCKING_APIS):
            inside = e.count - baseline.get(e.key, 0)
            found |= inside > 0
            print(f"   {e.key:28s} calls={e.count:<3d} baseline={baseline.get(e.key, 0):<3d} "
                  f"inside_qr={inside:<3d} cpu_total={e.cpu_time_total / 1e3:8.3f} ms")
    print(f"   -> blocking sync/memcpy issued by QR itself: {'YES' if found else 'none'}")
    full = [e for e in events if e.key == "Command Buffer Full"]
    if full:
        print(f"   Command Buffer Full: {full[0].count} times, host stalled {full[0].cpu_time_total / 1e3:.3f} ms "
              f"(launch queue full -> cudaLaunchKernel waits for the GPU to drain; back-pressure, not a sync)")
    launch = [e for e in events if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC")]
    for e in launch:
        print(f"   {e.key:28s} calls={e.count:<5d} cpu_total={e.cpu_time_total / 1e3:8.3f} ms")


def main():
    assert torch.cuda.is_available()
    print(f"{torch.cuda.get_device_name()}  torch={torch.__version__}  cuda={torch.version.cuda}")
    for b, m, r in SHAPES:
        Y = torch.randn(b, m, r, device="cuda", dtype=torch.float32)
        print("\n" + "=" * 100 + f"\nY = ({b}, {m}, {r}) fp32")

        torch.backends.cuda.preferred_linalg_library("default")
        host, gpu, wall = timing_split(Y)
        print(f"[1] timing (median of {ITERS}):  host_return={host:.3f} ms   gpu_event={gpu:.3f} ms   "
              f"wall_sync={wall:.3f} ms   host/gpu={host / gpu:.2f}")
        verdict = ("host is held for most of the GPU time (see [2]: sync API vs. launch-queue back-pressure)"
                   if host > 0.8 * gpu else "host returns early (asynchronous); cost is GPU execution")
        print(f"    -> {verdict}")

        print("[2] torch.profiler (1 call):")
        summarize(profile_once(Y), b, api_counts(profile_once(Y, run_qr=False)))

        print("[3] backend comparison (gpu_event / host_return, ms):")
        for lib in ("cusolver", "magma"):
            try:
                torch.backends.cuda.preferred_linalg_library(lib)
                h, g, _ = timing_split(Y)
                print(f"   {lib:9s} gpu={g:8.3f}  host={h:8.3f}")
            except Exception as ex:  # magma may be unavailable in some builds
                print(f"   {lib:9s} unavailable: {str(ex)[:80]}")
        torch.backends.cuda.preferred_linalg_library("default")


if __name__ == "__main__":
    main()
