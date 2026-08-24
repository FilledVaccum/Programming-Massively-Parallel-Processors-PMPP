# Chapter 2 — Heterogeneous Data Parallel Computing

Basics of CUDA C: kernel launches, and the thread/block/grid/warp hierarchy.

- `hello_gpu.cu` — minimal kernel launch, one thread printing its `threadIdx`.
- `block.cu` — block/grid/warp indexing: global thread ID computation,
  `blockIdx`/`blockDim`/`gridDim`, and warp/lane ID derivation.
- [`CUDA_EXECUTION_MODEL.md`](CUDA_EXECUTION_MODEL.md) — deep-dive reference
  on thread/warp/block/SM/GPC: what's software vs. hardware, why the hardware
  limits are what they are, and occupancy math.
- [`CUDA_SELF_TEST.md`](CUDA_SELF_TEST.md) — question-first companion with
  diagrams per category, covering fundamentals through applied
  debugging/spot-the-bug, with a full answer key.
- `diagrams/` — draw.io sources and exported SVGs visualizing the above.
