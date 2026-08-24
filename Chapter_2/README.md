# Chapter 2 — Heterogeneous Data Parallel Computing

Basics of CUDA C: kernel launches, and the thread/block/grid/warp hierarchy.

- `hello_gpu.cu` — minimal kernel launch, one thread printing its `threadIdx`.
- `block.cu` — block/grid/warp indexing: global thread ID computation,
  `blockIdx`/`blockDim`/`gridDim`, and warp/lane ID derivation.
