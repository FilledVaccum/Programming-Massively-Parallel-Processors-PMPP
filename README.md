# Programming Massively Parallel Processors (PMPP)

CUDA exercises following *Programming Massively Parallel Processors* by
Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj, organized by chapter.

## Structure

- [Chapter_1](Chapter_1/) — Introduction
- [Chapter_2](Chapter_2/) — Heterogeneous Data Parallel Computing
- [Chapter_3](Chapter_3/) — Multidimensional Grids and Data

Each chapter folder has its own README describing the exercises inside.

## How to think about kernel design

- [`COMPUTE_TARGET_DECISION_TREE.md`](COMPUTE_TARGET_DECISION_TREE.md) —
  the question that comes *before* writing any kernel: should this workload
  even run on a GPU, vs. a single CPU core, vs. a multi-core/SIMD CPU, vs. a
  distributed cluster, vs. an FPGA/ASIC/TPU? A structured decision tree from
  first principles.
- [`KERNEL_DESIGN_FIRST_PRINCIPLES.md`](KERNEL_DESIGN_FIRST_PRINCIPLES.md) —
  once you've decided "GPU," a from-first-principles template for
  approaching *any* kernel, simple or complex: the questions to ask, in
  order, from understanding the problem through mapping it onto
  threads/blocks, planning memory, writing a correct naive version, and
  optimizing only once you've profiled.

## Building

Each `.cu` file compiles standalone with `nvcc`, e.g.:

```
nvcc Chapter_2/hello_gpu.cu -o Chapter_2/hello_gpu
```
