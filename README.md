# Programming Massively Parallel Processors (PMPP)

CUDA exercises following *Programming Massively Parallel Processors* by
Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj, organized by chapter.

## Structure

- [Chapter_1](Chapter_1/) — Introduction
- [Chapter_2](Chapter_2/) — Heterogeneous Data Parallel Computing
- [Chapter_3](Chapter_3/) — Multidimensional Grids and Data

Each chapter folder has its own README describing the exercises inside.

## Building

Each `.cu` file compiles standalone with `nvcc`, e.g.:

```
nvcc Chapter_2/hello_gpu.cu -o Chapter_2/hello_gpu
```
