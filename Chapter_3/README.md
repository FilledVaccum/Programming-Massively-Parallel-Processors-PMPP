# Chapter 3 — Multidimensional Grids and Data

2D thread/block indexing applied to image data.

- `grayscale.cu` — RGB-to-grayscale conversion kernel using a 2D grid of 2D
  blocks, mapping `(col, row)` to a flattened pixel offset.
