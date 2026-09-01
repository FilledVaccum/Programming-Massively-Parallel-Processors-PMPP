# CUDA Chapter 3 — Complete Reference Guide

## Patterns, Thought Processes, Templates & Discoveries

---

## 1. The 5-Step Thought Process (for ANY CUDA kernel)

Use this **every time** before writing a kernel:

| Step | Question | What you decide |
| --- | --- | --- |
| 1 | What is my **DATA**? | Shape, size, how stored in memory (flat 1D in VRAM) |
| 2 | What does **ONE thread** do? | What it reads, computes, writes |
| 3 | How do I **MAP** threads → data? | The formula (col, row, index) — **one line of code, not a paragraph** |
| 4 | **Launch configuration**? | blockSize + gridSize (dim3 lines) |
| 5 | **Boundary checks** needed? | Filter excess threads |

---

## 2. The "Start From the OUTPUT" Decision Tree

```
Look at the OUTPUT
  │
  ├── How many inputs does ONE output value need?
  │     │
  │     ├── FEW (1–1000 inputs per output)
  │     │     → One thread = one output element
  │     │     → Thread loops through its inputs
  │     │     → This is Chapter 3 level (grayscale, blur, matmul)
  │     │
  │     └── MASSIVE (10,000+ inputs per output)
  │           → Many threads cooperate → partial results → combine
  │           → "Parallel Reduction" (Chapter 10+)
  │
  └── Multiple threads write to SAME output?
        → atomicAdd or private copies + merge (Chapter 9)

```

### Quick Decision Table:

| Problem | Output | Inputs/output | Strategy |
| --- | --- | --- | --- |
| Scale array | N values | 1 | One thread = one element |
| Grayscale | W×H pixels | 3 (R,G,B) | One thread = one pixel |
| Blur 3×3 | W×H pixels | 9 (neighbors) | One thread = one pixel |
| MatMul (M×N) | M×N values | K (dot product) | One thread = one element |
| Max per row | M values | N (whole row) | One thread = one row |
| Batch Norm | C values | N×H×W per channel | Reduction (advanced) |
| Histogram | 256 bins | Many pixels → same bin | Atomic (advanced) |

---

## 3. The Universal 2D Kernel Template

Every Chapter 3 kernel follows this **exact** structure:

```c
__global__ void kernel(type *input, type *output, int width, int height) {
    // 1. Who am I?
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Am I valid?
    if (row < height && col < width) {

        // 3. Where's my data?
        int index = row * width + col;

        // 4. Do my job (ONLY THIS PART CHANGES between kernels)
        output[index] = some_calculation(input, index);
    }
}

```

The **only thing that changes** between grayscale, blur, and matmul is step 4 — the calculation.

---

## 4. Launch Configuration Recipe

**Don't overthink this. It's always the same:**

### 2D problems (images, matrices):

```c
dim3 blockSize(16, 16);                                        // fixed default
dim3 gridSize((width + 15) / 16, (height + 15) / 16);        // ceil division
kernel<<<gridSize, blockSize>>>(input, output, width, height);

```

### 1D problems (arrays):

```c
dim3 blockSize(256);                          // fixed default
dim3 gridSize((N + 255) / 256);              // ceil division
kernel<<<gridSize, blockSize>>>(input, output, N);

```

### Why `+ 15` (or `+ blockSize - 1`)?

Integer ceiling division: rounds UP so you never miss data elements. Extra threads are filtered by the boundary check.

### Summary:

| What | Rule |
| --- | --- |
| Block size | Pick a fixed default: `(16,16)` for 2D, `(256)` for 1D |
| Grid size | `ceil(data_size / block_size)` = `(data + block - 1) / block` |
| Boundary check | `if (row < height && col < width)` — the safety net |

---

## 5. The dim3 Rule

```
dim3(x, y) → x = COLUMNS (horizontal), y = ROWS (vertical)

```

**This is the OPPOSITE of English "rows × columns."**

| Code | x (columns) | y (rows) | Total |
| --- | --- | --- | --- |
| `blockSize(4, 2)` | 4 cols | 2 rows | 8 threads/block |
| `gridSize(120, 68)` | 120 block-cols | 68 block-rows | 8,160 blocks |

**Memory trick:** `dim3` is a **coordinate system** — x is always horizontal.

**Common mistake:** Saying `dim3(4, 2)` = "4 rows, 2 columns" — WRONG. It's 4 columns, 2 rows.

---

## 6. Thread Identity Model

### A thread is a WORKER, not data, not an index.

| Thing | What it is | Where it lives |
| --- | --- | --- |
| **Thread** | A worker (runs instructions on a CUDA core) | CUDA core |
| **Index** | A number the worker COMPUTES (`i`, `row`, `col`) | Register |
| **Data** | Values sitting in VRAM at that address | VRAM/GDDR6 |

### The relationship:

```
Thread → computes index → uses index → accesses data in VRAM

```

**Thread ≠ data. Thread ≠ index. Thread = worker who USES an index to FIND data.**

---

## 7. Row-Major Linearization

### 2D → 1D:

```c
index = row * width + col

```

Why: `row * width` skips complete rows, `+ col` moves to position within that row.

### 1D → 2D (reverse):

```c
row = index / width
col = index % width

```

### Key rule: `width` = total columns in DATA (not blockDim.x)

---

## 8. RGB Pixel Indexing Pattern

```c
// Input array: 3 values per pixel (R, G, B packed sequentially)
int pixelIndex = row * width + col;        // which pixel
unsigned char R = input[pixelIndex * 3 + 0];   // ×3 because 3 channels!
unsigned char G = input[pixelIndex * 3 + 1];
unsigned char B = input[pixelIndex * 3 + 2];

// Output array: 1 value per pixel (grayscale)
output[pixelIndex] = gray;                  // no ×3 needed

```

---

## 9. CUDA Host-Side Pattern (main function)

Every CUDA program's `main()` follows this order:

```
1. Create data on CPU           → arrays in RAM
2. cudaMalloc                   → reserve empty space in GPU VRAM
3. cudaMemcpy (Host → Device)   → copy input data CPU → GPU (over PCIe)
4. dim3 + kernel<<<>>>          → CPU tells GPU to create threads and start
5. cudaDeviceSynchronize        → CPU waits for GPU to finish
6. cudaMemcpy (Device → Host)   → copy results GPU → CPU (over PCIe)
7. cudaFree                     → release GPU memory
8. Print/use results

```

---

## 10. Pointer Cheat Sheet

### Three symbols:

| Symbol | Meaning | When used |
| --- | --- | --- |
| `*` | "This is a pointer (holds an address)" | Declaring or receiving a parameter |
| `&` | "Give me the address OF this variable" | When a function needs to MODIFY your variable |
| nothing | "Here's the value stored in this variable" | Passing to a function that just reads it |

### The full lifecycle:

```c
unsigned char *deviceInput;              // Declare: blank piece of paper
cudaMalloc(&deviceInput, size);          // Fill: GPU writes address on your paper (&)
kernel<<<...>>>(deviceInput);            // Pass: hand the address to the kernel (no &)

```

```c
// Kernel receives it:
__global__ void kernel(unsigned char *input) {   // * = "this is a pointer"
    input[i] = ...;                               // access data at that address
}

```

### Key insight:

**Pointer is copied, DATA is shared.** Both the caller and the kernel hold the same address → point to the same VRAM → changes are visible to both.

### Why cudaMalloc needs `&`:

`cudaMalloc` must MODIFY the pointer (fill in the VRAM address). Without `&`, it gets a copy and your original stays empty.

### Why kernel does NOT need `&`:

The kernel just READS the address to access data. It never changes WHERE the pointer points.

---

## 11. Fast Kernel Checklist (priority order)

1. **Maximize parallelism** — launch 100K+ threads, keep all 58 SMs busy
2. **Unique output per thread** — no two threads writing to same address
3. **Minimize VRAM accesses** — read once → store in register → reuse
4. **Coalesce memory** — adjacent threads access adjacent addresses
5. **Balance work per thread** — enough threads to fill GPU, each with reasonable work
6. **Use fast memory** — registers (0 cycles) > shared mem (5) > L1 (30) > VRAM (400)

---

## 12. Software → Hardware Mapping

| Software (you write) | Hardware (silicon) |
| --- | --- |
| Grid | Entire GPU (all GPCs, all SMs) |
| Block | 1 SM (multiple blocks can share one SM) |
| Warp (32 threads) | 1 Warp Scheduler + 32 CUDA cores |
| Thread | 1 CUDA core (at execution time) |
| Local variable (`int i`) | Register (0 cycles, private per thread) |
| `__shared__` variable | Shared Memory (5 cycles, per block) |
| `data[i]` (pointer) | VRAM / GDDR6 (400 cycles, global) |
| `<<<gridSize, blockSize>>>` | GigaThread Engine (top-level scheduler) |

### L4 GPU Limits:

| Resource | Limit |
| --- | --- |
| SMs | 58 |
| CUDA cores per SM | 128 (total: 7,424) |
| Max threads per block | 1,024 |
| Max threads per SM | 1,536 |
| Max blocks per SM | 32 |
| Warp size | 32 (always) |
| Registers per SM | 65,536 |
| Shared memory per SM | 100 KB |
| VRAM | 23 GB GDDR6, ~300 GB/s |

---

## 13. Memory Hierarchy (slow → fast)

| Level | Latency | Bandwidth | Size (L4) | Scope | Persistence |
| --- | --- | --- | --- | --- | --- |
| Disk (SSD) | ~100,000 ns | ~3.5 GB/s | TBs | System | Permanent |
| CPU RAM | ~100 ns | ~50 GB/s | 16-512 GB | CPU | Until power off |
| **VRAM (GDDR6)** | **~400 cycles** | **~300 GB/s** | **23 GB** | **All SMs** | **Until cudaFree** |
| L2 Cache | ~200 cycles | ~2 TB/s | 48 MB | All SMs | Auto (cache) |
| L1 Cache | ~30 cycles | ~5 TB/s | 128 KB/SM | One SM | Auto (cache) |
| Shared Memory | ~5 cycles | ~10 TB/s | 100 KB/SM | One Block | Kernel lifetime |
| **Registers** | **0 cycles** | **∞ (on-chip)** | **256 KB/SM** | **One Thread** | **Thread lifetime** |

### Key insight: Compute (1 cycle) vs VRAM access (400 cycles). Memory is the bottleneck, not compute.

---

## 14. All Interconnects / Buses

| Interconnect | Connects | Bandwidth |
| --- | --- | --- |
| NVMe/SATA | Disk ↔ CPU RAM | 3.5–7 GB/s |
| DDR5 | CPU ↔ RAM | ~50-100 GB/s |
| **PCIe Gen4 x16** | **CPU RAM ↔ GPU VRAM** | **~32 GB/s** |
| NVLink (not on L4) | GPU ↔ GPU | 600-900 GB/s |
| NVSwitch (not on L4) | All GPUs ↔ All GPUs | ~1.8 TB/s |
| Memory Controllers | On-chip ↔ GDDR6 | ~300 GB/s |
| Crossbar/NoC | SMs ↔ L2 ↔ Mem Ctrl | multi-TB/s |

---

## 15. Discoveries From Experiments

| Experiment | What we found |
| --- | --- |
| `<<<1, 512>>>` | Output scrambled in chunks of 32 → **warps are real** |
| `<<<2, 16>>>` | threadIdx repeats 0–15 twice → **need blockIdx for global ID** |
| Remove `cudaDeviceSynchronize()` | No output → **kernel launch is async** |
| `<<<1, 1025>>>` | Silent failure → **need error checking (cudaGetLastError)** |
| Two host printfs + kernel | Host prints first → **CPU doesn't wait for GPU** |
| Variable work per warp | Light warp finishes first → **warps scheduled independently** |
| Block output order | Non-deterministic → **blocks dispatched to SMs in any order** |

---

## 16. nvcc Compilation Pipeline

```bash
nvcc -arch=sm_89 -o program program.cu

```

4 stages:

1. **Separation**: host code → gcc, device code → CUDA compiler
2. **PTX**: device code → virtual GPU assembly (portable)
3. **SASS**: PTX → actual machine code for sm_89 (Ada/L4)
4. **Linking**: host.o + device binary → fat binary executable

---

## 17. CUDA Built-in Variables

| Variable | Meaning |
| --- | --- |
| `threadIdx.x/y/z` | Thread's position WITHIN its block |
| `blockIdx.x/y/z` | Which block this thread belongs to |
| `blockDim.x/y/z` | Size of each block (threads per block) |
| `gridDim.x/y/z` | Size of the grid (blocks in grid) |

### The global ID formulas:

```c
// 1D:
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2D:
int col = blockIdx.x * blockDim.x + threadIdx.x;   // horizontal
int row = blockIdx.y * blockDim.y + threadIdx.y;   // vertical
int index = row * width + col;                      // linearization

```

---