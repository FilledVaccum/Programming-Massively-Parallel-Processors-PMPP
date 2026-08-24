# The CUDA Execution Model — Thread, Warp, Block, SM, GPC, GPU

A from-first-principles reference for the question everyone learning CUDA asks:
*"What's the actual difference between a thread, a warp, a block, an SM, and a
GPC — and which of these do I even control?"*

This document answers that at three levels: **what each thing is**, **who
decides it (you vs. the hardware)**, and **why the numbers are what they are**
— the engineering tradeoffs baked into the silicon, not just the vocabulary.

---

## 0. The one-sentence mental model

> You write code for **one thread**. You organize threads into **blocks** and
> **grids** — that's the *software* side, and you control every number in it.
> The GPU hardware then chops your blocks into **warps** of 32 and runs them
> on **SMs** — that's the *hardware* side, and you control none of it.

---

## 1. Two separate worlds

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│           SOFTWARE               │     │           HARDWARE               │
│    (what you write in code)      │     │   (physical silicon on the die)  │
│                                   │     │                                   │
│   You choose the size and shape. │     │   Fixed by NVIDIA when the chip  │
│   Lives in your .cu file.        │     │   was designed. You cannot       │
│                                   │     │   change it, only query it.      │
│                                   │     │                                   │
│   • Grid                         │     │   • GPU (the whole chip)         │
│   • Block                        │     │   • GPC (cluster of SMs)         │
│   • Thread                       │     │   • SM  (a physical processor)   │
│                                   │     │   • CUDA Core (one ALU)          │
└──────────────────────────────────┘     └──────────────────────────────────┘
```

**Warp sits in between.** It's created by hardware (you never write "make a
warp"), but it directly shapes how you should think about your code (warp
divergence, coalesced memory access, `__shfl` intrinsics). Treat it as *"a
hardware execution unit that leaks into how you reason about software."*

---

## 2. The software side (you decide these)

### Thread
- **What it is:** one execution of your kernel function, operating on its own
  data.
- **You control:** what code it runs (the kernel body is written *per thread*).
- **Identified by:** `threadIdx.x/y/z` — its position *within its block*.
- **Owns:** private registers and local variables. `int x = 5;` inside a
  kernel means *every thread* gets its own independent copy of `x`.

### Block
- **What it is:** a group of threads *you* choose to bundle together.
- **You control:** how many threads (1–1024) and their shape (1D/2D/3D via `dim3`).
- **Identified by:** `blockIdx.x/y/z` — its position *within the grid*.
- **Special power:** threads in the same block can cooperate — share data
  through `__shared__` memory and synchronize with `__syncthreads()`.
- **Guarantee:** a block always runs on exactly **one** SM, in full, never
  split across two.

### Grid
- **What it is:** every block belonging to one kernel launch.
- **You control:** how many blocks, via the first `<<<...>>>` parameter.
- **Created by:** `kernel<<<gridDim, blockDim>>>(...)`.

```cuda
dim3 blockSize(256);                 // YOU choose: 256 threads/block
dim3 gridSize(100);                  // YOU choose: 100 blocks
kernel<<<gridSize, blockSize>>>();   // Grid = 25,600 threads total, all your call
```

`blockDim.x` and `gridDim.x` are easy to swap in your head — keep them
straight:

| Variable | Meaning | In `<<<100, 256>>>` |
|---|---|---|
| `gridDim.x` | **how many blocks** in the grid | 100 |
| `blockDim.x` | **how many threads** per block | 256 |

---

## 3. The hybrid construct: Warp

- **What it is:** a group of exactly **32 threads** that execute in lockstep
  — same instruction, same clock cycle, on 32 CUDA cores at once (SIMT:
  Single Instruction, Multiple Threads).
- **Who creates it:** the hardware, automatically, the moment your block
  lands on an SM. You never write code that creates a warp.
- **The rule:** threads 0–31 of a block become warp 0, threads 32–63 become
  warp 1, and so on — always contiguous groups of 32, in `threadIdx` order.
- **Why it's the *real* unit of execution:** the GPU does not have circuitry
  to issue an instruction to one thread. It issues to a warp, full stop. A
  block of 128 threads isn't "128 things running" — it's "4 warps being
  scheduled."
- **Warp divergence:** if thread 5 in a warp takes `if(true)` and thread 6
  takes `if(false)`, the hardware cannot run both paths concurrently — all 32
  lanes execute the *true* branch first (with the false-branch lanes masked
  off/idle), then all 32 execute the *false* branch (with the true-branch
  lanes now masked). Both branches cost cycles; only one branch's-worth of
  *useful* work happens on each pass. This is why divergent branches inside a
  warp are expensive — it's serialized, not parallel.

---

## 4. The hardware side (fixed by the chip)

### CUDA Core
- **What it is:** a single ALU — does one arithmetic operation per clock.
- **Fixed count**, set at chip design time (e.g. an L4 has 7,424 total —
  58 SMs × 128 cores/SM).

### SM (Streaming Multiprocessor)
- **What it is:** a self-contained processor block on the die, with its own
  CUDA cores, warp scheduler(s), register file, shared memory, and L1 cache.
- **Fixed count** per GPU model (58 on an L4).
- **A block is assigned to exactly one SM** and stays there for its entire
  life — no migration, no splitting.
- **Multiple blocks can share one SM concurrently**, as long as the SM has
  spare registers/shared-memory/thread-slot capacity for them.
- **You have zero control over *which* SM gets which block**, or the order
  blocks run in. This is why kernels that `printf` show scrambled,
  run-to-run-varying output ordering between blocks.

### GPC (Graphics Processing Cluster)
- **What it is:** a hardware grouping of several SMs that share some
  supporting infrastructure (e.g. raster engines, an L1.5-ish cache slice).
- **Fixed count** per GPU model (roughly 7–8 on an L4-class chip).
- **Has no CUDA C++ counterpart.** There is no `gpcIdx`, no way to launch
  "per GPC." It's chip-floorplan detail that mainly affects cache/interconnect
  locality — invisible to how you write kernels, relevant mostly to very
  low-level performance analysis (e.g. profiler traces, NVLink/cache
  topology).

### GPU
- **What it is:** the entire chip — all GPCs, all SMs, plus global memory
  (VRAM), memory controllers, and the scheduling hardware that assigns
  blocks to SMs.

```
GPU (the whole chip)
 └── GPC 0..N            (hardware cluster of SMs — invisible to your code)
      └── SM 0..M         (physical processor: cores, registers, L1, sched.)
           └── warp(s)    (hardware execution granularity — 32 threads)
                └── thread (one lane, one CUDA core, one clock's worth of work)
```

---

## 5. Watching a launch happen: `kernel<<<4, 128>>>()`

**Step 1 — you describe the shape (software).** Nothing has executed yet.

```
Grid:
├── Block 0: 128 threads (threadIdx.x = 0..127)
├── Block 1: 128 threads
├── Block 2: 128 threads
└── Block 3: 128 threads
Total: 512 threads
```

**Step 2 — hardware slices each block into warps.** Automatic, non-negotiable,
always groups of 32:

```
Block 0 → Warp 0 (threads 0-31), Warp 1 (32-63), Warp 2 (64-95), Warp 3 (96-127)
Block 1 → same pattern
Block 2 → same pattern
Block 3 → same pattern
Total warps = 4 blocks × 4 warps = 16
```

**Step 3 — the hardware scheduler assigns each block to an SM.** On a 58-SM
L4, with only 4 blocks, most SMs sit idle:

```
Block 0 → SM 12   (no reason — the scheduler just picked it)
Block 1 → SM 3
Block 2 → SM 47
Block 3 → SM 12   (shares an SM with Block 0)

SM 0,1,2,4-11,13-46,48-57 → IDLE (54 of 58 SMs unused!)
```

This is *why* you should launch far more blocks than you have SMs — with
only 4 blocks you're using ~5% of the chip.

**Step 4 — inside one SM, warps get scheduled cycle by cycle.** SM 12 now
holds Block 0 (4 warps) and Block 3 (4 warps) = 8 warps resident. It has 4
warp schedulers, each able to issue one instruction per cycle:

```
Cycle 1:
  Scheduler 0 → Block0.Warp0 → issues ADD → 32 cores execute it
  Scheduler 1 → Block0.Warp1 → issues MUL → another 32 cores
  Scheduler 2 → Block3.Warp0 → issues LOAD (starts a ~400-cycle memory fetch)
  Scheduler 3 → Block3.Warp2 → issues ADD

Cycle 2:
  Block3.Warp0 is now WAITING on memory → scheduler 2 instantly swaps to
  Block0.Warp2 (which is ready) instead of sitting idle.
```

**This swap is free.** No overhead. It's the whole reason GPUs oversubscribe
threads relative to cores — while some warps wait ~400 cycles for memory, the
scheduler fills the gap with warps that are ready to compute, so the 128
cores on that SM are (ideally) never idle.

```
Warp 0: LOAD ........(waiting ~400 cycles)........ DATA ARRIVES → ADD → MUL
Warp 1:      ADD → MUL → LOAD ....(waiting)....
Warp 2:           ADD → ADD → LOAD ....(waiting)....
Warp 3:                ADD → MUL → ADD → LOAD ....
```

---

## 6. Why the limits are what they are (the engineering, not just the number)

These numbers are physically etched into the silicon — NVIDIA chose them as
tradeoffs, not arbitrarily. Exact values shift **per compute-capability
generation** (see §9 for how to look up yours) — the reasoning below is what
stays constant.

### Why is a warp always 32, never something else?
32 is the width NVIDIA built the SIMT execution lanes and the warp-tracking
hardware (program counter, active-mask, scheduling state) around, since
Tesla-generation GPUs. It's not a software choice at all — it's the
granularity the ALUs and instruction-issue hardware physically operate in.

### Why max 1024 threads per block, when an SM can hold more?
Three separate reasons converge on this number:
1. **`__syncthreads()` cost.** A block-wide barrier needs a physical counter
   that tracks "how many threads have arrived." The bigger the block, the
   longer that barrier takes to resolve. Uncapping this would make
   synchronization arbitrarily slow.
2. **Co-residency.** If the block cap equaled the full per-SM thread capacity,
   only one block could ever occupy an SM — you'd lose the ability to have
   multiple blocks interleave and hide each other's stalls.
3. **Resource partitioning.** Shared memory and registers must be divided
   *between* blocks resident on an SM. One block claiming the entire budget
   leaves nothing for a second block to co-reside with.

### Why is there a hard per-SM thread ceiling (e.g. 1536 on an L4)?
Driven by the **register file**. An L4 SM has 65,536 physical registers.
NVIDIA picks the max resident thread count so that, for realistic kernels,
each thread still gets enough registers to avoid spilling to slow local
memory:
- Allow 4096 threads → only 16 registers/thread average → most real kernels
  spill immediately (need 20–60+).
- Allow 512 threads → ~128 registers/thread, but too few warps in flight to
  hide memory latency.
- **1536 (= 48 warps)** is the sweet spot NVIDIA chose for Ada-class chips —
  enough warps in flight for latency hiding, enough registers per thread for
  real workloads to avoid spilling.

### Why is there a cap on resident *blocks* per SM (not just threads)?
Every resident block costs the SM real tracking hardware, independent of how
many threads are in it:
- a shared-memory partition (base/limit registers marking which chunk is
  "block 7's"),
- a `__syncthreads()` barrier unit (arrival counter + bitmask) *per block*,
- lifecycle state (is this block still running? can it be retired and
  replaced?).

Each of those is transistors spent on bookkeeping instead of compute. The
cap keeps that overhead bounded even when you launch many tiny blocks. This
is *why very small blocks hurt occupancy*: with 32-thread blocks, you'd need
48 of them to fill 1536 threads, but if the block-count cap is lower than
that, you hit the block ceiling before you hit the thread ceiling — leaving
threads on the table.

### The real occupancy formula
The number of threads actually resident on an SM at once is the **minimum**
of four independent, competing limits:

```
resident_threads = min(
    max_threads_per_SM,                            (hardware ceiling, e.g. 1536)
    max_blocks_per_SM × your_block_size,            (block-count ceiling)
    registers_per_SM  ÷ registers_used_per_thread,  (register ceiling)
    shared_mem_per_SM ÷ shared_mem_used_per_block   (shared-memory ceiling)
)
```

Whichever term is smallest is your bottleneck — and it's often *not* the
thread ceiling. A kernel that uses too many registers or too much shared
memory per block can starve occupancy long before you'd expect, purely from
resource math, with no code "bug" involved.

**Worked examples on a hypothetical 1536-thread, 32-registers-per-thread-budget SM:**

| Block size | Blocks needed for 1536 threads | Hits block-count cap first? | Resident threads |
|---|---|---|---|
| 1024 | 1.5 → 1 fits | No | 1024 (67% occupancy — the rest is wasted) |
| 256 | 6 | No | 1536 (100%) |
| 128 | 12 | No | 1536 (100%) |
| 32 | 48 | **Yes**, if block cap < 48 | Less than 1536 — capped by block count, not by threads |

This is why **128, 256, or 512 threads/block** are the usual sane defaults —
they divide the thread ceiling evenly without tripping the block-count
ceiling.

---

## 7. What's shared at each level

| Scope | Private | Shared |
|---|---|---|
| **Thread** | Registers, local variables | — nothing — |
| **Warp** | Each thread's own registers | Same instruction, same program counter (lockstep) |
| **Block** | Each thread's own registers | `__shared__` memory; `__syncthreads()` barrier |
| **SM** | Shared mem/registers, *partitioned* per resident block | L1 cache, warp schedulers, CUDA cores (time-shared across all resident warps) |
| **GPU** | — nothing — | Global memory (VRAM), L2 cache, constant memory, texture memory |

Speed and size, roughly (L4-class numbers — check yours, see §9):

| Memory | Latency | Typical size | Visible to |
|---|---|---|---|
| Registers | ~0 cycles | ~255/thread | that one thread only |
| Shared memory | ~5 cycles | ~100 KB/SM | all threads in one block |
| L1 cache | ~30 cycles | ~128 KB/SM | all blocks resident on that SM |
| L2 cache | ~200 cycles | tens of MB, chip-wide | all SMs |
| Global memory (VRAM) | ~400 cycles | GBs | every thread, everywhere |

**Why two blocks can never share data directly:** they may be running on
*physically different SMs* (separate shared-memory chips on the die, no wire
between them), and they may not even be alive at the same time — Block 0
could finish and be retired before Block 99 is even scheduled. The only
thing both can see is global memory (VRAM) — slow, and with no built-in
ordering/synchronization guarantee between blocks.

**`__syncthreads()` works across warps, not just within one.** It's a
barrier for the *whole block*: every thread, in every warp of that block,
must reach the call before any of them proceeds. Skipping it when threads
depend on each other's writes to shared memory is a race condition.

```cuda
__shared__ float cache[256];
cache[threadIdx.x] = compute_something();
__syncthreads();                 // every warp in the block waits here
float neighbor = cache[threadIdx.x + 1];   // now safe — guaranteed written
```

---

## 8. Common misconceptions, resolved directly

| Confusion | Resolution |
|---|---|
| "Warp and block are the same thing" | Block = the *room size you design* (any size 1–1024). Warp = a *fixed table of 32* the hardware carves out of that room automatically. A 256-thread block = 8 warps, always. |
| "Block and SM are the same thing" | Block = software (you create it in code). SM = hardware (NVIDIA built it into the chip). A block *runs on* an SM the way a room assignment happens in a building — the room (block) doesn't stop existing as a concept just because it's placed in a specific building section (SM). |
| "More threads than cores means some just don't run" | No — it means the scheduler *rotates* which warps occupy the cores each cycle, swapping out stalled warps for ready ones. Oversubscription is the mechanism that hides memory latency, not a limitation to work around. |
| "Threads in a warp run in true parallel, independent of each other" | They run in **lockstep** — same instruction, same cycle. If their control flow diverges, the warp serializes the divergent paths (§3). |
| "A block can run on more than one SM if the SM doesn't have room" | Never. A block waits until *one* SM has enough resources for the *entire* block, then runs there in full, for its whole lifetime. |
| "I can control which SM my block runs on" | No — the hardware scheduler decides, and it's effectively non-deterministic from your code's point of view. Never write code that assumes a particular block-to-SM assignment or block execution order. |

---

## 9. Check *your* GPU's real numbers — don't hardcode assumptions

Every limit in §6 varies by **compute capability** (the GPU architecture
generation). The reasoning is universal; the exact numbers are not. Query
them at runtime instead of trusting a number from a tutorial (including this
one):

```cuda
#include <cstdio>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM count:                    %d\n", prop.multiProcessorCount);
    printf("Max threads per block:        %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM:           %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Registers per SM:             %d\n", prop.regsPerMultiprocessor);
    printf("Shared mem per SM (bytes):    %zu\n", prop.sharedMemPerMultiprocessor);
    printf("Warp size:                    %d\n", prop.warpSize);
    return 0;
}
```

Compile and run this on the machine you're actually using — `nvcc query.cu -o query && ./query`
— and use *those* numbers when reasoning about occupancy for real work.

---

## 10. Self-test — check your understanding

A dedicated, much larger question set (10 categories, recall through
applied/debugging, with diagrams per category) lives in
[`CUDA_SELF_TEST.md`](CUDA_SELF_TEST.md) — use that file to actually test
yourself.

---

## 11. Quick-reference cheat sheet

```
YOU decide (software):        Thread body (the kernel code)
                               Block size  (blockDim, 1–1024 threads)
                               Grid size   (gridDim, how many blocks)

HARDWARE decides:              Warp grouping (always 32, from your block)
                               Which SM runs which block
                               Scheduling order, timing, interleaving

CANNOT change (hardware ceilings, query with cudaGetDeviceProperties):
                               Max threads per block   (commonly 1024)
                               Max threads per SM      (varies by architecture)
                               Max resident blocks/SM  (varies by architecture)
                               Warp size               (32, universal today)
```

```
Occupancy bottleneck = min(
    thread ceiling per SM,
    block-count ceiling per SM × your block size,
    register budget per SM ÷ registers your kernel uses per thread,
    shared-mem budget per SM ÷ shared mem your kernel uses per block
)
```

Related in this repo: [`diagrams/block_gpu_architecture.drawio.xml`](diagrams/block_gpu_architecture.drawio.xml)
visualizes this hierarchy for `block.cu`'s specific `<<<4,8>>>` launch, and
[`diagrams/block_code_flow.drawio.xml`](diagrams/block_code_flow.drawio.xml)
walks through the async-launch / synchronize control flow.
