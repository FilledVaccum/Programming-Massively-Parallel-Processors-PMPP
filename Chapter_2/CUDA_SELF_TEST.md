# CUDA Self-Test — Thread / Warp / Block / SM / GPC

A question-first companion to [`CUDA_EXECUTION_MODEL.md`](CUDA_EXECUTION_MODEL.md).
Each category opens with a small diagram to ground the concept, then a set of
questions ranging from recall to applied/debugging. Attempt every question
**before** reading the answer key at the bottom — that's the point of a
self-test.

**How to use this file:** cover the answer key, work through a category,
then check yourself. Don't skip to the answers — being *wrong* on a question
you thought you knew is the most useful signal in here.

---

## Category A — Software vs. Hardware

```
┌────────────────────────────┐     ┌────────────────────────────┐
│   SOFTWARE (you choose)     │     │   HARDWARE (chip decides)   │
│   Grid, Block, Thread       │     │   GPU, GPC, SM, CUDA Core    │
└────────────────────────────┘     └────────────────────────────┘
                    ↑
              Warp: hardware-created,
              but shapes how you write code
```

- **A1.** Sort these into "software" or "hardware": `gridDim`, warp, SM,
  `threadIdx`, GPC, CUDA core, block size, register file.
- **A2.** You change `blockDim` from 128 to 256 and recompile. Does the
  number of SMs on your GPU change? Does the number of warps per block
  change?
- **A3.** True or false: "the warp is a software construct because you can
  reason about it in your code." Justify your answer.
- **A4.** Why is there no `gpcIdx` built-in variable, when there *is* a
  `blockIdx` and a `threadIdx`?
- **A5.** Which of these can you find in `cudaDeviceProp`, and which can you
  only find in your own source code: max threads per block, your chosen
  block size, SM count, your chosen grid size?

---

## Category B — Thread & Indexing

```
Block (blockDim.x = 8)
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │   threadIdx.x = 0..7
└────┴────┴────┴────┴────┴────┴────┴────┘

globalId = blockIdx.x * blockDim.x + threadIdx.x
```

- **B1.** A thread has `blockIdx.x = 3`, `blockDim.x = 64`, `threadIdx.x = 10`.
  What's its global thread ID?
- **B2.** Two *different* threads, in two *different* blocks, both print
  `threadIdx.x = 5`. Is that a bug? Why or why not?
- **B3.** Write the global ID formula for a **2D** grid of 2D blocks (hint:
  this is exactly what `grayscale.cu` in Chapter 3 does for `col`/`row`).
- **B4.** Your image is 76×62 pixels. Your block is 16×16 threads. Why does
  the kernel need `if (col < width && row < height)` before touching memory?
  What would happen without it?
- **B5.** Can a thread know the *total* number of threads in the whole grid
  using only its own built-in variables? Write the expression.
- **B6.** `threadIdx` is relative to what? `blockIdx` is relative to what?
  (They are *not* both "relative to the whole grid" — that's the trap.)

---

## Category C — Block & Grid

```
Grid (gridDim.x = 4)
┌─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │
│ 8 thrds │ 8 thrds │ 8 thrds │ 8 thrds │
└─────────┴─────────┴─────────┴─────────┘
kernel<<<4, 8>>>()  →  gridDim.x=4, blockDim.x=8, 32 threads total
```

- **C1.** What is the hard ceiling on threads in a single block, on virtually
  every current GPU?
- **C2.** Is there a hard ceiling on the number of *blocks* in a grid? What
  does that imply about how large a problem you can express in one launch?
- **C3.** `dim3 block(16, 16, 2);` — how many threads is that, and is it
  legal?
- **C4.** You need to process 1,000,000 elements with 256 threads per block.
  What's the minimum `gridDim.x` you need? What must the kernel do about the
  leftover elements?
- **C5.** Two blocks in the *same* grid, both mid-execution. Can Block 0
  read a variable that Block 1 just wrote to `__shared__` memory?
- **C6.** True or false: "all blocks in a grid start executing at the same
  moment." Explain what actually determines when a block starts.

---

## Category D — Warp Fundamentals

```
Warp 0 — 32 lanes, ONE instruction issued to all of them per cycle
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│L0│L1│L2│L3│L4│L5│L6│L7│L8│L9│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│L31│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
        all 32 lanes execute the SAME instruction, SAME cycle (SIMT)
```

Divergence, visualized — `if (threadIdx.x < 8) { A(); } else { B(); }` inside one warp:

```
Cycle 1:  lanes 0-7  execute A()     lanes 8-31 MASKED OFF (idle, no-op)
Cycle 2:  lanes 0-7  MASKED OFF      lanes 8-31 execute B()
          → 2 cycles spent to do what would take 1 cycle if all lanes agreed
```

- **D1.** How many threads make up a warp? Is this ever configurable?
- **D2.** A block has `blockDim.x = 100`. How many warps does it become? Is
  the last warp "full"? What are the extra lanes in that last warp doing?
- **D3.** `block.cu`'s `learning_block_basic` kernel is launched with
  `<<<4, 8>>>`. How many warps total across the whole grid? What fraction of
  each warp's 32 lanes is actually doing useful work?
- **D4.** Explain, in your own words, why warp divergence costs cycles
  rather than just "not mattering" the way you might expect from
  independent-thread thinking.
- **D5.** Does a divergent `if/else` inside a warp ever run both branches on
  the *same* lane? Or does each lane only ever execute the branch it took?
- **D6.** Why is choosing `blockDim.x` as a multiple of 32 generally good
  practice?
- **D7.** You run the same kernel twice and the `printf` output is ordered
  differently between runs, but within any 32-line chunk the ordering
  pattern looks consistent-ish. Why chunks of 32 specifically?

---

## Category E — SM & Occupancy

```
┌──────────────── SM ────────────────┐
│  Warp Scheduler ×4                 │
│  Register File  (partitioned/block)│
│  CUDA Cores ×128                   │
│  Shared Memory / L1 (partitioned)  │
│                                     │
│  Resident: Block 0 (4 warps)        │
│            Block 3 (4 warps)        │
│  = 8 warps resident, 4 issue/cycle  │
└─────────────────────────────────────┘
```

- **E1.** A block is assigned to SM 12. Halfway through execution, does the
  scheduler ever move it to SM 13 to balance load? Why or why not?
- **E2.** SM 12 is running Block 0 and Block 3 concurrently. Do they share
  the *same* shared-memory contents, or separate partitions?
- **E3.** Your SM has 4 warp schedulers. A block contributes 4 warps. If
  that's the *only* block resident, how many warps are actively issuing per
  cycle, and how many are "resident but idle" that cycle?
- **E4.** Write the 4-term occupancy formula (threads, blocks, registers,
  shared memory) from memory. Which term is the "bottleneck" in a given
  situation?
- **E5.** A kernel uses 64 registers/thread. The SM has 65,536 registers
  total. What's the max thread count the register budget alone allows? Is
  that necessarily the actual occupancy?
- **E6.** Why can very small blocks (e.g. 32 threads) sometimes result in
  *lower* SM occupancy than medium ones (e.g. 256 threads), even though the
  SM's total thread ceiling is much larger than 32?
- **E7.** "More resident warps than the scheduler can issue per cycle is
  wasteful." True or false — explain using the concept of latency hiding.
- **E8.** If a kernel needs so much shared memory per block that only 1
  block fits per SM, what happens to the SM's ability to hide memory
  latency compared to a kernel where 6 blocks fit?

---

## Category F — GPC & Chip-Level

```
GPU
 ├── GPC 0 → SM 0, SM 1, SM 2, SM 3   (shares some raster/cache plumbing)
 ├── GPC 1 → SM 4, SM 5, SM 6, SM 7
 └── ...
```

- **F1.** What does a GPC group together, physically?
- **F2.** Can you write CUDA C++ code that behaves differently depending on
  which GPC a block landed in?
- **F3.** If GPCs are invisible to your kernel code, why do they still show
  up in GPU spec sheets and architecture whitepapers at all?

---

## Category G — Memory Hierarchy & Sharing

```
Registers   (private/thread)      ~0 cyc     ~255/thread
   ↓
Shared Mem  (private/block)       ~5 cyc     ~100 KB/SM
   ↓
L1 Cache    (shared/SM)           ~30 cyc    ~128 KB/SM
   ↓
L2 Cache    (shared/GPU)          ~200 cyc   tens of MB
   ↓
Global Mem  (shared/GPU, VRAM)    ~400 cyc   GBs
```

- **G1.** Which memory level is visible to *only one* thread, and which is
  visible to *every* thread in the whole grid?
- **G2.** Two threads in the same block, different warps. Can one read a
  value the other wrote to shared memory *without* a `__syncthreads()`
  between the write and the read? What could go wrong?
- **G3.** Why can't two blocks share data through `__shared__` memory even
  in principle — what's the physical reason, not just "the rules say so"?
- **G4.** A kernel declares more `__shared__` memory per block than fits
  in the SM's shared-memory budget for even 2 concurrent blocks. What's the
  performance consequence, even though the kernel still runs correctly?
- **G5.** What's "register spilling," and which memory tier does it spill
  into? Why is that expensive?
- **G6.** Rank registers, shared memory, and global memory from fastest to
  slowest, and from most private to most shared. Do the two rankings move
  in the same direction?
- **G7.** Global memory is described as having "no synchronization
  guarantee between blocks." Concretely, what could go wrong if Block 0
  writes a value to global memory and Block 1 reads it, assuming no explicit
  ordering mechanism?

---

## Category H — Execution Model: Async & Synchronization

```
HOST (CPU)                          DEVICE (GPU)
main()
 kernel<<<4,8>>>();  ┄┄┄┄async┄┄┄┄▶  starts running
 (continues immediately)             ...still running...
 cudaDeviceSynchronize()  ───block───▶ ...finishes...
 (now safe to read results)
```

- **H1.** After `kernel<<<...>>>()` returns control to the next line of host
  code, has the kernel finished running on the GPU?
- **H2.** What specifically does `cudaDeviceSynchronize()` do, and what
  would happen if you removed it right before printing a result computed on
  the GPU?
- **H3.** `block.cu` launches two kernels back-to-back, *then* calls
  `cudaDeviceSynchronize()` once. Are both kernels guaranteed to have started
  before either finishes? Are both guaranteed to be finished by the time
  `cudaDeviceSynchronize()` returns?
- **H4.** You launch a kernel with a bad configuration (e.g. 2000 threads in
  one block). No crash, no printed error. What silent-failure trap did you
  fall into, and what two API calls would have caught it?
- **H5.** Why does `printf` from inside a kernel still show up on your
  terminal, when the kernel is physically running on a completely separate
  chip from your CPU?
- **H6.** True or false: "since kernel launches are async, the host and
  device are always doing useful work simultaneously." What's missing from
  that claim?

---

## Category I — Predict the Output / Spot the Bug

For each snippet, predict what happens (or find the bug) before checking
the key.

- **I1.**
  ```cuda
  kernel<<<1, 2048>>>();
  cudaDeviceSynchronize();
  printf("done\n");
  ```
  What prints? Why?

- **I2.**
  ```cuda
  __global__ void k(float* data, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      data[i] = i * 2.0f;      // n is NOT a multiple of blockDim.x
  }
  ```
  What's missing, and what does it cause when `n` isn't an exact multiple of
  the block size?

- **I3.**
  ```cuda
  __shared__ float cache[256];
  cache[threadIdx.x] = threadIdx.x;
  float neighbor = cache[threadIdx.x + 1];   // no barrier before this read
  ```
  What's the bug, and what's the fix?

- **I4.**
  ```cuda
  learning_block_basic<<<4, 8>>>();
  learning_block_grid<<<4, 8>>>();
  // no cudaDeviceSynchronize() here
  return 0;
  ```
  Is it guaranteed that either kernel's `printf` output appears before the
  process exits? Why or why not?

- **I5.** A teammate says: "I launched `<<<58, 256>>>` on our 58-SM GPU, so
  every SM gets exactly one block and the GPU is fully utilized." What's the
  flaw in this reasoning, given what you know about occupancy?

---

## Category J — Explain It In Your Own Words

No diagram needed here — these check whether you can *generate* the
explanation, not just recognize it.

- **J1.** In 3–4 sentences, explain the full path from `kernel<<<4,8>>>()`
  in your source file to 32 threads actually executing on silicon.
- **J2.** Explain why GPU programming rewards launching *far more* threads
  than you have CUDA cores, when a CPU programmer would consider that
  absurd oversubscription.
- **J3.** Explain why "threads per block" is a number *you* pick freely up
  to 1024, but "warps per SM issuing per cycle" is a number you have *no*
  control over at all.
- **J4.** A friend says "occupancy is just threads launched divided by
  CUDA cores." Correct this misconception using the 4-term formula.

---

## Answer Key

### Category A
- **A1.** Software: `gridDim`, `threadIdx`, block size. Hardware: SM, GPC,
  CUDA core, register file. Warp: hybrid (hardware-created, software-visible).
- **A2.** SM count: unchanged — that's chip hardware, fixed regardless of
  your code. Warps per block: changes — `256/32 = 8` warps instead of `4`.
- **A3.** **False.** You can reason about warps and their effects
  (divergence, coalescing), but you never *create* one in code — the
  hardware carves your block into warps automatically. Being
  software-*relevant* isn't the same as being software-*defined*.
- **A4.** `blockIdx`/`threadIdx` exist because *you* create blocks and
  threads — the hardware needs to tell each one its own identity within a
  structure you designed. A GPC assignment isn't something you requested or
  designed around, so there's nothing for the hardware to "report back" in
  a way that would be actionable in your kernel.
- **A5.** In `cudaDeviceProp`: max threads per block, SM count (both are
  hardware facts you query). In your own source: your chosen block size,
  your chosen grid size (both are decisions you made, not properties of the
  chip).

### Category B
- **B1.** `3 * 64 + 10 = 202`.
- **B2.** Not a bug — `threadIdx.x` is scoped *within a block*, so it resets
  to 0 for every block. Two threads in different blocks having the same
  `threadIdx.x` is expected and extremely common; what must be unique across
  the whole grid is the *global* ID, not `threadIdx` alone.
- **B3.**
  ```
  col = blockIdx.x * blockDim.x + threadIdx.x
  row = blockIdx.y * blockDim.y + threadIdx.y
  ```
- **B4.** `76×62` doesn't divide evenly into `16×16` blocks — the grid is
  rounded *up* (`ceil`), so some threads at the right/bottom edges map to
  `col`/`row` values past the actual image bounds. Without the guard, those
  threads read/write out-of-bounds memory — undefined behavior, potentially
  a crash or silently corrupted neighboring memory.
- **B5.** Yes: `gridDim.x * blockDim.x` (extend with `.y`/`.z` for
  multi-dimensional launches).
- **B6.** `threadIdx` is relative to **its own block** (resets per block).
  `blockIdx` is relative to **the grid** (unique per block, grid-wide). Only
  the *computed* global ID is unique across the whole grid.

### Category C
- **C1.** **1024** threads per block, on essentially every CUDA GPU since
  2014-era Maxwell.
- **C2.** The per-dimension grid limits are enormous (commonly billions in
  `x`) — in practice you're constrained by problem size and memory, not by
  the grid-dimension ceiling. This means you can express massively parallel
  problems (millions+ of threads) in a single launch.
- **C3.** `16 * 16 * 2 = 512` threads — legal (under 1024, and each
  dimension is under its individual cap).
- **C4.** `gridDim.x = ceil(1,000,000 / 256) = 3907` (since `3906 * 256 =
  999,936 < 1,000,000`, you need 3907 to cover the rest). The kernel must
  bounds-check (`if (i < n)`) because the last block's threads will index
  past the real data.
- **C5.** **No.** Shared memory is scoped to one block; Block 1 cannot see
  Block 0's shared memory at all, regardless of timing.
- **C6.** **False.** Blocks start when the hardware scheduler assigns them
  to an SM with available capacity — this depends on how many SMs exist, how
  many blocks are already resident, and resource availability, not on a
  synchronized "go" signal for the whole grid.

### Category D
- **D1.** **32**, and no — it's fixed by the hardware generation, not
  something your code configures (unlike block size).
- **D2.** `ceil(100/32) = 4` warps. The last warp is **not full** — it
  covers threads 96–127, but only 96–99 are real (4 threads); lanes
  100–127 (28 of them) have no corresponding thread and sit masked/idle.
- **D3.** `4 blocks × 1 warp/block = 4 warps` total (since `8 < 32`, each
  block is exactly 1 partially-filled warp). Each warp has only `8/32 = 25%`
  of its lanes doing real work.
- **D4.** Because the hardware issues *one instruction per cycle to the
  whole warp* — there's no circuitry to give lane 5 an `ADD` and lane 6 a
  `SUB` in the same cycle. Divergent lanes don't run "in parallel on a
  different path" — the warp serializes: one pass per distinct path taken,
  with non-participating lanes masked off (not doing anything, but also not
  freeing up their slot for other work) each pass.
- **D5.** Each lane executes only the branch its own condition selected; a
  lane never executes a branch it didn't take. What's shared is the *cycle
  budget* — the warp as a whole spends cycles on both branches serially, but
  no individual lane runs code twice.
- **D6.** Because block size not divisible by 32 always leaves at least one
  partially-empty warp (wasted lanes, as in D2/D3). Multiples of 32 use every
  lane of every warp.
- **D7.** Warps are the actual scheduled unit — the hardware picks *which
  warp* runs next largely unpredictably (run to run), but once a warp is
  chosen, all 32 of its lanes execute their `printf` together in that same
  instruction stream, so their outputs land as a contiguous group even
  though which group goes first/second varies.

### Category E
- **E1.** **No.** A block is bound to the SM it started on for its entire
  lifetime — there is no migration mechanism. Moving it mid-flight would
  require moving its entire register/shared-memory state, which the hardware
  simply doesn't support.
- **E2.** **Separate partitions.** The SM divides its shared-memory budget
  between resident blocks; Block 0's `__shared__` array and Block 3's
  `__shared__` array occupy different physical regions, even though both
  live in the same SM's shared-memory hardware.
- **E3.** All **4 warps issue** that cycle (one per scheduler) — since 4
  warps exactly matches 4 schedulers, **0 are resident-but-idle** that
  particular cycle.
- **E4.**
  ```
  resident = min(max_threads_per_SM,
                  max_blocks_per_SM × block_size,
                  regs_per_SM ÷ regs_per_thread,
                  shared_mem_per_SM ÷ shared_mem_per_block)
  ```
  Whichever term evaluates smallest is the bottleneck for that specific
  kernel/launch config — it's rarely the same term for every kernel.
- **E5.** `65,536 ÷ 64 = 1024` threads by the register budget alone. **Not
  necessarily actual occupancy** — the other three terms in E4 (thread
  ceiling, block-count ceiling, shared-memory budget) could each independently
  cap it lower still; the real number is the *minimum* of all four.
- **E6.** Because the *block-count* ceiling on an SM is a separate hardware
  limit from the *thread-count* ceiling. If you need 48 tiny 32-thread
  blocks to fill the thread ceiling but the SM only allows, say, 24 resident
  blocks, you hit the block-count wall first and leave thread capacity
  unused — occupancy is capped below 100% by block count, not by threads.
- **E7.** **False, in general.** Extra resident warps beyond what a
  scheduler issues *per cycle* are exactly the point — they're the "reserve
  bench" that lets the scheduler instantly swap in a ready warp the moment
  an active one stalls on memory. Idle-this-cycle isn't wasted; it's latency
  insurance.
- **E8.** Fewer resident blocks means fewer resident warps overall, which
  means fewer "ready substitutes" available when an active warp stalls on a
  ~400-cycle memory fetch. With only 1 block's warps resident, a stall can
  leave the SM genuinely idle far more often than with 6 blocks' worth of
  warps to rotate through.

### Category F
- **F1.** Several SMs, bundled together with some shared supporting hardware
  (e.g. raster engines, a cache slice) — a chip-floorplan grouping, not a
  scheduling unit you interact with.
- **F2.** **No.** There's no built-in variable exposing GPC identity, and no
  CUDA C++ mechanism to target or query it from within a kernel.
- **F3.** Because GPC layout affects *performance characteristics* engineers
  and architects care about (cache locality, interconnect topology, chip
  yield/binning) — it's meaningful at the hardware-design and
  low-level-profiling level, just not at the "what do I write in my kernel"
  level.

### Category G
- **G1.** Registers: visible to exactly **one thread**. Global memory
  (VRAM): visible to **every thread in the entire grid** (and even across
  separate kernel launches, since it persists until freed).
- **G2.** It can read a **stale or partially-written** value — there's no
  guarantee the writing warp has executed relative to the reading warp
  without an explicit barrier. `__syncthreads()` between the write and read
  is required for correctness; skipping it is a classic race condition.
- **G3.** Blocks can run on **physically different SMs**, and each SM's
  shared memory is separate on-chip hardware with no interconnect between
  them for this purpose — there is no wire to carry Block 0's shared-memory
  contents to Block 1's SM even if you wanted it to.
- **G4.** The kernel is still **correct**, but occupancy drops — if only 1
  block's worth of shared memory fits (instead of, say, 4 blocks' worth),
  far fewer warps are resident, reducing the scheduler's ability to hide
  memory latency (see E8). Performance suffers even though output is right.
- **G5.** When a thread's live variables exceed available registers, the
  compiler moves the overflow into **local memory**, which physically lives
  in (slow) global memory/cache, not on-chip registers. It's expensive
  because every access now costs ~hundreds of cycles instead of ~0.
- **G6.** Fastest → slowest: registers, shared memory, global memory. Most
  private → most shared: registers, shared memory, global memory. **Yes,**
  both rankings move the same direction — in this hierarchy, more sharing
  consistently correlates with more latency.
- **G7.** Block 1 might read the value **before** Block 0 has actually
  written it (if Block 1 happens to run first, concurrently, or the write
  hasn't been made visible yet) — global memory has no cross-block ordering
  guarantee without explicit mechanisms (e.g. atomics, separate kernel
  launches acting as an implicit barrier, or cooperative-groups grid sync).

### Category H
- **H1.** **No** — the call returns to host code immediately; the kernel may
  still be running (or not yet started) on the GPU. That's what
  "asynchronous" means here.
- **H2.** It blocks the host thread until every previously-launched kernel
  (and other device work) on that device has finished. Without it, code that
  reads GPU-produced results (or just assumes output has been flushed) can
  race ahead of the actual computation — printed/copied data may be stale,
  incomplete, or reflect a still-running kernel.
- **H3.** Both are guaranteed to have been **issued** (in program order) —
  but whether they've actually *started* executing concurrently or
  sequentially on the GPU depends on the scheduler and available SMs.
  `cudaDeviceSynchronize()` **does** guarantee both are fully finished by the
  time it returns — that's its entire purpose.
- **H4.** The trap is **assuming CUDA throws or crashes on bad launch
  configs by default — it doesn't.** `cudaGetLastError()` (to catch the
  launch-configuration error itself) and checking the return value of
  `cudaDeviceSynchronize()` (to catch errors during execution) are the two
  calls that surface it.
- **H5.** The CUDA runtime maintains an internal buffer on the device for
  kernel `printf` calls; the *host* periodically drains that buffer and
  writes it to your terminal — commonly flushed by `cudaDeviceSynchronize()`
  or kernel completion, not by some direct wire from GPU silicon to your
  screen.
- **H6.** **False** as a blanket claim — the host frequently does have
  *nothing left to do* but wait (e.g. it needs the GPU's result before its
  next step, as in `block.cu`'s repeated launch → sync → launch pattern).
  Async launches only help when the host genuinely has independent work to
  overlap with the GPU's; if it doesn't, it just launches and then blocks
  anyway.

### Category I
- **I1.** **Launch fails; nothing related to the kernel executes; "done"
  still prints.** `2048` exceeds the 1024-threads-per-block limit, so the
  launch itself is rejected — but since there's no error-checking code here,
  the failure is silent and the rest of the program proceeds normally
  (`cudaDeviceSynchronize()` would return an error code here, but it's being
  ignored).
- **I2.** Missing a bounds check (`if (i < n)`). Without it, threads whose
  computed `i` is `≥ n` (which exist whenever `n` isn't an exact multiple of
  `blockDim.x`, since you must round the grid size *up*) write past the end
  of the `data` buffer — out-of-bounds memory access, undefined behavior.
- **I3.** Missing `__syncthreads()` between the write to `cache[threadIdx.x]`
  and the read of `cache[threadIdx.x + 1]`. Without it, a thread might read
  its neighbor's slot before that neighbor's warp has actually executed the
  write — a race condition. Fix: insert `__syncthreads();` between the two
  lines.
- **I4.** **No, not guaranteed.** Without `cudaDeviceSynchronize()` (or any
  other synchronizing call), the host can reach `return 0` and the process
  can exit before the GPU has finished — or even started — either kernel, in
  which case some or all `printf` output may never appear.
- **I5.** Occupancy isn't "1 block = 1 SM done." An SM has a much larger
  thread ceiling than one 256-thread block (E4/E6) — launching only 58
  blocks of 256 threads each uses just a fraction of *each* SM's actual
  capacity (e.g. 256 of a possible 1536 resident threads), leaving most of
  every SM's latency-hiding capacity unused. "One block per SM" is not the
  same as "SM fully utilized."

### Category J
- **J1.** *Model answer:* `kernel<<<4,8>>>()` describes a grid of 4 blocks
  with 8 threads each in software. The hardware scheduler assigns each block
  to some SM (each block staying there for its whole life), and — because
  `8 < 32` — carves each block into a single, mostly-empty warp. That SM's
  warp scheduler then issues that warp's instructions, one per cycle, to 32
  physical CUDA-core lanes (only 8 of which correspond to a real thread),
  until the kernel finishes.
- **J2.** *Model answer:* Because global-memory access (~400 cycles) vastly
  outlasts a typical arithmetic instruction (~1 cycle), a GPU core sitting
  idle during that wait is enormously wasteful across thousands of cores.
  Oversubscribing threads gives the warp scheduler a deep pool of *other*
  ready warps to swap in instantly whenever one stalls, so cores stay busy —
  the "waste" of unused threads on a CPU becomes "insurance against idling"
  on a GPU.
- **J3.** *Model answer:* Threads-per-block is a *software* decision because
  it's how you choose to organize *your* problem — the hardware just needs
  to know the number to plan resources. Warps-issued-per-cycle is fixed by
  how many physical warp-scheduler units exist on the SM (silicon,
  unchangeable) — no code you write can add a 5th scheduler to a 4-scheduler
  SM.
- **J4.** *Model answer:* That definition ignores that a thread can be
  *launched* without being *resident* at the same time as all the others —
  real occupancy is capped by whichever of four independent hardware budgets
  (thread slots, block slots, registers, shared memory) runs out first for
  *that specific kernel's* resource usage, not simply by dividing a thread
  count by a core count.

---

Related: [`CUDA_EXECUTION_MODEL.md`](CUDA_EXECUTION_MODEL.md) for the full
explanation these questions are drawn from, and
[`diagrams/block_gpu_architecture.drawio.xml`](diagrams/block_gpu_architecture.drawio.xml) /
[`diagrams/block_code_flow.drawio.xml`](diagrams/block_code_flow.drawio.xml)
for the editable draw.io versions of the concepts sketched in ASCII above.
