# Designing a CUDA Kernel From First Principles — A Thought-Process Template

A reusable checklist for the question every kernel starts with: *"I have a
problem I want the GPU to solve — where do I even begin, and how do I not
paint myself into a corner?"*

This is not a syntax reference. It's the **sequence of questions an
experienced CUDA programmer asks themselves**, in order, before and while
writing a kernel — from something as small as `vecAddKernel` to something as
involved as a tiled convolution or a sparse-matrix solver. The questions
don't change with complexity. Only the answers do. That's the whole point of
having a template: you stop re-inventing your process for every new kernel
and start recognizing which *pattern* you're in.

Work through the phases **in order**. Skipping ahead to "make it fast" before
finishing "make it correct" is the single most common way beginners waste
time — they optimize a kernel that was wrong to begin with.

> **Before Phase 0:** this document assumes you've already decided the
> workload belongs on a GPU. If you haven't answered *that* yet, start one
> level up with
> [`COMPUTE_TARGET_DECISION_TREE.md`](COMPUTE_TARGET_DECISION_TREE.md) —
> "should this even run on a GPU, vs. CPU, vs. a cluster, vs. something
> else?" is Phase −1, and it's easy to skip past by accident.

---

## 0. The one-sentence mental model

> A CUDA kernel is not "a program that runs on the GPU." It is **one
> function, written from the point of view of a single thread**, that you
> then stamp out thousands of times. Every design decision you make is really
> answering one question: *what should ONE thread do, and what does it need
> to see to do it?*

Everything below is that one question, expanded into a checklist.

---

## Phase 0 — Understand the problem *before* touching CUDA

Do this on paper or in plain pseudocode. No `__global__`, no `blockIdx`, not
yet. If you can't describe the computation as ordinary sequential code, you
have no business parallelizing it — you'll just be parallelizing confusion.

Ask:

1. **What are the inputs and outputs, precisely?** Their types, their sizes,
   whether the sizes are known at compile time or only at runtime.
2. **What is the mathematical/algorithmic relationship between them?** Write
   the loop a CPU would run. `for i in range(n): C[i] = A[i] + B[i]` — that
   loop *is* the kernel, before it's a kernel.
3. **Classify the dependency pattern.** This single classification predicts
   most of your later design decisions. Ask "to produce one output element,
   what inputs do I need, and do outputs depend on each other?"

   | Pattern | One output depends on | Example |
   |---|---|---|
   | **Map** (elementwise) | exactly one (or a fixed few) inputs, independently | `C[i]=A[i]+B[i]` |
   | **Stencil** | a fixed *neighborhood* of inputs | blur, convolution |
   | **Reduction** | *all* inputs, combined into few/one output | sum, max, dot product |
   | **Scan / prefix** | all *previous* inputs, in order | running total, prefix sum |
   | **Map+Reduce** | a full row/column of one input combined against another | matrix multiply |
   | **Gather/Scatter** | data-dependent, irregular addresses | sparse matmul, histograms, graphs |

   Map problems have zero cross-thread dependency — they're the easy case.
   Everything past Map means threads need to *see each other's work*, which
   means you'll need shared memory, `__syncthreads()`, atomics, or multiple
   kernel launches. Knowing which bucket you're in on day one saves you from
   discovering it the hard way after a race condition.
4. **What's the actual size of the problem?** Is `n` in the tens (GPU is
   probably the wrong tool — kernel-launch overhead alone will dominate) or
   in the millions (GPU is exactly the right tool)? Order-of-magnitude
   matters before you write anything.

---

## Phase 1 — Decide what ONE thread does

This is the single most important design decision in the entire process —
get it right and the rest mostly falls out automatically.

Ask:

- **What is the smallest unit of independent work?** One array element? One
  matrix row? One pixel? One output tile? One tree node?
- **Is there enough of that unit to give the GPU thousands of them?** A
  modern GPU wants tens of thousands of threads in flight to hide memory
  latency. If your "one thread : one unit" mapping produces only a few
  hundred units total, either the problem is too small for a GPU, or you
  need a coarser-to-finer remap (e.g. one thread computes *several* output
  elements via a grid-stride loop, to give the GPU more independent work per
  thread instead of more threads).
- **Do threads need to communicate?** If thread *i*'s answer depends on
  thread *j*'s answer (reduction, scan, tiling with reuse), those threads
  must either:
  - live in the **same block** (so they can use `__shared__` memory +
    `__syncthreads()`), or
  - be split across **separate kernel launches** (a launch boundary is a
    free, implicit grid-wide synchronization point — the only one that
    exists), or
  - communicate via **atomics** in global memory (works, but serializes —
    last resort, not first).

If the answer to "do threads need to communicate" is *no* (pure Map), you're
already most of the way to a working kernel. If it's *yes*, that answer is
what tells you to reach for shared memory later, not global-memory chatter.

---

## Phase 2 — Map the work onto grid / block / thread

Now translate Phase 1's answer into actual launch geometry.

1. **Match the hierarchy's dimensionality to the data's natural shape.**
   A flat array → 1D grid of 1D blocks. An image or matrix → 2D grid of 2D
   blocks (`row = blockIdx.y*blockDim.y + threadIdx.y`,
   `col = blockIdx.x*blockDim.x + threadIdx.x`). Volumetric data → 3D.
   Forcing 2D data through a 1D index (`i = row*width + col` computed by
   hand) *works*, but 2D-native indexing keeps your boundary checks and
   memory-access reasoning far more legible — use it when the data is
   genuinely 2D.
2. **Pick a block size.** Defaults that are almost always reasonable on a
   first pass: **128, 256, or 512** threads per block, and always a
   **multiple of 32** (the warp size — a block size of 100 wastes 28 lanes
   of the last warp on every launch). Don't agonize over the exact number
   yet; you will revisit this in Phase 8 once you can *measure* occupancy
   instead of guessing it.
3. **Compute the grid size with ceiling (round-up) division, always:**
   ```
   blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
   ```
   Plain `n / threadsPerBlock` truncates and silently drops whatever doesn't
   divide evenly — the last several elements simply never get a thread. This
   is a **one-line bug that produces correct-looking output on convenient
   input sizes and wrong output the moment `n` isn't a clean multiple.**
   Assume it never is.
4. **Write the boundary guard — always, no exceptions:**
   ```c
   if (i < n) { ... }          // 1D
   if (row < rows && col < cols) { ... }   // 2D — AND, never OR
   ```
   Rounding the grid *up* means the last block almost always has threads
   whose computed index falls past the end of your real data. Without the
   guard those threads read/write past the end of your `cudaMalloc`'d
   buffers — undefined behavior, not a crash you can rely on seeing.

---

## Phase 3 — Plan the memory before you plan the math

Experienced CUDA programmers think about **data movement before compute**,
because on real GPUs the bottleneck is almost always moving bytes, not doing
arithmetic. Ask, for every array your kernel touches:

- **Where does this value live?** Global memory (VRAM, big & slow),
  shared memory (per-block scratchpad, small & fast), constant memory
  (read-only, broadcast-friendly), or just a register (per-thread private)?
- **How many times does each byte get read across *all* threads?** If every
  thread in a block re-reads the *same* input bytes from global memory
  (e.g. each output pixel's convolution window overlaps its neighbors'), that
  redundant traffic is exactly what **shared-memory tiling** exists to
  eliminate: load the shared region once per block, into `__shared__`
  memory, then have every thread in the block read from there instead of
  going back to global memory.
- **Is the access pattern coalesced?** When consecutive threads in a warp
  read consecutive addresses (`A[i]`, `i = threadIdx.x + ...`), the hardware
  combines them into one wide memory transaction. When threads jump around
  (`A[i*stride]` with a large stride, or worse, pointer-chased/random
  access), each thread pays for its own transaction — often 10-30x slower in
  practice.
- **What's the arithmetic intensity?** Roughly, *FLOPs performed ÷ bytes
  moved from global memory*. Low intensity (like `C[i]=A[i]+B[i]`: one add
  per 12 bytes moved) means you are **memory-bound** — no amount of clever
  arithmetic will help; only moving fewer/bigger/more-coalesced bytes will.
  High intensity (like a compute-heavy stencil or matmul with good tiling)
  means you are **compute-bound**, and the memory system is not your
  bottleneck. Knowing which regime you're in *before* writing code tells you
  which entire category of optimization is even worth attempting later.

---

## Phase 4 — Write the naive, obviously-correct version first

Resist every urge to add shared memory, tiling, unrolling, or
warp-shuffle tricks right now. Write the simplest kernel that could possibly
be correct:

- One thread, one unit of work (from Phase 1), reading directly from global
  memory, guarded by the boundary check from Phase 2.
- No premature cleverness. `vecAddKernel` — one line of real work, one `if` —
  *is* what every kernel looks like at this stage, regardless of how complex
  the final version will eventually be.

This version is your **correctness baseline**. You will compare every later,
optimized version's output against it (or against a CPU reference — see
Phase 6). Never skip straight to the "smart" version; you'll have no way to
tell a wrong-but-fast kernel from a right-but-fast one.

---

## Phase 5 — Host-side orchestration checklist

The host function's job is mechanical and the same shape every time:

1. **Allocate** device memory for every buffer the kernel touches
   (`cudaMalloc`, sized in *bytes* — `n * sizeof(type)`, not "n elements").
2. **Copy inputs in** (`cudaMemcpy(..., HostToDevice)`) — only the buffers
   the kernel *reads*. Don't copy outputs in; there's nothing there yet.
3. **Compute the launch configuration** (Phase 2's numbers) and **launch**
   the kernel. Remember this call is asynchronous — the host does not wait
   here.
4. **Copy outputs back** (`cudaMemcpy(..., DeviceToHost)`). On the default
   stream this call is implicitly synchronizing — it's *this* line that
   actually waits for the kernel to finish, not the launch line.
5. **Free** every device allocation (`cudaFree`) once you're done with it.
6. **Check errors.** At minimum during development, wrap calls or check
   `cudaGetLastError()` after the launch — a kernel that fails to launch
   (bad config, out-of-resources) fails *silently* by default; nothing in
   your program will tell you unless you ask.

Don't reach for streams, pinned memory, or overlap-transfer-with-compute yet
— that's a Phase 8 optimization, not a Phase 5 necessity.

---

## Phase 6 — Verify correctness before you ever measure speed

- Write (or reuse) a plain CPU implementation of the same algorithm and
  **diff the GPU's output against it**, element by element.
- For floating point, compare with a tolerance (`fabs(a-b) < 1e-5`), *not*
  `==` — except in the rare case where the arithmetic is guaranteed exact
  (e.g. adding small integers-as-floats, like `1.0f + 2.0f`).
- Deliberately test the sizes that are most likely to expose off-by-one and
  boundary bugs: `n = 0`, `n = 1`, `n` exactly equal to `blockDim.x`, `n`
  equal to `blockDim.x + 1`, and a large size that is *not* a multiple of
  your block size (this is the common case in production, so it should be
  the common case in your test).
- If you have access to `compute-sanitizer` (formerly `cuda-memcheck`), run
  it now. It catches out-of-bounds accesses and races that happen to produce
  *coincidentally correct* output on your test machine but are still bugs.

**Do not proceed to optimization until this phase passes.** A fast wrong
answer is worthless, and it's much harder to tell a fast kernel is wrong than
a slow one.

---

## Phase 7 — Profile. Don't guess.

Once the naive kernel is verified correct, *measure* before changing
anything:

- **Nsight Systems** (timeline view): is the host waiting on the GPU
  unnecessarily? Is memory transfer time comparable to or larger than kernel
  execution time? (If so, the biggest win might be reducing transfers or
  overlapping them — not touching the kernel at all.)
- **Nsight Compute** (per-kernel deep dive): what's the achieved occupancy
  (resident warps per SM vs. the theoretical max)? What's the memory
  throughput as a percentage of peak? What's the compute throughput as a
  percentage of peak? What are warps actually *stalling* on (memory
  dependency, execution dependency, synchronization barrier)?

The stall reason and the memory-vs-compute throughput numbers directly
answer the question Phase 3 asked you to predict: is this kernel
memory-bound or compute-bound? **That answer determines which of the levers
in Phase 8 will actually move the needle** — spending an afternoon on
arithmetic tricks for a memory-bound kernel is time you will not get back.

---

## Phase 8 — Optimize, in the order experts actually reach for these

Apply these roughly in this order — each is usually cheaper to attempt and
higher-impact than the one after it. Re-profile after *every* change; keep
only what measurably helps, and revert what doesn't (a change that "should"
help but doesn't measure as helping is not helping).

1. **Fix uncoalesced global memory access first.** Almost always the
   cheapest, biggest win, and requires no algorithm change — just reordering
   which thread touches which address.
2. **Add shared-memory tiling** where Phase 3 found redundant re-reads of the
   same global data across nearby threads (classic move in convolution and
   matrix multiply: load a tile into `__shared__` once per block, have every
   thread in the block reuse it).
3. **Reduce warp divergence.** If neighboring threads in a warp take
   different branches of an `if`, the warp executes *both* paths and masks
   off the inactive lanes each time — pure waste. Restructure branches so
   whole warps tend to agree, where possible.
4. **Tune occupancy.** Adjust block size, or reduce registers/shared-memory
   used per thread, so more warps can be resident on each SM at once (more
   resident warps = more ability to hide memory latency by switching to
   another warp while one waits).
5. **Eliminate shared-memory bank conflicts** — multiple threads in a warp
   hitting the same shared-memory bank serializes what should be a
   single-cycle broadcast.
6. **Reach for warp-level primitives** (`__shfl_down_sync` and friends) to
   replace `shared memory + __syncthreads()` for small, warp-local
   reductions — skips the round-trip through shared memory entirely.
7. **Reduce atomic contention.** If many threads atomically update the same
   address, restructure so each thread (or each block) accumulates a private
   partial result first, and only a few atomics combine those partials at
   the end ("privatization").
8. **Overlap transfer and compute** with streams and asynchronous memcpy,
   once the kernel itself is efficient — no point overlapping a slow kernel
   with a transfer; fix the kernel first.
9. **Micro-tune last**: loop unrolling, `#pragma unroll`, instruction-level
   scheduling. Smallest gains, most fragile, most likely to be undone by the
   next compiler version — do this only after everything above is done and
   you're chasing the last few percent.

---

## The scaling ladder — same nine phases, harder answers

The phases above don't change as problems get harder. What changes is only
the *answer* to "what's the dependency pattern" and "what does one thread
do." Recognizing which rung of this ladder a new problem sits on tells you
most of the design before you write a line of code:

| Kernel type | One thread does | Dependency pattern | Key structural device | Canonical example |
|---|---|---|---|---|
| Elementwise map | Reads a fixed few inputs, writes one output | None | Boundary guard only | `vecAddKernel` (this repo) |
| Stencil / convolution | Reads a neighborhood, writes one output | Overlapping neighbor reads | Shared-memory tile with a "halo" border | image blur, edge detection |
| Matrix multiply | Reads a full row *and* a full column, writes one output | Map + reduction per output | Shared-memory tiling of *both* operands | tiled matmul |
| Reduction | Combines many inputs into one | Tree-structured combination | Shared memory + warp-shuffle tree reduce; multi-pass or atomics for the final merge | sum/min/max reduction |
| Scan / prefix sum | Each output depends on all previous inputs | Strictly sequential dependency, parallelized via a trick | Hillis-Steele or Blelloch parallel scan (log₂n passes) | running totals |
| Histogram | Many threads may write the *same* output bucket | Scatter with write collisions | Atomics, or per-block private histograms merged at the end | pixel-value histogram |
| Sparse / irregular (graphs, sparse matrices) | Variable amount of work per thread | Data-dependent, unpredictable at compile time | Load balancing via work queues, thread-coarsening, or dynamic parallelism | SpMV, BFS |

Every row still starts at Phase 0 ("what's the dependency pattern?") and ends
at Phase 8 ("profile, then optimize in that order"). Complexity buys you a
harder Phase 1 and Phase 3 — it does not buy you a different process.

---

## The one-page version — pin this above your editor

1. **What's the CPU pseudocode?** Can't parallelize what you can't write
   sequentially first.
2. **What's the dependency pattern?** Map / stencil / reduction / scan /
   map+reduce / scatter-gather.
3. **What does ONE thread do?** This is the actual design decision;
   everything else follows from it.
4. **Do threads need to talk to each other?** If yes → shared memory +
   `__syncthreads()`, or separate kernel launches, or atomics (in that order
   of preference).
5. **What shape is the data?** → matching grid/block dimensionality.
6. **Block size**: multiple of 32, start at 256, revisit after profiling.
7. **Grid size**: `(n + block - 1) / block`, *always* ceiling division.
8. **Boundary guard**: `if (i < n)` (or the AND'd 2D/3D form) — *always*,
   no exceptions.
9. **Where does each array live**, and is each access coalesced? Redundant
   reads across nearby threads → shared-memory tiling candidate.
10. **Write the naive version. Verify it against a CPU reference** before
    optimizing anything — tolerant float compare, and test the boundary
    sizes on purpose (0, 1, exact multiple, one more than exact multiple).
11. **Profile before optimizing.** Memory-bound or compute-bound? That
    answer picks which levers in step 12 are worth pulling.
12. **Optimize in order:** coalescing → shared-memory tiling → reduce
    divergence → occupancy tuning → bank conflicts → warp shuffles →
    reduce atomics → overlap transfer/compute → micro-tuning. Re-profile
    after each change; discard anything that doesn't measurably help.

---

## Pitfalls the checklist won't save you from unless you already know them

- **Forgetting `__syncthreads()` after a shared-memory load.** Every thread
  in the block must finish *writing* the tile into shared memory before
  *any* thread starts reading it back out. Skipping this is a race — and it
  will often look correct on small test inputs by sheer luck of scheduling.
- **`__syncthreads()` only synchronizes a block, never the whole grid.**
  There is no such thing as mid-kernel grid-wide synchronization (outside of
  cooperative-groups APIs); if block *A* needs to see the finished output of
  block *B*, that has to happen across two separate kernel launches.
- **2D boundary checks need AND, not OR.** `if (row < rows || col < cols)`
  lets threads through that are out of bounds on *one* axis — it should
  almost always be `&&`.
- **Integer overflow on large problem sizes.** `blockIdx.x * blockDim.x +
  threadIdx.x` computed in `int` can overflow well before you'd expect on
  genuinely large arrays; know when to promote to `size_t`/`long long`.
- **Halo/edge regions in tiled stencils reading uninitialized shared
  memory.** The tile's border threads often need data from a *neighboring*
  block's tile — that data has to be explicitly loaded (with its own
  boundary check for the edge of the whole array), not assumed to already
  be there.
- **Silent launch failures.** A misconfigured launch (too many threads per
  block, too much shared memory requested) does not throw — it just doesn't
  run, and your output buffer quietly still contains whatever garbage it had
  before. Check `cudaGetLastError()` while developing.
- **Float equality.** `==` on floating point is only safe when you can prove
  the arithmetic is exact (small integer values, powers of two); otherwise
  always compare with a tolerance.

---

## Worked example: applying this to `Chapter_2/vecAddKernel.cu`

This repo's simplest kernel maps onto the template like this — use it as the
calibration point for how much (or little) machinery a genuinely simple
problem needs:

| Phase | Answer for `vecAddKernel` |
|---|---|
| 0. Dependency pattern | Pure **Map** — `C[i]` depends only on `A[i]`, `B[i]`. Zero cross-thread dependency. |
| 1. One thread does | Exactly one output element: read two floats, add, write one float. |
| 2. Grid/block shape | 1D data → 1D grid/block. `threadsPerBlock=256`, `blocksPerGrid=(n+255)/256`. |
| 3. Memory plan | Each byte of `A`/`B` is read exactly once, by exactly one thread — no reuse, so no shared-memory tiling is possible or needed. Arithmetic intensity is very low (one add per 12 bytes moved) → this kernel is **memory-bound** by construction; it could never be made meaningfully "faster" by clever arithmetic. |
| 4. Naive version | *Is* the final version — there's nothing further to correctly add. |
| 5-6. Host + verify | Standard malloc/cudaMalloc/memcpy/launch/memcpy/free skeleton (see `CUDA_EXECUTION_MODEL.md` and the `vecAddKernel_flow` diagram in `diagrams/`); verified against the exact expected constant `3.0f`. |
| 7-8. Profile/optimize | Nothing to optimize — a memory-bound, one-read-per-byte kernel is already at its structural ceiling. Recognizing *that it's memory-bound and therefore done* is itself the correct outcome of Phase 7, not a failure to find more work. |

Contrast that with where the *same* template lands for, say, a tiled matrix
multiply: Phase 0 becomes map+reduce instead of pure map, Phase 1's "one
thread" now reads an entire row *and* an entire column, Phase 3 finds heavy
data reuse (every element of a row is re-read by every thread computing a
different column), which is exactly what pulls shared-memory tiling out of
Phase 8 and into a near-mandatory part of even the *first* correct-and-decent
version. The questions didn't change. The problem's shape changed the
answers — which is the entire point of thinking this way from the start.
