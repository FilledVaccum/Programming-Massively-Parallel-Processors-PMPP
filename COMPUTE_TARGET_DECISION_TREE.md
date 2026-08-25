# Which Processor Should Even Run This? — A Decision Tree for CPU vs. GPU vs. Other

This is the question that comes **before** [`KERNEL_DESIGN_FIRST_PRINCIPLES.md`](KERNEL_DESIGN_FIRST_PRINCIPLES.md)'s
Phase 0. That document assumes you've already decided "yes, this belongs on
a GPU" and walks you through designing the kernel. This document is about
**how to decide that in the first place** — and about not assuming a GPU
(or any particular hardware) is the answer just because it's the
fashionable one.

Treat this as **Phase −1**. Skipping it is how people spend a week porting
something to CUDA that a single CPU core would have finished, correctly, in
the time it took to write the `cudaMalloc` calls.

---

## 0. The one-sentence mental model

> A GPU is not "the fast processor." It is a **throughput machine**: it
> trades away single-task latency, branching flexibility, and low overhead
> in exchange for the ability to do enormous numbers of *simple, uniform,
> independent* operations at once. The question is never "is a GPU faster
> than a CPU?" in the abstract — it's **"does this specific workload have
> the shape a GPU is built to exploit, and is there enough of it to pay back
> the overhead of getting there?"** Most workloads, most of the time, do
> not. The ones that do, benefit enormously. Telling the two apart is the
> entire skill.

The same logic applies one level up, too: a distributed cluster trades
single-machine simplicity for scale, at the cost of network latency and
coordination overhead; an FPGA/ASIC trades general-purpose flexibility for
the lowest possible fixed latency and power, at the cost of engineering time
and the inability to change your mind later. **Every hardware choice is a
trade, not an upgrade.**

---

## 1. The gating questions, in order

Work top to bottom. Each question is a gate — as soon as one produces a
clear "no," you usually have your answer, or you've been routed to the next
relevant question. Don't skip ahead to "GPU vs. CPU" performance folklore
before answering these in order for *your* actual workload.

### Q1 — Is the workload dominated by *waiting*, not computing?

Waiting on disk I/O, network responses, user input, a database, another
service? Profile it (even roughly) before assuming otherwise — "feels slow"
is not evidence of being compute-bound.

- **Yes →** This was never a "which processor" question. Neither more CPU
  cores nor a GPU speeds up waiting. **Fix the concurrency model**
  (async I/O, event loops, more parallel I/O queues/connections, caching)
  instead. **Stop here.**
- **No, it's genuinely compute-bound →** continue to Q2.

### Q2 — Does the algorithm actually decompose into independent work?

Can you describe the computation as "do this same small operation, many
times, mostly without the instances needing each other's results"? Or is it
inherently a chain — each step strictly needs the previous step's output
(a tight recurrence, a sequential simulation stepped one tick at a time, an
unbatched recursive algorithm)?

- **No, it's inherently sequential →** More cores or a GPU's thousands of
  lanes buy you close to nothing — you'd be paying transfer/launch overhead
  to run mostly-idle hardware. The lever here is a **single fast CPU
  core** (highest clock speed, best single-thread IPC, best branch
  prediction) plus algorithmic improvements (a better big-O algorithm beats
  more parallel hardware every time). **Stop here — go optimize the
  algorithm or accept single-core CPU as the ceiling.**
- **Yes, real independent parallelism exists →** continue to Q3.

### Q3 — Is there *enough* independent work to pay back parallel-hardware overhead?

Every step away from "run it directly on one core" has a fixed overhead you
pay before seeing any benefit:

| Moving to... | Rough fixed overhead per dispatch |
|---|---|
| Another CPU thread | ~1-10 microseconds (thread wake/sync) |
| A GPU kernel launch + PCIe transfer | ~tens of microseconds launch + memory-transfer time proportional to data size |
| Another machine over the network | ~0.1-1+ millisecond round trip, plus coordination |

Rules of thumb (not laws — always profile a representative case, see Q7):

- **A few thousand simple operations or less, sub-millisecond total
  work →** the overhead of threading *or* a GPU launch is often bigger than
  the work itself. **Single-threaded CPU, plain loop.** This is not a
  simplification for beginners — it is frequently the objectively fastest
  and always the simplest correct answer at this scale.
- **Tens of thousands to a few million independent, lightweight
  operations, fits comfortably in one machine's RAM →** multi-core CPU
  territory; possibly GPU territory if the per-item work is regular (→ Q4).
- **Many millions to billions of independent, regular numeric
  operations, or needs to run continuously at high volume →** GPU is a
  strong candidate (→ Q4), or if it doesn't fit one machine/one GPU's
  memory, distributed territory (→ Q6).
- **Dataset or sustained workload exceeds a single machine's RAM/VRAM, or
  must run continuously across a fleet →** distributed/cluster computing
  (→ Q6), independent of whether individual nodes also use GPUs.

### Q4 — Is the per-item work *regular*? (only relevant if Q3 pointed at GPU-scale)

A GPU's throughput advantage depends on **thousands of threads doing
essentially the same thing at once**, reading memory in a pattern the
hardware can serve efficiently. Ask:

- Is control flow roughly **uniform** across items (few or no data-dependent
  branches that would make neighboring threads diverge onto different
  code paths)?
- Is memory access **regular/contiguous** (array indexing, tileable 2D/3D
  data) rather than **pointer-chasing** (linked lists, trees, hash tables,
  graphs with irregular adjacency)?
- Is the arithmetic intensity high enough that you're not purely waiting on
  memory transfer for work that's over in a few cycles? (A single add per
  element, like `vecAddKernel`, is legitimately GPU-shaped *and* still
  memory-bound — see the worked example below; that's fine, it's just a
  ceiling to be aware of, not a disqualifier.)

- **Yes, regular/dense/numeric →** **GPU is the right target.** Proceed to
  `KERNEL_DESIGN_FIRST_PRINCIPLES.md`, Phase 0.
- **No — heavy branching, pointer-chasing, data-dependent irregular
  structure →** A GPU's per-thread throughput advantage gets eaten alive by
  warp divergence and uncoalesced memory access; it can easily end up
  *slower* than a CPU here despite having far more raw FLOPs on paper.
  Options, in order of preference:
  1. **Restructure the algorithm into a more regular form** first (e.g.
     level-synchronous/batched BFS instead of pointer-chasing DFS, sorting
     or bucketing irregular data before processing it) and *then*
     reconsider the GPU.
  2. If restructuring isn't practical, **multi-core CPU** — its bigger
     caches, out-of-order execution, and branch prediction are specifically
     good at exactly the irregular case a GPU is bad at.

### Q5 — (Medium-scale branch from Q3) Does it vectorize, or does a library already exist?

For the "tens of thousands, fits on one machine, not clearly GPU-scale"
middle ground:

- **Dense numeric loops over contiguous data, or a well-known operation
  (linear algebra, FFT, sorting, standard ML layers) →** check for a
  **mature, hardware-tuned library first** (BLAS/LAPACK/Eigen/MKL/oneDNN on
  CPU; a hand-rolled loop competing with decades of tuned assembly is
  usually a losing and unnecessary fight). If no library fits, **SIMD
  vectorization** (the compiler auto-vectorizing a clean loop, or
  explicit AVX/NEON intrinsics as a last resort) plus a **thread pool**
  across cores is the pragmatic answer.
- **Independent but irregular/branchy tasks, or I/O-adjacent work mixed
  in →** plain **task-based multi-threading** (thread pool, work queue) —
  no need for SIMD or a GPU; the win is just "use more cores."

### Q6 — (Massive-scale branch from Q3) What kind of "massive"?

"Too big for one machine" splits several ways, and they lead to different
hardware:

- **Dense, regular, numeric, and the same narrow operation repeated at
  extreme, sustained volume** (e.g., training/serving large neural
  networks, large batch scientific/genomics pipelines) **→**
  **multi-GPU / multi-node GPU cluster** is the standard answer. If the
  operation set is *very* narrow and fixed (dense matrix multiply +
  a small set of activation functions, run at hyperscale, for years) a
  **TPU or other AI ASIC** can beat general-purpose GPUs on
  performance-per-watt and performance-per-dollar for that exact workload —
  but only pays off at a scale where the fixed non-recurring engineering
  cost of targeting it is worth it.
- **Deterministic, ultra-low, fixed latency requirement with a fixed
  computation** (line-rate network packet processing, real-time control
  loops, high-frequency-trading tick-to-trade paths, signal processing in
  hardware) **→ FPGA or a custom ASIC.** You hard-wire the exact
  computation with zero general-purpose overhead, at the cost of long
  development time and near-zero flexibility to change the algorithm later.
- **"Big data" where the per-record computation itself is simple/branchy**
  (log parsing, ETL, filtering/joining records, business rules) **→**
  the bottleneck is I/O and orchestration, not FLOPs. A **distributed
  CPU cluster** (Spark/Dask/similar data-parallel frameworks) is the right
  shape — adding GPUs here usually helps nothing because the per-record work
  never had the "thousands of uniform numeric ops" shape a GPU wants.

---

## 2. ASCII summary of the tree

```
Q1: Dominated by waiting (I/O/network)?
 ├─ YES → fix concurrency model (async I/O). STOP.
 └─ NO
     Q2: Does the algorithm decompose into independent work?
      ├─ NO (sequential/recurrence) → single fast CPU core + better algorithm. STOP.
      └─ YES
          Q3: How much independent work, relative to overhead?
           ├─ Tiny (≲ thousands of ops / sub-ms total) → single-thread CPU. STOP.
           ├─ Medium (10^4–10^6, fits one machine)
           │    Q5: Vectorizable / library exists?
           │     ├─ YES → CPU + SIMD, or call BLAS/FFTW/MKL/etc.
           │     └─ NO  → CPU thread pool (task parallelism)
           ├─ Large (10^6–10^9+ regular numeric ops, fits one GPU's memory)
           │    Q4: Is the per-item work regular (uniform branches,
           │        coalesced memory), not pointer-chasing?
           │     ├─ YES → GPU. Go to KERNEL_DESIGN_FIRST_PRINCIPLES.md, Phase 0.
           │     └─ NO  → restructure into regular form and re-check Q4,
           │               else multi-core CPU (better at irregular workloads)
           └─ Massive (exceeds one machine/GPU, or must run at fleet scale)
                Q6: What kind of massive?
                 ├─ Dense numeric at extreme sustained volume → GPU cluster
                 │    (narrow fixed tensor ops at hyperscale → TPU/ASIC)
                 ├─ Fixed computation, deterministic ultra-low latency → FPGA/ASIC
                 └─ Simple/branchy per-record, I/O/orchestration-bound → distributed CPU cluster
```

---

## 3. Cross-cutting factors that override the tree's default answer

The tree above optimizes for raw throughput-per-workload-shape. These
factors can legitimately change the answer even when the tree points
elsewhere — an expert checks these *before* committing:

- **Does an optimized library already exist for this exact operation?**
  If `cuBLAS`/`cuDNN`/`cuFFT` (GPU) or `MKL`/`OpenBLAS`/`FFTW`/`Eigen` (CPU)
  already implements it, use that instead of hand-writing a kernel or a
  hand-rolled loop, on *either* hardware target — vendor libraries are
  usually near the achievable performance ceiling and writing your own is
  rarely worth the risk of subtle bugs, regardless of which processor you
  land on.
- **Development time is a resource too.** A correct CPU solution shipped
  today frequently beats a faster GPU/distributed solution shipped in three
  weeks, especially for one-off analyses, prototypes, or low-traffic code
  paths. Factor "how many times will this run, and how much engineering
  time is available" into the decision, not just steady-state runtime.
- **Latency vs. throughput.** Even a workload with a GPU-shaped *algorithm*
  can be wrong for GPU if it's a **single request needing sub-millisecond
  response** — kernel-launch and PCIe round-trip overhead alone (tens to a
  few hundred microseconds) can dwarf a tiny amount of actual compute.
  GPUs win on **batched throughput**; if requests arrive one at a time and
  must return immediately, staying on CPU (or batching requests before
  sending them to the GPU) is usually correct.
- **Precision and determinism requirements.** GPUs and especially
  TPUs/AI ASICs often push toward reduced precision (fp16/bf16/tf32/int8)
  for speed. If the domain requires strict, reproducible fp64 behavior
  (certain scientific simulations, financial ledger arithmetic, anything
  requiring bitwise-reproducible results across runs/hardware), that
  constrains — sometimes rules out — those accelerators, or forces explicit
  precision-mode choices.
- **Power, thermal, and cost envelope.** Embedded, mobile, or
  battery-powered contexts may rule out a discrete GPU or a cluster
  entirely regardless of the workload's shape — a low-power CPU core or a
  small on-device NPU may be the only hardware actually available, and the
  question becomes "how do I fit this workload into that budget," not
  "what's theoretically fastest."

---

## 4. The meta-principle: measure before you migrate

This ties directly back to Phase 0 of `KERNEL_DESIGN_FIRST_PRINCIPLES.md`:
**write the simplest single-threaded CPU version first, and profile it,**
before deciding you need parallelism, a GPU, or a cluster at all. Three
things this buys you that guessing doesn't:

1. **A correctness baseline** to check every later, fancier version against.
2. **Actual numbers** — total runtime, where the time goes — instead of
   intuition about where you *assume* the time goes. Profilers routinely
   surprise experienced engineers.
3. **Evidence for whether you're even in a regime where hardware
   parallelism helps**, per Q2/Q3 above, instead of a guess.

If the naive CPU version is already fast enough for the requirement, that's
not a failure to optimize — that's the decision tree working correctly and
saving you a costly, unnecessary migration.

---

## 5. Worked example: was `vecAddKernel.cu` even a good candidate for GPU?

Applying the tree honestly to this repo's own example, at its actual size
(`n = 10,000` floats):

- **Q1 (I/O-bound?)** No — pure arithmetic, no I/O in the hot path.
- **Q2 (decomposes into independent work?)** Yes — textbook **Map**, zero
  cross-element dependency (see `KERNEL_DESIGN_FIRST_PRINCIPLES.md` §Phase
  0's dependency-pattern table).
- **Q3 (enough work to pay back overhead?)** This is the honest part: at
  `n = 10,000`, total useful work is 10,000 floating-point additions —
  a single CPU core executes that in well under a microsecond of pure
  compute. The GPU path pays a kernel-launch cost *and* two PCIe transfers
  (`A`,`B` in, `C` out) that are each individually likely to take **longer**
  than the CPU would take to just run the loop directly. At this exact
  size, **a plain CPU `for` loop is very plausibly faster end-to-end than
  the GPU version**, once transfer overhead is counted honestly.
- **Q4 (is it regular?)** Yes, maximally — which is *why* it's the right
  **teaching example** for kernel mechanics (Map is the simplest
  dependency pattern to reason about) even though it's a **poor example**
  of when you'd actually reach for a GPU in production. Those are different
  claims, and conflating them is a common beginner mistake: "this kernel
  is correct and idiomatic" is not the same claim as "this was worth
  putting on a GPU at this data size."
- **Where a GPU earns its overhead back on this exact algorithm:** the
  *shape* stays identical (still pure Map) once `n` grows into the
  hundreds of thousands to millions — real vector-add-shaped workloads at
  that scale (large-scale numeric pipelines, one stage of a bigger GPU
  pipeline where the data is already resident on the device from a prior
  kernel) are exactly where this becomes a genuinely good GPU candidate,
  because the same low arithmetic intensity that makes it "boring"
  computationally also means the GPU's memory bandwidth advantage (far
  higher than a CPU's) is the entire game — and at large `n` that bandwidth
  advantage finally outweighs the fixed launch/transfer overhead.

The lesson generalizes: **the dependency-pattern classification tells you
the algorithm's *shape*; the decision tree in this document tells you
whether the *scale* justifies moving that shape to different hardware.**
You need both answers, and they can point in different directions even for
the same, perfectly-written kernel.
