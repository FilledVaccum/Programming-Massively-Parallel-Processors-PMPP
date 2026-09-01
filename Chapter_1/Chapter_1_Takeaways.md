A few core ideas that I was able to capture in this Chapter:

1. Two design philosophies: CPUs are latency-oriented (make each thread fast), GPUs are throughput-oriented (make many threads finish collectively fast — at the cost of individual thread latency)

2. Amdahl's Law: The achievable speedup is bounded by the sequential fraction. If only 30% of your app is parallelizable, even infinite speedup on that portion caps you at ~1.43x overall.

3. The real bottleneck is often memory, not compute: Naive parallelization saturates DRAM bandwidth (~10x speedup ceiling), and getting beyond that requires on-chip memory management tricks.  - If you want to relate to it - you can see why memory pricing are increasing and way higher than compute - So in GenAI world - when you want to run your model - You need memory to load model weights in GPU VRAM - which could be in TB for large models and you also need memory to store the KV Cache.

4. Four challenges that will be talked about in upcoming chapters would be around: work efficiency, memory access optimization, handling irregular data distributions, and synchronization overhead.