**Chapter 3 Final Exam — 30 Questions**

No peeking at the reference doc. Write all answers, then I'll grade.

* * *

**Section A: Thread Identity (5 questions)**

**A1.** What is a thread? (one sentence — not data, not an index) a thread is a function that fetched from GPA perform operations on that and then save the final value to again GPU RAM

**A2.** A thread computes `i = 5`. It then reads `data[5]` which contains the value `42`. Label each: thread, index, data.
The thread is five. The index is five and the data is 42.

**A3.** Where does the variable `int i = 5` physically live during execution? (register, shared memory, VRAM, or CPU RAM?)
During the physical live execution, the variable stays in register 

**A4.** Where does `data[5] = 42` physically live? (same options)
The actual data stays in GPU VRAM, but it moves to register for operation

**A5.** Two threads on different SMs both write to `output[100]`. What problem occurs and what's the fix?
When two threats on different streaming multiprocessor both right to same variable than in that case, there could be inconsistency in the value of the variable, as if I remember correctly, the fixes to basically put a lock on the variable while one operation is being performed

* * *

**Section B: dim3 and Launch Config (5 questions)**

**B1.** `dim3 blockSize(32, 8)` — how many rows and columns of threads per block?
This will have 32 columns and eight rows per block

**B2.** You have data with 1080 rows × 1920 columns. Write the two `dim3` lines (blockSize and gridSize) using 16×16 blocks.
In this case, the block size would be 16 cross 16 and the great size would be 1080÷16 and 1920÷16

**B3.** Why do we write `(width + 15) / 16` instead of `width / 16`?
Rewrite it because we want to make sure that the great size covers all the values across with and column and we write press 15 because we want to have the ciel feature

**B4.** `dim3 gridSize(120, 68)` — how many total blocks?
In order to calculate total of blocks, we can divide 120×68÷16*16

**B5.** With `blockSize(16,16)` and `gridSize(120, 68)`, how many total threads? How many are "extra" beyond 1920×1080?


* * *

**Section C: Formulas (5 questions)**

**C1.** Write the formula for `col` using CUDA built-in variables.
col = blockIDx.x * blockDim.x + threadIdx.x

**C2.** Write the formula for `row`.
row = blockIDx.y * blockDim.y + threadIdx.y

**C3.** Write the row-major linearization formula (2D → 1D index). What does `width` represent?
index = row * width + col - width represent a complete row

**C4.** Thread at `blockIdx=(3, 5)`, `threadIdx=(7, 12)`, `blockSize(16, 16)`. What is col, row, and index? (data width = 200)
col = 3*16 + 7
row = 5*16 + 12
index = row * 200 + col

**C5.** Pixel at index 847 in a 100-column image. What is its row and col?
here width = 100
row = 8 
col = 47


* * *

**Section D: Memory & Pointers (5 questions)**

**D1.** Fill in the blank with the correct symbol (`*`, `&`, or nothing):

unsigned char \_\_\_deviceInput; *

cudaMalloc(\_\_\_deviceInput, 100); &

kernel<<<1,1>>>(\_\_\_deviceInput); 

\_\_global\_\_ void kernel(unsigned char \_\_\_input) { ... } * 

**D2.** Stack vs Heap — when does a stack array crash?
because OS by default give it only 8 MB of space

**D3.** Why does `cudaMalloc` need `&deviceInput` but the kernel call doesn't?
I am not sure

**D4.** List the 7 memory levels from slowest to fastest.
I Don't remember


**D5.** A read to VRAM costs ~400 cycles. A register access costs how many?
one

* * *

**Section E: Kernel Design (5 questions)**

**E1.** Apply the 5-step thought process to: "Subtract two 500×500 matrices: C\[i\] = A\[i\] - B\[i\]". Write all 5 steps.

**E2.** A grayscale pixel at (row=5, col=12) in a color image (width=100). Where are R, G, B in the input array? Where does gray go in the output array?

**E3.** In the blur kernel, why do we need TWO levels of boundary checking?

**E4.** In matrix multiplication, what does the `k` loop do? How many iterations for A(128×256) × B(256×64)?

**E5.** Write the complete matmul kernel body (just the `__global__` function — not main).

* * *

**Section F: Profiling & Performance (5 questions)**

**F1.** Your ncu report shows Compute Throughput = 2%, Occupancy = 11%. What's the likely cause?

**F2.** Your ncu report shows Compute = 99.6%, Memory = 99.6%. What does this mean?

**F3.** Why did your 3×2 × 2×3 matmul show 0.02% compute, but 300×200 × 200×300 showed 79%?

**F4.** Name the 6 rules for writing a fast kernel (in priority order).

**F5.** `nvcc -arch=sm_89 -o blur blur.cu` — what does `-arch=sm_89` mean and why is it needed?

* * *

Take your time. All 30 questions.