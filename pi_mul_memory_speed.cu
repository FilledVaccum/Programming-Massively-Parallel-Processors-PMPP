#include<stdio.h>
#include<cuda_runtime.h> // I tried to check why cuda_runtime is needed as I haven't used cuda_runtime in chapter 3 code, so when .cu is complied using nvcc, nvcc aut inject this cuda_runtime.h but if you use gcc complier and compile as c code it will not do the same so it's good habit to include cuda_runtime.h


// The trick: every kernel does the EXACT SAME number of
// add operations (LOOP times). The ONLY difference is WHERE
// the running value is stored between iterations.



#define N (1024*1024) // no of threads
#define LOOP 10000 // how many time the thread toches the memory
#define THREADS 256 // this is no of threads per block

//=================================
// Kernel 1 - Global Memory
// The variable thjat store the let's say total sum lives on vram
// Every iteration reads from VRAM and writes back to VRAM.

__global__ void globalMemKernel(int *data) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < N) {
		for (int k = 0; k < LOOP; k++) {
			data[i] = data[i] * 1; // read VRAM, add, write VRAM (every loop!)
		}
	}
}

//=================================
// KERNEL 2: SHARED MEMORY
// The variable thjat store the let's say total sum lives on Shared Memory
// Each thread uses one slot in the block's shared array.

__global__ void sharedMemKernel(int *data) {
	__shared__ int cache[THREADS]; //here I am telling GPU I need a shared memory that every thread can see and I need the shared mempry on chip and i need 256 int type memory blocks
	

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = threadIdx.x;

	if ( i < N) {
		cache[t] = data[i]; //here I am load once from VRAM -> shared
		for ( int k = 0; k < LOOP; k++) {
			cache[t] = cache[t] * 1; //read shared, add, write shared (fast!)
		}
		data[i] = cache[t]; // loading back from shared memory to vram
	}
	
}

//===================================
// KERNEL 3: REGISTER
// The variable thjat store the let's say total sum lives on Register

__global__ void registerKernel(int *data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( i < N) {
		int acc = data[i];	// load once from VRAM -> register

		for(int k = 0; k < LOOP; k++) {
			acc = acc * 1; // read register, add, write register (0 cycles!)
		}
		data[i] = acc; // store once back to VRAM
	}
}

//======================
// Helper Functin to time
// this is interesting as this is a function with parameter 1.) a function which take int data type and returns nothing, 2.) pointer to data first memeory address
// 3.) vaiable of type int that capture block values, 4.) vaiable of type int that capture threads values
float timeKernel( void (*kernel)(int*), int *data, int blocks, int threads) {

	cudaEvent_t start, stop; // so cudaEvent_t is a data type and it declares a variable that will hold an event
	cudaEventCreate(&start); //build them (& = fill me in)
	cudaEventCreate(&stop);
	
	cudaEventRecord(start); //mark start time on GPU
	kernel<<<blocks, threads>>>(data); //Run kernel on GPU
	cudaEventRecord(stop); //mark sotp time on GP

	cudaEventSynchronize(stop); 
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return ms;

}



//===================================
// Main Function

int main() {
	
	int blocks = (N + THREADS - 1)/THREADS; // It like a ciel function
	
	// Allocating data in GPU VRAM
	int *data;	 //created a pointer that will stored the starting address of memory location in GPU that will have actual values of data
	cudaMalloc(&data, N * sizeof(int));	//allocating memory for the data
	cudaMemset(data, 0, N * sizeof(int)); // filling with zero


	printf("Config: %d threads, each does %d add operations\n", N, LOOP);

	printf("Total operations: %lld\n\n", (long long)N * LOOP);

	// --- Warm-up run (first launch is always slower: driver init, caching) ---
	registerKernel<<<blocks, THREADS>>>(data);
	cudaDeviceSynchronize();

	// --- Time each memory tier ---
	float tGlobal = timeKernel(globalMemKernel, data, blocks, THREADS);
	float tShared = timeKernel(sharedMemKernel, data, blocks, THREADS);
	float tReg = timeKernel(registerKernel, data, blocks, THREADS);

	// --- Results ---

	printf("=================================================\n");

	printf(" Memory Tier | Time (ms) | Relative Speed\n");

	printf("=================================================\n");

	printf(" Global (VRAM) | %8.3f | 1x (baseline)\n", tGlobal);

	printf(" Shared (on-SM) | %8.3f | %.1fx faster\n", tShared, tGlobal / tShared);

	printf(" Register (core) | %8.3f | %.1fx faster\n", tReg, tGlobal / tReg);

	printf("=================================================\n\n");

	printf("WHY:\n");

	printf(" - Global: every one of the %d loops hits VRAM (~400 cycles each)\n", LOOP);

	printf(" - Shared: loads from VRAM ONCE, then loops in shared (~5 cycles each)\n");

	printf(" - Register: loads from VRAM ONCE, then loops in a register (0 cycles each)\n");

	cudaFree(data);

	return 0;
}
