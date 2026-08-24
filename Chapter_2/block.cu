#include <stdio.h>

// Code for Basic Block
__global__ void learning_block_basic() {
//	printf("We are in - Learning Block Basic Function - : \n");
	printf("Here the Block %d, Thread %d, GlobalId is %d\n", blockIdx.x, threadIdx.x,  blockIdx.x * blockDim.x + threadIdx.x);
}


// Code for Grid
__global__ void learning_block_grid() {
//	printf("We are in - Learning Block Grid Function - : \n");
	printf("Block %d/%d, Thread %d/%d, Global %d/%d\n",
		blockIdx.x, gridDim.x,
		threadIdx.x, blockDim.x,
		blockIdx.x * blockDim.x + threadIdx.x,
		gridDim.x * blockDim.x);
}

//Code for Wraps
__global__ void learning_block_wrap() {
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	printf("Global %3d | Block %d | Warp %d | Lane %2d\n",
		globalId, blockIdx.x, warpId, laneId);
}

int main() {
        printf("We are in - Learning Block Basic Function - : \n");
	learning_block_basic<<<4,8>>>();
        printf("We are in - Learning Block Grid Function - : \n");
	learning_block_grid<<<4,8>>>();
//	learning_block_basic<<<1, 1025>>>();
	cudaDeviceSynchronize();

	printf("We are in - Learning Block Basic Function - : \n");
	learning_block_basic<<<4,8>>>();
	cudaDeviceSynchronize();  // ← WAIT for first kernel to finish

	printf("We are in - Learning Block Grid Function - : \n");
	learning_block_grid<<<4,8>>>();
	cudaDeviceSynchronize();  // ← WAIT for second kernel to finish

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	return 0;
}
