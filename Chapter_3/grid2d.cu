#include <stdio.h>

__global__ void grid2d() {

	// Calculate which row and column this thread is responsible for
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Only print some threads (to keep output readable)
	if (row < 4 && col < 8) {
		printf("Block(%d, %d) Thread(%d, %d) -> Row=%d, Col=%d\n",
		blockIdx.y, blockIdx.x,
		threadIdx.y, threadIdx.x,
		row, col);
	}
}

int main() {
	dim3 blockSize(4, 2);	// 4 threads in x (cols), 2 threads in y (rows)
	dim3 gridSize(3, 2);	// 3 blocks in x, 2 blocks in y

	grid2d<<<gridSize, blockSize>>>();
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}
	
	return 0;
}
