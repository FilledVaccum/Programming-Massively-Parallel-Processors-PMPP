#include <stdio.h>

// A function that will run on GPU but can be called by Host CPU
__global__ void linearize(int width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = row * width + col;

	if (row < 4 && col < 8) {
		printf("Row %d, Col %d -> Index %d\n", row, col, index);
	}
}

// Main Function
int main () {
	int width = 8;
	
	dim3 blockSize(4, 2);
	dim3 gridSize(2, 2);

	linearize<<<gridSize, blockSize>>>(width);
	cudaDeviceSynchronize();
	return 0;
}


