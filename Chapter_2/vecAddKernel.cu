#include <stdio.h>

//Step 1: Kernel Function
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

//Step 2: Host Function 
void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
	// Part 1: Allocate device memory and copy input data
	float *A_d, *B_d, *C_d;
	int size = n * sizeof(float);	//GPU doesn't know about "elements" — it needs exact byte count
	
	cudaMalloc((void**)&A_d, size);		//cudaMalloc needs the address of your pointer so it can write the GPU address into it
	cudaMalloc((void**)&B_d, size); 
	cudaMalloc((void**)&C_d, size);
	
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	// Part 2: Kernel Launch
	int threadsPerBlock = 256;
	int blocksPerGrid = (n+ threadsPerBlock -1)/threadsPerBlock;
	vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

	// Part 3: Copy results back and free back the memory
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
	
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

//Step 3: Main Function
int main() {
	int n = 10000;
	int size = n * sizeof(float);

	//Allocate host Memory
	float *A_h = (float*)malloc(size);
	float *B_h = (float*)malloc(size);
	float *C_h = (float*)malloc(size);


	// Initialize input vectors
	for ( int i =0; i < n; i++) {
		A_h[i] = 1.0f;
		B_h[i] = 2.0f;
	}

	// Call the host function ( which handles GPU work)
	vecAdd(A_h, B_h, C_h, n);
	
	
	// verify result
	int correct = 1;
	for ( int i =0 ; i < n; i++) {
		if (C_h[i] != 3.0f) {
			printf("Error at the index %d: %f\n", i , C_h[i]);
			correct = 0;
			break;
		}
	}
	
	if (correct) {
		printf("SUCCESS - All %d elements are correct. \n", n);
	}

	
	// Free Host Memory
	free(A_h);
	free(B_h);
	free(C_h);

//	printf("Hello CUDA!\n");
	return 0;
}
