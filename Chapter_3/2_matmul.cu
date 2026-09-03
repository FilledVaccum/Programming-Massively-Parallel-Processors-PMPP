#include<stdio.h>
#include<stdlib.h>

__global__ void matmul(int *A, int *B, int *C, int M, int K, int N) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if( row < M && col < N) {
		int sum = 0;
		for ( int k = 0 ; k < K; k++) {
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

int main() {

	// Defining the shape of matrix
	int a_row = 3000;
	int a_col = 200;
	int a_total_values = a_row * a_col;

	int b_row = 200;
	int b_col = 3000;
	int b_total_values = b_row  * b_col;

	int c_row = a_row;
	int c_col = b_col;
	int c_total_values = c_row *  c_col;

	// Let's get them some space in CPU RAM
//	int matrix_a[a_total_values];
//	int matrix_b[b_total_values];
//	int matrix_c[c_total_values];

	int *matrix_a = (int *)malloc(a_total_values * sizeof(int));
	int *matrix_b = (int *)malloc(b_total_values * sizeof(int));
	int *matrix_c = (int *)malloc(c_total_values * sizeof(int));
	
	//let's initialize them with random values
	for ( int i = 0; i < a_total_values  ; i++ ) {
		matrix_a[i] = i % 10;
	}

        for ( int i = 0; i < b_total_values ; i++ ) {
                matrix_b[i] = i % 10;
        }

	// Now let's create a pointer where we will store the address of first memory that will have the values copied in GPU Memory
	int *a_matrix_deviceInput;
	int *b_matrix_deviceInput;
	int *c_matrix_deviceOutput;

	// Now allocating the memory at GPU
	cudaMalloc(&a_matrix_deviceInput, a_total_values * sizeof(int));
	cudaMalloc(&b_matrix_deviceInput, b_total_values * sizeof(int));
	cudaMalloc(&c_matrix_deviceOutput, c_total_values * sizeof(int));

        // Once Memory is allocated in GPU RAM - we will copy the values from CPU RAM TO GPU RAM 
	cudaMemcpy(a_matrix_deviceInput, matrix_a, a_total_values * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_matrix_deviceInput, matrix_b, b_total_values * sizeof(int), cudaMemcpyHostToDevice);

	// Now we have copied data from CPU RAM to GPU RAM - let's set up things for GPU to perform calculation in PARALLEL
	
	// Let's setup the Launch Configuration
	dim3 blockSize(16, 16);
	dim3 gridSize((c_row+15)/16, (c_col+15)/16);

	//Now calling the kernel function and passing the argument
	matmul<<<gridSize, blockSize>>>(a_matrix_deviceInput, b_matrix_deviceInput, c_matrix_deviceOutput, a_row, a_col, b_col); //a_col and b_row have same value

	//We have to ask CPU to wait for GPU task completion
	cudaDeviceSynchronize();
	
	//Once Calcualtion is done on GPU - get the output at CPU to present at screen
	cudaMemcpy(matrix_c, c_matrix_deviceOutput, c_total_values * sizeof(int), cudaMemcpyDeviceToHost);

	//Free GPU Memory
        cudaFree(a_matrix_deviceInput);
        cudaFree(b_matrix_deviceInput);
        cudaFree(c_matrix_deviceOutput);

        for ( int i = 0; i < 10; i++) {
                printf("Index %d: Value = %d\n", i, matrix_c[i]);
        }  

	//Free CPU RAM
	free(matrix_a);
	free(matrix_b);
	free(matrix_c);
}
