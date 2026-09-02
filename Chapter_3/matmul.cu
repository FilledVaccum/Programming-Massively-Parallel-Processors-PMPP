#include<stdio.h>
#include<stdlib.h>

__global__ matmul() {

}

int main() {

	// Defining the shape of matrix
	int a_row = 3;
	int a_col = 2;
	int a_total_values = a_row * a_col;

	int b_row = 2;
	int b_col = 3;
	int b_total_values = b_row  * b_col;

	int c_row = 3;
	int c_col = 3;
	int c_total_values = c_row *  c_col;

	// Let's get them some space in CPU RAM
	int matrix_a[a_total_values];
	int matrix_b[b_total_values];
	int matrix_c[c_total_values];

	//let's initialize them with random values
	for ( int i = 0; i < a_total_values  ; i++ ) {
		matrix_a[a_total_values] = i % 10;
	}

        for ( int i = 0; i < b_total_values ; i++ ) {
                matrix_a[b_total_values] = i % 10;
        }

	// Now let's create a pointer where we will store the address of first memory that will have the values copied in GPU Memory
	int *a_matrix_deviceInput;
	int *b_matrix_deviceInput;
	int *c_matrix_deviceOutput;

	// Now allocating the memory at GPU
	cudaMalloc(&a_matrix_deviceInput, a_total_values * sizeof(int));
	cudaMalloc(&b_matrix_deviceInput, b_total_values * sizeof(int));
	cudaMalloc(&c_matrix_deviceOuput, c_total_values * sizeof(int));

        // Once Memory is allocated in GPU RAM - we will copy the values from CPU RAM TO GPU RAM 
	cudaMemcpy(a_matrix_deviceInput, matrix_a, a_total_values * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_matrix_deviceInput, matrix_b, b_total_values * sizeof(int), cudaMemcpyHostToDevice);

	// Now we have copied data from CPU RAM to GPU RAM - let's set up things for GPU to perform calculation in PARALLEL
	
	// Let's setup the Launch Configuration
	dim

}
