#include <stdio.h>

__global__ void sayHello() {
	printf("I am a thread %d\n", threadIdx.x);
}

int main() {
	sayHello<<<2, 16>>>();
//	cudaDeviceSynchronize();
	return 0;
}
