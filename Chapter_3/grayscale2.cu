#include<stdio.h>

__global__ void grayscale2(unsigned char *input, unsigned char *output, int width, int height) { // No of parameters here would be equal to argument passed while calling this function in main function

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( row < height && col < width) {
		int pixelIndex = row * width + col;
		unsigned char r = input[pixelIndex * 3 + 0];
		unsigned char g = input[pixelIndex * 3 + 1];
		unsigned char b = input[pixelIndex * 3 + 2]; 
		unsigned char gray = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
		output[pixelIndex] = gray;
	}
}


// Host Function - main function
int main() {

	// This is where we are defining the values of image size in dimensions width and height	
	int width = 8;
	int height = 4;
	int numPixels = width * height;

	// But the input image would be coloured so we need 3 channels( R G B ) per pixel
	unsigned char hostInput[96]; // (8*4) * 3 - Also unsigned char because pixel value are from 0 - 255 and char = 1 byte = 8 bits = 2 power 8 = 256 possible values - basically occuopying less memory

	// To put in random values
	for ( int i =0; i < numPixels * 3; i++) {
		hostInput[i] = i % 256; // to keep the values between 0 and 256
	}

	unsigned char hostOutput[32]; // These would be final value - one value per pixel

	
	// ABove everything happen for CPU and CPU RAM  and on CPU
	// Now the below code would be executed on CPU but may perform operation on GPU/GPU Memory

	unsigned char *deviceInput; // These pointers are create on CPU - These are variable that will hold GPU memory address as a value
	unsigned char *deviceOutput;

	// Allocating the space and memory to store the values in GPU
	cudaMalloc(&deviceInput, numPixels * 3 * sizeof(unsigned char)); // Here cudaMalloc is a function that says somethile like this - Hey GPU, I need some space to reserve, and the space would be equal to numPixels * 3 * sizeof(unsigned char and store the starting address of that continous space at *deviceInput 
	cudaMalloc(&deviceOutput, numPixels * sizeof(unsigned char)); //cudaMalloc needs to modify your pointer (fill in the address). In C, to let a function modify your variable, you pass its address with &. Without &, cudaMalloc would get a COPY of the pointer and your original would stay empty.

	// Once memory is allocated and values are in - copy those values from CPU to GPU
	cudaMemcpy(deviceInput, hostInput, numPixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice); // Here the values are copied from host to device input values and this happens over PCIe - this is much slower thank nvlink(Which does samething but among GPU Chips)


	
	// Launch COnfiguration and kernel
	dim3 blockSize(16, 16);
	dim3 gridSize((width + 15)/16, (height + 15)/16); // doing +15 to do a ciel to - get enough blocks

	grayscale2<<<gridSize, blockSize>>>(deviceInput, deviceOutput, width, height); // this is kernel - think of like a way which will launch (threads) amount of funtions and all of them would have same paramter passed down 

	cudaDeviceSynchronize();
	
	// Copy result back from GPU to CPU
	cudaMemcpy(hostOutput, deviceOutput, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost); // cudaMemcpy(TO, FROM, HOW_MUCH, WHICH_DIRECTION);


	// Free GPU Memory - Because it's costly
	cudaFree(deviceInput);
	cudaFree(deviceOutput);


	for ( int i = 0; i < numPixels; i++) {
		printf("Pixel %d: gray = %d\n", i, hostOutput[i]);
	}

	return 0;
}
