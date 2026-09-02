#include<stdio.h>
#include<stdlib.h>

__global__ void blur(unsigned char *input, unsigned char *output, int width, int height) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelIndex = row * width + col;

	if ( row < height && col < width ) {
		int sum = 0;
		int count = 0;
		for ( int dy = -1; dy <= 1; dy++) {
			for ( int dx = -1; dx <= 1; dx++) {
				int ny = row + dy;
				int nx = col + dx;
				if ( ny >=0 && ny < height && nx >=0 && nx < width) {
					int localIndex = ny * width + nx;	//local index in 3 * 3 grid
					sum = sum + input[localIndex];
					count = count + 1;
				}
			}
		}
		output[pixelIndex] = (unsigned char)(sum/count);
	}
}

int main() {

	// Let's create a random image with grayscale 
	int width = 10000;
	int height = 10000;
	int numPixel = width * height;

	// Now we have defined the shape and number of pixel of grayscale image
	// Now let's get some memory in CPU - 
	// unsigned char hostInput[numPixel]; //it crashes with large size - I tried i got segmentaiton fault
	unsigned char *hostInput = (unsigned char *)malloc(numPixel * sizeof(unsigned char));	

	// Now we have got space in memory - let's fill them up with random value
	for ( int i = 0; i < numPixel; i++) {
		hostInput[i] = i % 256;
	}

	// Now we have initialized the values 
	// Now let's get some space reserved in CPU RAM to save the output value - that would be calcualted by GPU and copied to CPU
	//unsigned char hostOutput[numPixel];
	unsigned char *hostOutput = (unsigned char *)malloc(numPixel * sizeof(unsigned char));

	// Till this point we have defined the shape of image(grayscale) and got memory for input - initialized the values - and memory for the output
	// Now we will start working towards GPU
	unsigned char *deviceInput; // These pointers are create on CPU - These are variable that will hold GPU memory starting address
	unsigned char *deviceOutput; 

	// Now we will actually allocate memory at GPU
	cudaMalloc(&deviceInput, numPixel * sizeof(unsigned char));
	cudaMalloc(&deviceOutput, numPixel * sizeof(unsigned char));

	// Once Memory is allocated in GPU RAM - we will copy the values from CPU RAM TO GPU RAM
	cudaMemcpy(deviceInput, hostInput, numPixel * sizeof( unsigned char), cudaMemcpyHostToDevice);
	
	// Now we have copied data from CPU RAM to GPU RAM - let's set up things for GPU to perform calculation in PARALLEL
	
	// Let's set up the Launch Configuration
	dim3 blockSize(16, 16);
	dim3 gridSize( (width+15)/16, (height+15)/16 );
	
	// Now calling the kernel function and passing the arguments
	blur<<<gridSize, blockSize>>>(deviceInput, deviceOutput, width, height);

	//While Calculation is happening we will wait for it ( CPU WILL WAIT FOR IT by using following function)
	cudaDeviceSynchronize();
	
	//Once Calculation is done - we need to copy value from GPU RAM TO CPU RAM
	cudaMemcpy(hostOutput, deviceOutput, numPixel * sizeof( unsigned char), cudaMemcpyDeviceToHost);

	//Free GPU Memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);

	// Limiting the output to 30 values
	for ( int i = 0; i < 30; i++) {
		printf("Pixel %d: blur = %d\n", i, hostOutput[i]);
	} 	

}
