#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHANNELS 3

// =============================================
// CUDA Kernel: Color to Grayscale Conversion
// =============================================
__global__
void colorToGrayscaleConversion(unsigned char *Pout,
                                 unsigned char *Pin,
                                 int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // 1D offset for the grayscale output (1 byte per pixel)
        int grayOffset = row * width + col;

        // 1D offset for the RGB input (3 bytes per pixel)
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset    ];  // Red
        unsigned char g = Pin[rgbOffset + 1];  // Green
        unsigned char b = Pin[rgbOffset + 2];  // Blue

        // Weighted sum for human-perceived brightness
        Pout[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

// =============================================
// Host Code
// =============================================
int main() {
    // Image dimensions
    int width  = 76;
    int height = 62;
    int numPixels = width * height;

    // Allocate host memory
    unsigned char *h_Pin  = (unsigned char *)malloc(numPixels * CHANNELS * sizeof(unsigned char));
    unsigned char *h_Pout = (unsigned char *)malloc(numPixels * sizeof(unsigned char));

    // Fill input with a simple test pattern (gradient)
    for (int i = 0; i < numPixels; i++) {
        h_Pin[i * 3 + 0] = (unsigned char)(i % 256);         // R
        h_Pin[i * 3 + 1] = (unsigned char)((i * 2) % 256);   // G
        h_Pin[i * 3 + 2] = (unsigned char)((i * 3) % 256);   // B
    }

    // Allocate device memory
    unsigned char *d_Pin, *d_Pout;
    cudaMalloc((void **)&d_Pin,  numPixels * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void **)&d_Pout, numPixels * sizeof(unsigned char));

    // Copy input image to device
    cudaMemcpy(d_Pin, h_Pin, numPixels * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch kernel with 16x16 thread blocks
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);

    printf("Grid  dimensions: (%d, %d)\n", dimGrid.x, dimGrid.y);
    printf("Block dimensions: (%d, %d)\n", dimBlock.x, dimBlock.y);
    printf("Total threads:    %d x %d = %d\n", dimGrid.x * dimBlock.x, dimGrid.y * dimBlock.y,
           dimGrid.x * dimBlock.x * dimGrid.y * dimBlock.y);
    printf("Image pixels:     %d x %d = %d\n", width, height, numPixels);
    printf("\n");

    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(d_Pout, d_Pin, width, height);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_Pout, d_Pout, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Print first 10 pixels to verify
    printf("First 10 pixels (RGB -> Gray):\n");
    printf("%-6s  %-5s %-5s %-5s  ->  %-5s\n", "Pixel", "R", "G", "B", "Gray");
    printf("------  ----- ----- -----  --  -----\n");
    for (int i = 0; i < 10; i++) {
        unsigned char r = h_Pin[i * 3 + 0];
        unsigned char g = h_Pin[i * 3 + 1];
        unsigned char b = h_Pin[i * 3 + 2];
        printf("%-6d  %-5d %-5d %-5d  ->  %-5d\n", i, r, g, b, h_Pout[i]);
    }

    // Cleanup
    free(h_Pin);
    free(h_Pout);
    cudaFree(d_Pin);
    cudaFree(d_Pout);

    printf("\nDone! Grayscale conversion successful.\n");
    return 0;
}
