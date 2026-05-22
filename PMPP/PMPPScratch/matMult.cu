#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <exception>

__global__ void multMatrixKernel (int* c, const int* a, const int* b, const unsigned int  width)  {

    int row = threadIdx.x / width;
    int col = threadIdx.x % width;

    //printf("%i : %i \n", threadIdx.x , col);
    int sumValue = 0;
    for (int k = 0; k < width; ++k) {
        sumValue += a[row * width + k] * b[k * width + col];
    }
    c[row * width + col] = sumValue;
}

__host__ cudaError_t runMultMatrix() {

    const unsigned int width = 4;
    const unsigned int matrixSize = 16;
    const int a[matrixSize] = {  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16 };
    const int b[matrixSize] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160 };
    int c[matrixSize] = { 0 };

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaError_t cudaStatus;

    try {
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {

            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            throw std::exception();
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, matrixSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            throw std::exception();
        }

        cudaStatus = cudaMalloc((void**)&dev_a, matrixSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            throw std::exception();
        }

        cudaStatus = cudaMalloc((void**)&dev_b, matrixSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            throw std::exception();
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            throw std::exception();
        }

        cudaStatus = cudaMemcpy(dev_b, b, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            throw std::exception();
        }

        // Launch a kernel on the GPU with one thread for each element.
        multMatrixKernel <<<1, matrixSize >>> (dev_c, dev_a, dev_b, width);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            throw std::exception();
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            throw std::exception();
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            throw std::exception();
        }

        for (int i = 0; i < matrixSize; ++i) 
            printf("value at pos %i : %i \n", i, c[i]);
  
    }
    catch (std::exception  e) {
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

    }

    return cudaSuccess;

}