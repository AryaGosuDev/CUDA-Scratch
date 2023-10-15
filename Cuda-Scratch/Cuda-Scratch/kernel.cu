﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    checkCudaErrors(addWithCuda(c, a, b, arraySize));
    
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCudaErrors(cudaDeviceReset());
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCudaErrors(cudaMalloc((void**)&dev_c, size * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_b, size * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCudaErrors(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaSuccess;
}
