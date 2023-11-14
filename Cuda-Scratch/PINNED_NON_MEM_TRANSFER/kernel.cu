#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <random>
#include <time.h>
#include <chrono>
#include <iostream>

using namespace std;

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

int main(int argc, char** argv)
{
    int dev = 0;
    printf("%s Starting...\n", argv[0]);
    checkCudaErrors(cudaSetDevice(0));

    // memory size
    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev,
        deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));

    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        checkCudaErrors(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    // allocate the host memory
    float* h_a = (float*)malloc(nbytes);

    // allocate pinned host memory
    float* h_a_pin;
    checkCudaErrors(cudaMallocHost((float**)&h_a_pin, nbytes));

    // allocate the device memory
    float* d_a;
    checkCudaErrors(cudaMalloc((float**)&d_a, nbytes));

    // allocate device memory
    float* d_a_pin;
    checkCudaErrors(cudaMalloc((float**)&d_a_pin, nbytes));

    // initialize the host memory
    for (unsigned int i = 0; i < isize; i++) h_a[i] = 0.5f;
    
    // initialize host memory
    memset(h_a_pin, 0, nbytes);
    /*
    for (int i = 0; i < isize; i++) h_a_pin[i] = 100.10f;
    auto start = std::chrono::steady_clock::now();
    // transfer data from the host to the device
    checkCudaErrors(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    // transfer data from the device to the host
    checkCudaErrors(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::steady_clock::now(); auto diff = end - start;
    double elapsed = std::chrono::duration<double>(diff).count();
    printf("memTransfer            elapsed %f sec\n", elapsed);
    */
    auto start = std::chrono::steady_clock::now();
    // transfer data from the host to the device
    checkCudaErrors(cudaMemcpy(d_a_pin, h_a_pin, nbytes, cudaMemcpyHostToDevice));
    // transfer data from the device to the host
    checkCudaErrors(cudaMemcpy(h_a_pin, d_a_pin, nbytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::steady_clock::now(); auto diff = end - start;
    auto elapsed = std::chrono::duration<double>(diff).count();
    printf("pinMemTransfer         elapsed %f sec\n", elapsed);
    
    // free memory
    checkCudaErrors(cudaFree(d_a));
    free(h_a);
    checkCudaErrors(cudaFree(d_a_pin));
    checkCudaErrors(cudaFreeHost(h_a_pin));

    // reset device
    checkCudaErrors(cudaDeviceReset());
    return EXIT_SUCCESS;
}
