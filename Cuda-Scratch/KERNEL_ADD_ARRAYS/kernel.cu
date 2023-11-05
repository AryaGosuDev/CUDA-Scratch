#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <random>
#include <time.h>
#include <chrono>
#include <iostream>

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}

void initialData(float* ip, int size) {
    time_t t; srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumArraysOnHost(float* A, float* B, float* C, const int N) {
    for (int idx = 0; idx < N; idx++) C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i < N) C[i] = A[i] + B[i];
    C[i] = A[i] + B[i];
}
__global__ void sumArraysOnGPU2(float* A, float* B, float* C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i < N) {
        C[i] = A[i] + B[i]; C[i+N] = A[i+N] + B[i+N];
    //}
    //C[i] = A[i] + B[i];
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    checkCudaErrors(cudaSetDevice(0));

    // nElem = 16,777,216
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    auto start = std::chrono::steady_clock::now();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (diff).count();
    printf("sumArraysOnHost Time elapsed %f millisecs\n", elapsed);

    float* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((float**)&d_A, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_B, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, nBytes));

    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    int iLen = 256;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);
    std::cout << "grid : " << grid.x << " " << "block : " << block.x << std::endl;
    sumArraysOnGPU <<<grid, block >>> (d_A, d_B, d_C, nElem);
    checkCudaErrors(cudaDeviceSynchronize());

    // check kernel error
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nElem);

    dim3 grid2(((nElem/2) + block.x - 1) / block.x);
    std::cout << "grid : " << grid2.x << " " << "block : " << block.x << std::endl;
    memset(gpuRef, 0, nBytes);
    checkCudaErrors(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
    sumArraysOnGPU2 << <grid2, block >> > (d_A, d_B, d_C, nElem/2);
    checkCudaErrors(cudaDeviceSynchronize());

    // check kernel error
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}