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

#define LEN 1<<22

struct alignas(32) AoS{
    float x;
    float y;
};

struct SoA{
    float x[LEN];
    float y[LEN];
};
 
void initialAoSStruct(AoS* ip, int size) {
    for (int i = 0; i < size; i++){
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }
}

void testAoSStructHost(AoS* A, AoS* C, const int n) {
    for (int idx = 0; idx < n; idx++){
        C[idx].x = A[idx].x + 10.f;
        C[idx].y = A[idx].y + 20.f;
    }
}

// functions for inner array outer struct
void initialSoAArray(SoA* ip, int size) {
    for (int i = 0; i < size; i++) {
        ip->x[i] = (float)(rand() & 0xFF) / 100.0f;
        ip->y[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void testSoAArrayHost(SoA* A, SoA* C, const int n)
{
    for (int idx = 0; idx < n; idx++)
    {
        C->x[idx] = A->x[idx] + 10.f;
        C->y[idx] = A->y[idx] + 20.f;
    }

    return;
}

void checkAoSStruct(AoS* hostRef, AoS* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                hostRef[i].x, gpuRef[i].x);
            break;
        }

        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                hostRef[i].y, gpuRef[i].y);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void printfHostResult(SoA* C, const int n) {
    for (int idx = 0; idx < n; idx++) {
        printf("printout idx %d:  x %f y %f\n", idx, C->x[idx], C->y[idx]);
    }

    return;
}

void checkSoAArray(SoA* hostRef, SoA* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon)
        {
            match = 0;
            printf("different on x %dth element: host %f gpu %f\n", i,
                hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon)
        {
            match = 0;
            printf("different on y %dth element: host %f gpu %f\n", i,
                hostRef->y[i], gpuRef->y[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void testStructAoS(AoS* data, AoS* result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        AoS tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

__global__ void testStructSoA(SoA* data, SoA* result,
    const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    checkCudaErrors(cudaSetDevice(0));

    // allocate host memory
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(AoS);

    AoS* h_A = (AoS*)malloc(nBytes);
    AoS* hostRef = (AoS*)malloc(nBytes);
    AoS* gpuRef = (AoS*)malloc(nBytes);

    SoA* h_A_SoA = (SoA*)malloc(nBytes);
    SoA* hostRef_SoA = (SoA*)malloc(nBytes);
    SoA* gpuRef_SoA = (SoA*)malloc(nBytes);

    // initialize host array
    initialAoSStruct(h_A, nElem);
    testAoSStructHost(h_A, hostRef, nElem);

    initialSoAArray(h_A_SoA, nElem);
    testSoAArrayHost(h_A_SoA, hostRef_SoA, nElem);
    
    // allocate device memory
    AoS* d_A, * d_C;
    checkCudaErrors(cudaMalloc((AoS**)&d_A, nBytes));
    checkCudaErrors(cudaMalloc((AoS**)&d_C, nBytes));

    SoA* d_A_SoA, * d_C_SoA;
    checkCudaErrors(cudaMalloc((SoA**)&d_A_SoA, nBytes));
    checkCudaErrors(cudaMalloc((SoA**)&d_C_SoA, nBytes));

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_SoA, h_A_SoA, nBytes, cudaMemcpyHostToDevice));

    int blocksize = 128;

    if (argc > 1) blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // kernel 2: testInnerStruct
    auto start = std::chrono::steady_clock::now();
    testStructAoS << <grid, block >> > (d_A, d_C, nElem);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now(); auto diff = end - start; auto elapsed = std::chrono::duration<double>(diff).count();
    printf("testStructAoS <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkAoSStruct(hostRef, gpuRef, nElem);
    checkCudaErrors(cudaGetLastError());

    //kernel 2 SoA
    start = std::chrono::steady_clock::now();
    testStructSoA << <grid, block >> > (d_A_SoA, d_C_SoA, nElem);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start; elapsed = std::chrono::duration<double>(diff).count();
    printf("testStructSoA   <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef_SoA, d_C_SoA, nBytes, cudaMemcpyDeviceToHost));
    checkSoAArray(hostRef_SoA, gpuRef_SoA, nElem);
    checkCudaErrors(cudaGetLastError());

    // free memories both host and device
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A_SoA));
    checkCudaErrors(cudaFree(d_C_SoA));
    free(h_A);
    free(hostRef);
    free(gpuRef);
    free(h_A_SoA);
    free(hostRef_SoA);
    free(gpuRef_SoA);

    // reset device
    checkCudaErrors(cudaDeviceReset());
    return EXIT_SUCCESS;
}