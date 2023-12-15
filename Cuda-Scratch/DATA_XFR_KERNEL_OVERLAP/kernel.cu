#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <Windows.h>
#include <cstdlib> 

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

#define NSTREAM 4
#define BDIM 128

void initialData(float* ip, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float* A, float* B, float* C, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        for (int i = 0; i < N; ++i)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}


int main(int argc, char** argv)
{
    printf("> %s Starting...\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    checkCudaErrors(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
        deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);



    // set up max connection
    SetEnvironmentVariable("CUDA_DEVICE_MAX_CONNECTIONS", "1");
    char buffer[100];
    DWORD bufferSize = sizeof(buffer) / sizeof(char);
    GetEnvironmentVariable("CUDA_DEVICE_MAX_CONNECTIONS", buffer, bufferSize);
    printf("%s = %s\n", "CUDA_DEVICE_MAX_CONNECTIONS", buffer);
    printf("> with streams = %d\n", NSTREAM);

    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float* h_A, * h_B, * hostRef, * gpuRef;
    checkCudaErrors(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((float**)&d_A, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_B, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, nBytes));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // invoke kernel at host side
    dim3 block(BDIM);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x,
        block.y);

    // sequential operation
    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float memcpy_h2d_time;
    checkCudaErrors(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    sumArrays << <grid, block >> > (d_A, d_B, d_C, nElem);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float kernel_time;
    checkCudaErrors(cudaEventElapsedTime(&kernel_time, start, stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float memcpy_d2h_time;
    checkCudaErrors(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

    printf("\n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n",
        memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n",
        memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n",
        kernel_time, (nBytes * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n",
        itotal, (nBytes * 2e-6) / itotal);

    // grid parallel operation
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    checkCudaErrors(cudaEventRecord(start, 0));

    // initiate all work on the device asynchronously in depth-first order
    for (int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem;
        checkCudaErrors(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes,
            cudaMemcpyHostToDevice, stream[i]));
        checkCudaErrors(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes,
            cudaMemcpyHostToDevice, stream[i]));
        sumArrays << <grid, block, 0, stream[i] >> > (&d_A[ioffset], &d_B[ioffset],
            &d_C[ioffset], iElem);
        checkCudaErrors(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes,
            cudaMemcpyDeviceToHost, stream[i]));
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float execution_time;
    checkCudaErrors(cudaEventElapsedTime(&execution_time, start, stop));

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
        execution_time, (nBytes * 2e-6) / execution_time);
    printf(" speedup                : %f \n",
        ((itotal - execution_time) * 100.0f) / itotal);

    // check kernel error
    checkCudaErrors(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // free host memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(hostRef));
    checkCudaErrors(cudaFreeHost(gpuRef));

    // destroy events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // destroy streams
    for (int i = 0; i < NSTREAM; ++i)
    {
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }

    checkCudaErrors(cudaDeviceReset());
    return(0);
}