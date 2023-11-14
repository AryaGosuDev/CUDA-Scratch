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

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void initialData(float* ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}


void sumArraysOnHost(float* A, float* B, float* C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float* A, float* B, float* C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffset(float* A, float* B, float* C, const int n,
    int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

int main(int argc, char** argv)
{
    int dev = 0; cudaDeviceProp deviceProp;
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    checkCudaErrors(cudaSetDevice(dev));

    // set up array size
    int nElem = 1 << 20; // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // set up offset for summary
    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset = atoi(argv[1]);

    if (argc > 2) blocksize = atoi(argv[2]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float* h_A = (float*)malloc(nBytes);
    float* h_B = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    auto start = std::chrono::steady_clock::now();
    //  summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);
    auto end = std::chrono::steady_clock::now(); auto diff = end - start;
    auto elapsed = std::chrono::duration<double>(diff).count();
    printf("sumArraysOnHost         elapsed %f sec\n", elapsed);

    // allocate device memory
    float* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((float**)&d_A, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_B, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));
    std::vector<int> offsetMarkArr = { 0,4,6,8,16,32,64,96,128,160,192,224,256 };
    for (int i = 0; i < offsetMarkArr.size(); ++i) {

        start = std::chrono::steady_clock::now();
        readOffset << <grid, block >> > (d_A, d_B, d_C, nElem, offsetMarkArr[i]);
        checkCudaErrors(cudaDeviceSynchronize());
        end = std::chrono::steady_clock::now(); diff = end - start;
        elapsed = std::chrono::duration<double>(diff).count();
        printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
            block.x, offsetMarkArr[i], elapsed);
        checkCudaErrors(cudaGetLastError());

        // copy kernel result back to host side and check device results
        checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
        sumArraysOnHost(h_A, h_B, hostRef, nElem, offsetMarkArr[i]);
        checkResult(hostRef, gpuRef, nElem - offsetMarkArr[i]);

    }
      
    // free host and device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    checkCudaErrors(cudaDeviceReset());
    return EXIT_SUCCESS;
}