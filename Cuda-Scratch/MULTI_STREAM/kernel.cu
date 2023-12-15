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

#define N 300000
#define NSTREAM 4

__global__ void kernel_1()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char** argv)
{
    int n_streams = NSTREAM;
    int isize = 1;
    int iblock = 1;
    int bigcase = 0;

    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connection
    SetEnvironmentVariable("CUDA_DEVICE_MAX_CONNECTIONS", "32");
    char buffer[100];
    DWORD bufferSize = sizeof(buffer) / sizeof(char);
    GetEnvironmentVariable("CUDA_DEVICE_MAX_CONNECTIONS", buffer, bufferSize);
    printf("%s = %s\n", "CUDA_DEVICE_MAX_CONNECTIONS", buffer);

    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name,
        n_streams);
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

    // Allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++)
    {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration
    dim3 block(iblock);
    dim3 grid(isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    /*
    // record start event
    checkCudaErrors(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        kernel_1 << <grid, block, 0, streams[i] >> > ();
        kernel_2 << <grid, block, 0, streams[i] >> > ();
        kernel_3 << <grid, block, 0, streams[i] >> > ();
        kernel_4 << <grid, block, 0, streams[i] >> > ();
    }

    // record stop event
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    // calculate elapsed time
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n",
        elapsed_time / 1000.0f);
        */
    // record start event
    checkCudaErrors(cudaEventRecord(start, 0));

    // dispatch job with breadth first ordering
    for (int i = 0; i < n_streams; i++)
        kernel_1 << <grid, block, 0, streams[i] >> > ();

    for (int i = 0; i < n_streams; i++)
        kernel_2 << <grid, block, 0, streams[i] >> > ();

    for (int i = 0; i < n_streams; i++)
        kernel_3 << <grid, block, 0, streams[i] >> > ();

    for (int i = 0; i < n_streams; i++)
        kernel_4 << <grid, block, 0, streams[i] >> > ();

    // record stop event
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    // calculate elapsed time
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n",
        elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++)
    {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    // destroy events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // reset device
    checkCudaErrors(cudaDeviceReset());

    return 0;
}