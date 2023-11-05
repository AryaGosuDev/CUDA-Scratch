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

int recursiveReduce(int* data, int const size) {
    if (size == 1) return data[0];
    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];
    return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(int* g_idata, int* g_odata,
    unsigned int n) {

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
            idata[index] += idata[index + stride];
        
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n) {

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling16(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 16;

    // unrolling 16
    if (idx + 15 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        int c1 = g_idata[idx + 8 * blockDim.x];
        int c2 = g_idata[idx + 9 * blockDim.x];
        int c3 = g_idata[idx + 10 * blockDim.x];
        int c4 = g_idata[idx + 11 * blockDim.x];
        int d1 = g_idata[idx + 12 * blockDim.x];
        int d2 = g_idata[idx + 13 * blockDim.x];
        int d3 = g_idata[idx + 14 * blockDim.x];
        int d4 = g_idata[idx + 15 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 + c1 + c2 + c3 + c4 + d1 + d2 + d3 + d4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int* g_idata, int* g_odata,
    unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int* g_idata, int* g_odata,
    unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling last warp
    if (tid < 32)
    {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    checkCudaErrors(cudaSetDevice(0));

    bool bResult = false;

    int size = 1 << 24; // total number of elements
    printf("    with array size %d  ", size);

    // initial block size
    int blocksize = 512;   

    if (argc > 1)
        blocksize = atoi(argv[1]);

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int* h_idata = (int*)malloc(bytes);
    int* h_odata = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
        h_idata[i] = (int)(rand() & 0xFF);
    
    memcpy(tmp, h_idata, bytes);

    int gpu_sum = 0;

    // allocate device memory
    int* d_idata = NULL;
    int* d_odata = NULL;
    checkCudaErrors(cudaMalloc((void**)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));

    // cpu reduction
    auto start = std::chrono::steady_clock::now();
    int cpu_sum = recursiveReduce(tmp, size);
    auto end = std::chrono::steady_clock::now(); auto diff = end - start;
    double elapsed = std::chrono::duration<double> (diff).count();
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", elapsed, cpu_sum);

    // kernel 1: reduceNeighbored
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceNeighbored << <grid, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceNeighboredLess << <grid, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceInterleaved << <grid, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceUnrolling2 << <grid.x / 2, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x / 2, block.x);

    // kernel 5: reduceUnrolling4
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceUnrolling4 << <grid.x / 4, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x / 4, block.x);

    // kernel 6: reduceUnrolling8
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceUnrolling8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x / 8, block.x);

    //for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];

    // kernel 8: reduceUnrollWarps8
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceUnrollWarps8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x / 8, block.x);


    // kernel 9: reduceCompleteUnrollWarsp8
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceCompleteUnrollWarps8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x / 8, block.x);

    // kernel 9: reduceCompleteUnroll
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();

    switch (blocksize) {
    case 1024:
        reduceCompleteUnroll<1024> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 512:
        reduceCompleteUnroll<512> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 256:
        reduceCompleteUnroll<256> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 128:
        reduceCompleteUnroll<128> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 64:
        reduceCompleteUnroll<64> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    }

    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
        cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double> (diff).count();
    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x / 8, block.x);

    // kernel 10: reduceUnrolling16
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    reduceUnrolling16 << <grid.x / 16, block >> > (d_idata, d_odata, size);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now(); diff = end - start;
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x / 16 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];
    elapsed = std::chrono::duration<double>(diff).count();
    printf("gpu Unrolling16 elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", elapsed, gpu_sum, grid.x / 16, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    // reset device
    checkCudaErrors(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if (!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
