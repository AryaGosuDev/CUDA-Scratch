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

#define BDIMX 32
#define BDIMY 32
#define IPAD  1

void printData(char* msg, int* in, const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%5d", in[i]);
        fflush(stdout);
    }

    printf("\n");
    return;
}

__global__ void setRowReadRow(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}


__global__ void setRowReadColDyn(int* out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    // shared memory store operation
    tile[row_idx] = row_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // mapping from thread index to global memory offset
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynPad(int* out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[row_idx] = g_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

/*
 nv-nsight-cu-cli --metrics sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld,
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_st,
sm__sass_l1tex_pipe_lsu_wavefronts_mem_shared 
SHARED_MEM_ROW_COL_ACCESS.exe
*/

int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    checkCudaErrors(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    checkCudaErrors(cudaDeviceGetSharedMemConfig(&pConfig));
    printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    // set up array size 2048
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
        block.y);

    // allocate device memory
    int* d_C;
    checkCudaErrors(cudaMalloc((int**)&d_C, nBytes));
    int* gpuRef = (int*)malloc(nBytes);

    checkCudaErrors(cudaMemset(d_C, 0, nBytes));
    auto start = std::chrono::steady_clock::now();
    setColReadCol << <grid, block >> > (d_C);
    auto end = std::chrono::steady_clock::now(); auto diff = end - start; auto elapsed = std::chrono::duration<double>(diff).count();
    printf("setColReadCol <<< %3d, grid <%3d,%3d> >>> elapsed %f sec\n", grid.x, block.x, block.y,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)  printData("set col read col   ", gpuRef, nx * ny);

    checkCudaErrors(cudaMemset(d_C, 0, nBytes));
    start = std::chrono::steady_clock::now();
    setRowReadRow << <grid, block >> > (d_C);
    end = std::chrono::steady_clock::now(); diff = end - start; elapsed = std::chrono::duration<double>(diff).count();
    printf("setRowReadRow <<< %3d, grid <%3d,%3d> >>> elapsed %f sec\n", grid.x, block.x, block.y,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)  printData("set row read row   ", gpuRef, nx * ny);

    checkCudaErrors(cudaMemset(d_C, 0, nBytes));
    start = std::chrono::steady_clock::now();
    setRowReadCol << <grid, block >> > (d_C);
    end = std::chrono::steady_clock::now(); diff = end - start; elapsed = std::chrono::duration<double>(diff).count();
    printf("setRowReadCol <<< %3d, grid <%3d,%3d> >>> elapsed %f sec\n", grid.x, block.x, block.y,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)  printData("set row read col   ", gpuRef, nx * ny);

    checkCudaErrors(cudaMemset(d_C, 0, nBytes));
    start = std::chrono::steady_clock::now();
    setRowReadColDyn << <grid, block, BDIMX* BDIMY * sizeof(int) >> > (d_C);
    end = std::chrono::steady_clock::now(); diff = end - start; elapsed = std::chrono::duration<double>(diff).count();
    printf("setRowReadColDyn <<< %3d, grid <%3d,%3d> >>> elapsed %f sec\n", grid.x, block.x, block.y,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)  printData("set row read col dyn", gpuRef, nx * ny);

    checkCudaErrors(cudaMemset(d_C, 0, nBytes));
    start = std::chrono::steady_clock::now();
    setRowReadColPad << <grid, block >> > (d_C);
    end = std::chrono::steady_clock::now(); diff = end - start; elapsed = std::chrono::duration<double>(diff).count();
    printf("setRowReadColPad <<< %3d, grid <%3d,%3d> >>> elapsed %f sec\n", grid.x, block.x, block.y,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)  printData("set row read col pad", gpuRef, nx * ny);

    checkCudaErrors(cudaMemset(d_C, 0, nBytes));
    start = std::chrono::steady_clock::now();
    setRowReadColDynPad << <grid, block, (BDIMX + IPAD)* BDIMY * sizeof(int) >> > (d_C);
    end = std::chrono::steady_clock::now(); diff = end - start; elapsed = std::chrono::duration<double>(diff).count();
    printf("setRowReadColDynPad <<< %3d, grid <%3d,%3d> >>> elapsed %f sec\n", grid.x, block.x, block.y,
        elapsed);
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)  printData("set row read col DP ", gpuRef, nx * ny);

    // free host and device memory
    checkCudaErrors(cudaFree(d_C));
    free(gpuRef);

    // reset device
    checkCudaErrors(cudaDeviceReset());
    return EXIT_SUCCESS;
}