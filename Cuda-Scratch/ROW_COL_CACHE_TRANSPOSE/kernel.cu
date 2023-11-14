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

#define BDIMX 16
#define BDIMY 16

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f; //100.0f;
    }

    return;
}

void printData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%dth element: %f\n", i, in[i]);
    }

    return;
}

void checkCudaErrorsResult(float* hostRef, float* gpuRef, const int size, int showme)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                gpuRef[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float* out, float* in, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; ++iy)
    {
        for (int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

__global__ void warmup(float* out, float* in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// case 0 copy kernel: access data in rows
__global__ void copyRow(float* out, float* in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}
 
// case 1 copy kernel: access data in columns
__global__ void copyCol(float* out, float* in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}
     
// case 2 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float* out, float* in, const int nx,
    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// case 3 transpose kernel: read in columns and write in rows
__global__ void transposeNaiveCol(float* out, float* in, const int nx,
    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__ void transposeUnroll4Row(float* out, float* in, const int nx,
    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float* out, float* in, const int nx,
    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[ti] = in[to];
        out[ti + blockDim.x] = in[to + blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}

/*
 * case 6 :  transpose kernel: read in rows and write in colunms + diagonal
 * coordinate transform
 */
__global__ void transposeDiagonalRow(float* out, float* in, const int nx,
    const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

/*
 * case 7 :  transpose kernel: read in columns and write in row + diagonal
 * coordinate transform.
 */
__global__ void transposeDiagonalCol(float* out, float* in, const int nx,
    const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

 

//profile with both L1 cache turned off and off : to turn off l1 load cache, compile with flag  --ptxas-options=-dlcm=cg
int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    checkCudaErrors(cudaSetDevice(0));

    // set up array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) iKernel = atoi(argv[1]);

    if (argc > 2) blockx = atoi(argv[2]);

    if (argc > 3) blocky = atoi(argv[3]);

    if (argc > 4) nx = atoi(argv[4]);

    if (argc > 5) ny = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // allocate host memory
    float* h_A = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx * ny);

    // transpose at host side
    transposeHost(hostRef, h_A, nx, ny);

    // allocate device memory
    float* d_A, * d_C;
    checkCudaErrors(cudaMalloc((float**)&d_A, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // warmup to avoide startup overhead
    auto start = std::chrono::steady_clock::now();
    warmup << <grid, block >> > (d_C, d_A, nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now(); auto diff = end - start;
    double elapsed = std::chrono::duration<double>(diff).count();
    printf("warmup         elapsed %f sec\n", elapsed);
    checkCudaErrors(cudaGetLastError());

    vector < pair<void (*)(float*, float*, int, int), string> > kernelArray;
     
    kernelArray.push_back({ &copyRow, "CopyRow       " });
    kernelArray.push_back({ &copyCol, "CopyCol       " });
    kernelArray.push_back({ &transposeNaiveRow, "NaiveRow       " });
    kernelArray.push_back({ &transposeNaiveCol, "NaiveCol       " });
    kernelArray.push_back({ &transposeUnroll4Row, "Unroll4Row       " });
    kernelArray.push_back({ &transposeUnroll4Col, "Unroll4Col       " });
    kernelArray.push_back({ &transposeDiagonalRow, "DiagonalRow       " });
    kernelArray.push_back({ &transposeDiagonalCol, "DiagonalCol       " });

    for (int iKernel = 0; iKernel < kernelArray.size(); ++iKernel) {
        // execution configuration
        dim3 block(blockx, blocky);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

        if (iKernel == 4 || iKernel == 5) grid.x = (nx + block.x * 4 - 1) / (block.x * 4);

        start = std::chrono::steady_clock::now();
        kernelArray[iKernel].first << <grid, block >> > (d_C, d_A, nx, ny);
        checkCudaErrors(cudaDeviceSynchronize());
        end = std::chrono::steady_clock::now(); diff = end - start;
        elapsed = std::chrono::duration<double>(diff).count();

        // calculate effective_bandwidth
        float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / elapsed;
        printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> effective "
            "bandwidth %f GB\n", (kernelArray[iKernel].second).c_str(), elapsed, grid.x, grid.y, block.x,
            block.y, ibnd);
        checkCudaErrors(cudaGetLastError());

        // checkCudaErrors kernel results
        if (iKernel > 1)
        {
            checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
            checkCudaErrorsResult(hostRef, gpuRef, nx * ny, 1);
        }
    }


    // free host and device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    checkCudaErrors(cudaDeviceReset());
    return EXIT_SUCCESS;
}