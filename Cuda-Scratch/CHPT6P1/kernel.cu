
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublasLt.h>

#include <stdio.h>

#include <numeric>
#include<algorithm>
#include <iostream>

#define ARRAY_SIZE 8
#define TILE_WIDTH 2

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

void printMatrix(int (& matrix)[ARRAY_SIZE * ARRAY_SIZE]) {
    int indx = 0;
    std::for_each(std::begin(matrix), std::end(matrix), [&](int& a) { std::cout << indx++ << ":" << a << std::endl; });
}

void printMatrix(int *& matrix, size_t size ) {
    int indx = 0;
    std::for_each(matrix, matrix + size, [ &](int& a) { std::cout << indx++ << ":" << a << std::endl; });
}

void initMatrix(int(&matrix)[ARRAY_SIZE * ARRAY_SIZE]) {
    std::iota(std::begin(matrix), std::end(matrix), 1);
}

void matMult(const int* A, const int* B, int* C, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            int cValue = 0;
            for (size_t k = 0; k < size; ++k) {
                cValue += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = cValue;
        }
    }
}

__global__ void matMultKernel(int* A, int* B, int* C);

int main()
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((ARRAY_SIZE + block.x - 1) / block.x, (ARRAY_SIZE + block.x - 1) / block.x);
    size_t matrixBytes = ARRAY_SIZE * ARRAY_SIZE * sizeof(int);

    int A[ARRAY_SIZE * ARRAY_SIZE] = {};
    int B[ARRAY_SIZE * ARRAY_SIZE] = {};
    int C_raw[ARRAY_SIZE * ARRAY_SIZE] = {};
    
    //int C[ARRAY_SIZE * ARRAY_SIZE] = {};
    int* C = (int*)malloc(matrixBytes);

    initMatrix(A); initMatrix(B);
    //printMatrix(A);

    matMult(A, B, C_raw, ARRAY_SIZE);
    //printMatrix(C_raw);


    // allocate device memory
    
    int* d_A = NULL;
    int* d_B = NULL;
    int* d_C = NULL;
    checkCudaErrors(cudaMalloc((void**)&d_A, matrixBytes));
    checkCudaErrors(cudaMalloc((void**)&d_B, matrixBytes));
    checkCudaErrors(cudaMalloc((void**)&d_C, matrixBytes));

    checkCudaErrors(cudaMemcpy(d_A, A, matrixBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, matrixBytes, cudaMemcpyHostToDevice));

    matMultKernel << < grid, block >> > (d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(C, d_C, matrixBytes, cudaMemcpyDeviceToHost));

    printMatrix(C, ARRAY_SIZE * ARRAY_SIZE);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCudaErrors(cudaDeviceReset());

    return 0;
}

__global__ void matMultKernel(int* A, int* B, int* C) {

    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int result = 0;

    for (int ph = 0; ph < ARRAY_SIZE / TILE_WIDTH; ++ph) {

        Mds[ty][tx] = A[Row * ARRAY_SIZE + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = B[Col * ARRAY_SIZE + ph * TILE_WIDTH + ty];

        __syncthreads();


        for (int k = 0; k < TILE_WIDTH; ++k) {
            result += Mds[ty][k] + Nds[k][tx];
        }

        __syncthreads();

    }

    C[Row * ARRAY_SIZE + Col] = result;
}
