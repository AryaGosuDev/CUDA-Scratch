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

void initialData(float* ip, const int size) {
    for ( int i = 0; i < size; i++) ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny) {
    float* ia = A; float* ib = B; float* ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)  {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n"); else printf("Arrays do not match.\n\n");
}

__global__ void sumMatrixOnGPU2D(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    checkCudaErrors(cudaSetDevice(0));

    // set up data size of matrix
    // elem size = 16,384
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    
    auto start = std::chrono::steady_clock::now();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (diff).count();

    float* d_MatA, * d_MatB, * d_MatC;
    checkCudaErrors(cudaMalloc((void**)&d_MatA, nBytes));
    checkCudaErrors(cudaMalloc((void**)&d_MatB, nBytes));
    checkCudaErrors(cudaMalloc((void**)&d_MatC, nBytes));

    checkCudaErrors(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    sumMatrixOnGPU2D << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nxy);

    checkCudaErrors(cudaFree(d_MatA));
    checkCudaErrors(cudaFree(d_MatB));
    checkCudaErrors(cudaFree(d_MatC));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    checkCudaErrors(cudaDeviceReset());

    return (0);
}
