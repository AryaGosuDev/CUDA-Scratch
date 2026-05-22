#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <exception>

__host__ cudaError_t runMultMatrix();

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__host__ cudaError_t runAddKernel() {

    cudaError_t cudaStatus;
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    try {
        const unsigned int arraySize = 5;
        const int a[arraySize] = { 1, 2, 3, 4, 5 };
        const int b[arraySize] = { 10, 20, 30, 40, 50 };
        int c[arraySize] = { 0 };

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            throw std::exception();
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            throw std::exception();
        }

        cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            throw std::exception();
        }

        cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            throw std::exception();
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            throw std::exception();
        }

        cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            throw std::exception();
        }

        // Launch a kernel on the GPU with one thread for each element.
        addKernel <<<1, arraySize >>> (dev_c, dev_a, dev_b);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            throw std::exception();
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            throw std::exception();
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            throw std::exception();
        }

        printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
            c[0], c[1], c[2], c[3], c[4]);
    }
    catch (std::exception e) {
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
    }

    return cudaSuccess;

}

int main()
{
    cudaError_t cudaStatus ;
     
    cudaStatus = runMultMatrix();
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
