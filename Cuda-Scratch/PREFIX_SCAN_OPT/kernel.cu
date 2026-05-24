
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

static constexpr unsigned int arraySize = 2048;
static constexpr unsigned int COURSE_FACTOR = 4; 

static bool initData(int* data, const size_t dataSize) {
    if (data != NULL) {
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, 10);
        std::for_each(data, data + dataSize, [&](int& a) { a = distribution(generator); });
        return true;
    }
    return false;
}

static void prefixSum(int* data, const size_t dataSize) {
    for (size_t i = 1; i < dataSize; ++i) data[i] += data[i - 1];
}

static void printData(int* data, const size_t dataSize) {
    if (data != NULL) {
        std::for_each(data, data + dataSize, [&](int& a) { std::cout << a << std::endl; });
    }
}

static bool verifyData(const int* groundTruthData, const int* data, const size_t dataSize) {
    if (groundTruthData != NULL && data != NULL) {
        bool result = true;
        for (size_t i = 0; i < dataSize; ++i) {
            if (groundTruthData[i] != data[i]) result = false;
        }
        if (result) {
            std::cout << "Data match !" << std::endl;
            return true;
        }
        else {
            std::cout << "Error : Data does not match !" << std::endl;
            return false;
        }
    }
    std::cerr << "Error in verifyData, null pointers." << std::endl;
    return false;
}

__global__ void scan(const int* image, int* histogram);


int main()
{
    // set up device
    int dev = 0;
    checkCudaErrors(cudaSetDevice(dev));

    int * hData = (int*)malloc(arraySize * sizeof(int));
    int * hgpuData = (int*)malloc(arraySize * sizeof(int));
    if (!initData(hData, arraySize)) { std::cerr << "Failed to init data." << std::endl;  return 0; }

    //printData(hData, arraySize);

    int* dData;
    int* dOutput;

    checkCudaErrors(cudaMalloc((int**)&dData, arraySize * sizeof(int)));
    checkCudaErrors(cudaMalloc((int**)&dOutput, arraySize * sizeof(int)));
    checkCudaErrors(cudaMemcpy(dData, hData, arraySize * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(arraySize/COURSE_FACTOR);
    dim3 grid(1);
    std::cout << "grid : " << grid.x << " " << "block : " << block.x << std::endl;
    scan << <grid, block >> > (dData, dOutput);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(hgpuData, dOutput, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

    prefixSum(hData, arraySize);
    //printData(hData, arraySize);
    //printf("%d,%d,%d,%d \n", hData[4], hData[5], hData[6], hData[7]);
    //printData(hgpuData, arraySize);
    verifyData(hData, hgpuData, arraySize);

    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFree(dOutput));
    free(hData);
    free(hgpuData);

    return 0;
}

__global__ void scan(const int* data, int* output) {
    __shared__ int input_s[arraySize];
    unsigned int t = threadIdx.x;
    
    // Load shared memory
    for (unsigned int offsetFactor = 0; offsetFactor < COURSE_FACTOR; ++offsetFactor) {
        input_s[t + offsetFactor * (arraySize / COURSE_FACTOR)] = data[t + offsetFactor * (arraySize / COURSE_FACTOR)];
    }
    __syncthreads();
    
    // Prefix sum for every worker thread segment
    
    for (unsigned int prefixOffset = 1; prefixOffset < COURSE_FACTOR; ++prefixOffset) {
        std::size_t courseFactorOffset = t * COURSE_FACTOR;
        input_s[courseFactorOffset + prefixOffset] += input_s[courseFactorOffset + (prefixOffset - 1)];
    }

    __syncthreads();
    
    // Kogge-stone scan
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        std::size_t courseFactorOffset = t * COURSE_FACTOR + (COURSE_FACTOR - 1);
        if (t >= stride)
            temp = input_s[courseFactorOffset] + input_s[courseFactorOffset - stride * COURSE_FACTOR];
        __syncthreads();
        if (t >= stride)
            input_s[courseFactorOffset] = temp;
    }
    
    // Third phase to add last segment prefix sum scan to next segment
    
    for (unsigned int prefixOffset = 1; prefixOffset < COURSE_FACTOR; ++prefixOffset) {
        std::size_t courseFactorOffset = t * COURSE_FACTOR + (COURSE_FACTOR - 1);
        if (t < (arraySize / COURSE_FACTOR) - 1) {
            input_s[courseFactorOffset + prefixOffset] += input_s[courseFactorOffset];
        }
    }
    __syncthreads();
    

    // Write to output
    for (unsigned int offsetFactor = 0; offsetFactor < COURSE_FACTOR; ++offsetFactor) {
        output[t + offsetFactor * (arraySize / COURSE_FACTOR)] = input_s[t + offsetFactor * (arraySize / COURSE_FACTOR)];
    }

    __syncthreads();

}
