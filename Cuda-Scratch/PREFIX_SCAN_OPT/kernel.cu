
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
        if (!result) {
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


__global__ void scan(const int* image, int* histogram, const size_t imageSize);


int main()
{
    constexpr int arraySize = 2048;
    int * hData = (int*)malloc(arraySize * sizeof(int));
    int * hgpuData = (int*)malloc(arraySize * sizeof(int));
    if (!initData(hData, arraySize)) { std::cerr << "Failed to init data." << std::endl;  return 0; }

    printData(hData, arraySize);

    int* d_data;

    checkCudaErrors(cudaMalloc((int**)&d_data, arraySize * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_data, hData, arraySize * sizeof(int), cudaMemcpyHostToDevice));


    return 0;
}
