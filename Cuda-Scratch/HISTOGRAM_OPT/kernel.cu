
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"

#include <stdio.h>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <deque>
#include <list>
#include <stack>
#include <queue>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <algorithm>
#include <iterator>

#include <numeric>
#include <cmath>
#include <bit>

#include <utility>
#include <tuple>
#include <optional>
#include <variant>

#include <memory>

#include <limits>
#include <cstdint>

#include <functional>
#include <random>
#include <chrono>

#include <exception>
#include <stdexcept>

#include <thread>
#include <mutex>
#include <atomic>

#include <ranges>
#include "kernel.h"

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \


static bool initImage( int* image, const size_t imageSize) {
    if (image != NULL) {
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, 255);
        std::for_each(image, image + imageSize, [&](int& a) { a = distribution(generator); });
        return true;
    }
    return false;
}


static bool createGroundTruthHistogram( const int* image, int * histogram, const size_t imageSize) {
    if (image != NULL && histogram != NULL) {
        for (size_t i = 0; i < imageSize; ++i) {
            histogram[image[i]]++;
        }
        return true;
    }
    return false;
}


static void printHistogram(int* histogram, const size_t histogramSize) {
    if (histogram != NULL) {
        int histIndx = 0;
        std::for_each(histogram, histogram + histogramSize, [&](int& a) { std::cout << (histIndx++) << " : " << a << std::endl; });
    }
}

static bool verifyHistogram(const int* groundTruthHistogram, const int* histogram, const size_t histogramSize) {
    if (groundTruthHistogram != NULL && histogram != NULL) {
        size_t histIndx = 0;
        bool result = std::any_of(groundTruthHistogram, groundTruthHistogram + histogramSize, [&](const int & a) { return a != histogram[histIndx++]; });
        if (!result) {
            std::cout << "Histograms match !" << std::endl;
            return true;
        }
        else {
            std::cout << "Error : Histograms do not match !" << std::endl;
            return false;
        }

    }
    std::cerr << "Error in verifyHistogram, null pointers." << std::endl;
    return false; 

}

__global__ void calculateHistogram(const int* image, int* histogram, const size_t imageSize);

__global__ void calculateHistogramPrivate(const int* image, int* privateHistogram, const size_t imageSize, const size_t binSize);


int main()
{
    constexpr size_t imageSize = 1920 * 1080;  
    constexpr size_t histogramSize = 256;

    int* image = (int*)malloc(imageSize * sizeof(int));
    if (!initImage(image, imageSize)) { std::cerr << "Failed to init image." << std::endl;  return 0; }

    int groundTruthHistogram[histogramSize] = {};
    int histogram[histogramSize] = {};

    std::fill(groundTruthHistogram, groundTruthHistogram + histogramSize, 0);
    std::fill(histogram, histogram + histogramSize, 0);

    if ( !createGroundTruthHistogram(image, groundTruthHistogram, imageSize) ) { std::cerr << "Failed to init ground truth histogram." << std::endl;  return 0; }

    int* d_image;
    int* d_histogram;
    
    checkCudaErrors(cudaMalloc((int**)&d_image, imageSize * sizeof(int)));
    checkCudaErrors(cudaMalloc((int**)&d_histogram, histogramSize * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_image, image, imageSize * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_histogram, histogram, histogramSize * sizeof(int), cudaMemcpyHostToDevice));

    /*************************** CUDA HISTOGRAM ***************************/

    dim3 block(1024);
    dim3 grid((imageSize + block.x - 1) / block.x);
    std::cout << "grid : " << grid.x << " " << "block : " << block.x << std::endl;
    calculateHistogram << <grid, block >> > (d_image, d_histogram, imageSize);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(histogram, d_histogram, histogramSize * sizeof(int), cudaMemcpyDeviceToHost));

    if (!verifyHistogram(groundTruthHistogram, histogram, histogramSize) ) return 0 ;


    /*************************** PRIVATE HISTOGRAM ***************************/
    int* privateHistogram = (int*)malloc(histogramSize * grid.x * sizeof(int));
    std::fill(privateHistogram, privateHistogram + ( histogramSize * grid.x ), 0);
    int* d_privateHistogram;
    checkCudaErrors(cudaMalloc((int**)&d_privateHistogram, histogramSize * grid.x * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_privateHistogram, privateHistogram, histogramSize * grid.x * sizeof(int), cudaMemcpyHostToDevice));
    calculateHistogramPrivate << <grid, block >> > (d_image, d_privateHistogram, imageSize, histogramSize);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(privateHistogram, d_privateHistogram, histogramSize * grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    //printHistogram(privateHistogram, histogramSize);

    if (!verifyHistogram(groundTruthHistogram, privateHistogram, histogramSize)) return 0;

    

    return 0;
}

__global__ void calculateHistogram(const int* image, int* histogram, const size_t imageSize) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < imageSize) {
        atomicAdd(&histogram[image[tid]], 1);
    }
}

__global__ void calculateHistogramPrivate(const int* image, int* privateHistogram, const size_t imageSize, const size_t binSize) {
    /*
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < imageSize) {
        atomicAdd(&privateHistogram[image[tid]], 1);
    }
    */
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < imageSize) {
        
        atomicAdd(&privateHistogram[blockIdx.x * binSize + image[tid]], 1);
    }
    __syncthreads();

    /*
    if (blockIdx.x > 0 && tid < gridDim.x * binSize  ) {
        int histVal = privateHistogram[tid];
        //printf("%i \n", tid);
        atomicAdd(&privateHistogram[ (tid % binSize) ], histVal);
        
    }
    */

    if (blockIdx.x > 0) {
        for (size_t bin = threadIdx.x; bin < binSize; bin += blockDim.x) {
            int binValue = privateHistogram[blockIdx.x * binSize + bin];
            if (binValue > 0) {
                atomicAdd(&(privateHistogram[bin]), binValue);
            }
        }
    }
    __syncthreads();
    
}