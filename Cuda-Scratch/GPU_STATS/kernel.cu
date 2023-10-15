
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>


#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \


void returnGPUCudaInfoResources() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return;
    }

    std::cout << "Number of CUDA devices found: " << deviceCount << std::endl;
    const int nameWidth = 50;  // Width for property name
    const int valueWidth = 40; // Width for property value

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "------------------------------\n";
        std::cout << "Device " << device << ": " << deviceProp.name << "\n";
        std::cout << "------------------------------\n";

        std::cout << std::left
            << std::setw(nameWidth) << "Name:" << std::setw(valueWidth) << deviceProp.name << "\n"
            << std::setw(nameWidth) << "Compute Capability:" << std::setw(valueWidth) << (std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor)) << "\n"
            << std::setw(nameWidth) << "Total Global Memory (bytes):" << std::setw(valueWidth) << deviceProp.totalGlobalMem << "\n"
            << std::setw(nameWidth) << "Shared Memory Per Block (bytes):" << std::setw(valueWidth) << deviceProp.sharedMemPerBlock << "\n"
            << std::setw(nameWidth) << "Registers Per Block:" << std::setw(valueWidth) << deviceProp.regsPerBlock << "\n"
            << std::setw(nameWidth) << "Warp Size:" << std::setw(valueWidth) << deviceProp.warpSize << "\n"
            << std::setw(nameWidth) << "Max Threads Per Block:" << std::setw(valueWidth) << deviceProp.maxThreadsPerBlock << "\n"
            << std::setw(nameWidth) << "Total Const Memory (bytes):" << std::setw(valueWidth) << deviceProp.totalConstMem << "\n"
            << std::setw(nameWidth) << "Texture Alignment (bytes):" << std::setw(valueWidth) << deviceProp.textureAlignment << "\n"
            << std::setw(nameWidth) << "Max Threads Dimensions (x,y,z):" << std::setw(valueWidth) << std::to_string(deviceProp.maxThreadsDim[0]) + ", " + std::to_string(deviceProp.maxThreadsDim[1]) + ", " + std::to_string(deviceProp.maxThreadsDim[2]) << "\n"
            << std::setw(nameWidth) << "Max Grid Size (x,y,z):" << std::setw(valueWidth) << std::to_string(deviceProp.maxGridSize[0]) + ", " + std::to_string( deviceProp.maxGridSize[1]) + ", " + std::to_string( deviceProp.maxGridSize[2] )<< "\n"
            << std::setw(nameWidth) << "Clock Rate (kHz):" << std::setw(valueWidth) << deviceProp.clockRate << "\n"
            << std::setw(nameWidth) << "Memory Clock Rate (kHz):" << std::setw(valueWidth) << deviceProp.memoryClockRate << "\n"
            << std::setw(nameWidth) << "Memory Bus Width (bits):" << std::setw(valueWidth) << deviceProp.memoryBusWidth << "\n"
            << std::setw(nameWidth) << "L2 Cache Size (bytes):" << std::setw(valueWidth) << deviceProp.l2CacheSize << "\n"
            << std::setw(nameWidth) << "Number of Multiprocessors:" << std::setw(valueWidth) << deviceProp.multiProcessorCount << "\n"
            << std::setw(nameWidth) << "Warps per Multiprocessor:" << std::setw(valueWidth) << deviceProp.maxThreadsPerMultiProcessor / 32 << "\n"
            << std::setw(nameWidth) << "Device Overlap:" << std::setw(valueWidth) << (deviceProp.deviceOverlap ? "Supported" : "Not Supported") << "\n"
            << std::setw(nameWidth) << "Concurrent Kernels:" << std::setw(valueWidth) << (deviceProp.concurrentKernels ? "Supported" : "Not Supported") << "\n"
            << std::setw(nameWidth) << "ECC Enabled:" << std::setw(valueWidth) << (deviceProp.ECCEnabled ? "Yes" : "No") << "\n"
            << std::setw(nameWidth) << "TCC Driver:" << std::setw(valueWidth) << (deviceProp.tccDriver ? "Yes" : "No") << "\n"
            << std::setw(nameWidth) << "Can Map Host Memory:" << std::setw(valueWidth) << (deviceProp.canMapHostMemory ? "Supported" : "Not Supported") << "\n"
            << std::setw(nameWidth) << "Compute Mode:" << std::setw(valueWidth) << (deviceProp.computeMode == cudaComputeModeExclusive ? "Exclusive" : deviceProp.computeMode == cudaComputeModeProhibited ? "Prohibited" : "Default") << "\n"
            << std::setw(nameWidth) << "Maximum Texture1D Width:" << std::setw(valueWidth) << deviceProp.maxTexture1D << "\n"
            << std::setw(nameWidth) << "Maximum Texture2D Dimensions (width x height):" << std::setw(valueWidth) << std::to_string(deviceProp.maxTexture2D[0])+ " x "+ std::to_string( deviceProp.maxTexture2D[1]) << "\n"
            << std::setw(nameWidth) << "Maximum Texture3D Dimensions (width x height x depth):" << std::setw(valueWidth) << std::to_string(deviceProp.maxTexture3D[0]) +" x "+ std::to_string( deviceProp.maxTexture3D[1]) + " x "+ std::to_string( deviceProp.maxTexture3D[2] )<< "\n"
            //<< std::setw(nameWidth) << "Maximum Texture2D Array Width:" << std::setw(valueWidth) << deviceProp.maxTexture2DArray[0] << "\n"
            //<< std::setw(nameWidth) << "Maximum Texture2D Array Height:" << std::setw(valueWidth) << deviceProp.maxTexture2DArray[1] << "\n"
            //<< std::setw(nameWidth) << "Maximum Texture2D Array Depth:" << std::setw(valueWidth) << deviceProp.maxTexture2DArray[2] << "\n"
            << std::setw(nameWidth) << "Surface Alignment:" << std::setw(valueWidth) << deviceProp.surfaceAlignment << "\n"
            << std::setw(nameWidth) << "Concurrent Copy and Kernel Execution:" << std::setw(valueWidth) << (deviceProp.deviceOverlap ? "Yes with " + std::to_string(deviceProp.asyncEngineCount) + " copy engine(s)" : "No") << "\n"
            << std::setw(nameWidth) << "Run Time Limit on Kernels:" << std::setw(valueWidth) << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") << "\n"
            << std::setw(nameWidth) << "Integrated GPU Sharing Host Memory:" << std::setw(valueWidth) << (deviceProp.integrated ? "Yes" : "No") << "\n"
            << std::setw(nameWidth) << "Unified Addressing (UVA):" << std::setw(valueWidth) << (deviceProp.unifiedAddressing ? "Yes" : "No") << "\n"
            << std::setw(nameWidth) << "PCI Bus ID:" << std::setw(valueWidth) << deviceProp.pciBusID << "\n"
            << std::setw(nameWidth) << "PCI Device ID:" << std::setw(valueWidth) << deviceProp.pciDeviceID << "\n"
            << std::setw(nameWidth) << "Pitch Alignment:" << std::setw(valueWidth) << deviceProp.texturePitchAlignment << "\n"
            //<< std::setw(nameWidth) << "Texture Cubemap Width:" << std::setw(valueWidth) << deviceProp.maxTextureCubemapWidth << "\n"
            << std::setw(nameWidth) << "Texture Cubemap Layered Dimensions:" << std::setw(valueWidth) << std::to_string(deviceProp.maxTextureCubemapLayered[0] )+ ", "+ std::to_string( deviceProp.maxTextureCubemapLayered[1]) << "\n"
            << std::setw(nameWidth) << "Surface 2D Layered Dimensions:" << std::setw(valueWidth) << std::to_string(deviceProp.maxSurface2DLayered[0])+ ", " + std::to_string( deviceProp.maxSurface2DLayered[1]) << "\n"
            //<< std::setw(nameWidth) << "Surface Cubemap Width:" << std::setw(valueWidth) << deviceProp.maxSurfaceCubemapWidth << "\n"
            << std::setw(nameWidth) << "Surface Cubemap Layered Dimensions:" << std::setw(valueWidth) << std::to_string(deviceProp.maxSurfaceCubemapLayered[0])+ ", " + std::to_string(deviceProp.maxSurfaceCubemapLayered[1]) << "\n"
            << std::setw(nameWidth) << "Maximum 1D Layered Texture Width and Layers:" << std::setw(valueWidth) << std::to_string(deviceProp.maxTexture1DLayered[0])+ ", " + std::to_string( deviceProp.maxTexture1DLayered[1] )<< "\n"
            << std::setw(nameWidth) << "Maximum 1D Linear Texture Width:" << std::setw(valueWidth) << deviceProp.maxTexture1DLinear << "\n"
            << std::setw(nameWidth) << "Maximum 2D Linear Texture Dimensions:" << std::setw(valueWidth) << std::to_string( deviceProp.maxTexture2DLinear[0])+ ", " + std::to_string( deviceProp.maxTexture2DLinear[1] )<< "\n"
            //<< std::setw(nameWidth) << "Maximum 2D Linear Texture Pitch (bytes):" << std::setw(valueWidth) << deviceProp.maxTexture2DLinearPitch << "\n"
            << std::setw(nameWidth) << "Maximum Mipmap Level for all Texture Types:" << std::setw(valueWidth) << deviceProp.maxTexture2DMipmap << "\n"
            << std::setw(nameWidth) << "Number of Async Engines:" << std::setw(valueWidth) << deviceProp.asyncEngineCount << "\n"
            << std::setw(nameWidth) << "Pageable Memory Access Uses Host Page Tables:" << std::setw(valueWidth) << (deviceProp.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No") << "\n"
            << std::setw(nameWidth) << "Direct Managed Memory Access from Host:" << std::setw(valueWidth) << (deviceProp.directManagedMemAccessFromHost ? "Yes" : "No") << "\n";

        std::cout << "\n";
    }

}

int main() {
    returnGPUCudaInfoResources();
    return 0;
}