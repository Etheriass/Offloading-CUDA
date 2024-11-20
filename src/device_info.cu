// device_info.cu
#include <iostream>
#include <cuda_runtime.h>
#include <sys/sysinfo.h>
#include <fstream>
#include <string>
#include <thread>

// Function to print GPU Information using CUDA APIs
void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Number of CUDA-capable GPUs: " << deviceCount << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "GPU " << device << " - " << deviceProp.name << ":" << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << (deviceProp.sharedMemPerBlock / 1024.0) << " KB" << std::endl;
        std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Maximum Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum Thread Dimensions: [" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Maximum Grid Dimensions: [" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]" << std::endl;
    }
}

// Function to read and print CPU Information from /proc/cpuinfo
void printCPUInfo() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::string model_name;
    std::string cpu_cores;
    std::string cpu_mhz;
    std::string cache_size;
    bool info_found = false;

    if (cpuinfo.is_open()) {
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                model_name = line.substr(line.find(":") + 2);
                info_found = true;
            }
            if (line.find("cpu cores") != std::string::npos) {
                cpu_cores = line.substr(line.find(":") + 2);
            }
            if (line.find("cpu MHz") != std::string::npos) {
                cpu_mhz = line.substr(line.find(":") + 2);
            }
            if (line.find("cache size") != std::string::npos) {
                cache_size = line.substr(line.find(":") + 2);
            }
        }
        cpuinfo.close();
    }

    if (info_found) {
        std::cout << "CPU Model: " << model_name << std::endl;
        std::cout << "Number of CPU Cores: " << cpu_cores << std::endl;
        std::cout << "CPU Frequency: " << cpu_mhz << " MHz" << std::endl;
        std::cout << "Cache Size: " << cache_size << std::endl;
    } else {
        std::cerr << "Unable to read CPU information from /proc/cpuinfo" << std::endl;
    }
}

// Function to print RAM Information using sysinfo
void printRAMInfo() {
    struct sysinfo sys_info;

    if (sysinfo(&sys_info) == 0) {
        std::cout << "Total RAM: " << (sys_info.totalram / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "Free RAM: " << (sys_info.freeram / (1024 * 1024)) << " MB" << std::endl;
    } else {
        std::cerr << "Error retrieving RAM information." << std::endl;
    }
}

int main() {
    std::cout << "==== GPU Information ====" << std::endl;
    printGPUInfo();

    std::cout << "\n==== CPU Information ====" << std::endl;
    printCPUInfo();

    std::cout << "\n==== RAM Information ====" << std::endl;
    printRAMInfo();

    return 0;
}
