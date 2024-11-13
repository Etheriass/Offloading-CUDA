#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

class OffVector {
private:
    float* d_data;
    size_t size; // real size of the vector
    size_t* d_size; // pointer to size of the vector on device
    size_t* h_size; // pointer to size of the vector on host
    std::string filename;
    bool* d_need_load; // flag to indicate if data needs to be loaded (on device)
    bool* h_need_load; // flag to indicate if data needs to be loaded (on host)

    void load_chunk(size_t start, size_t chunk_size);

public:
    OffVector(size_t N, std::string filename);
    ~OffVector();
    __host__ float& host_access(size_t i); // host access function
    __device__ float device_access(size_t i) const; // device access function
    __device__ __host__ float operator[](size_t i) ;
    void check_and_load();
};

OffVector::OffVector(size_t N, std::string filename) : size(N), filename(filename) {
    cudaMalloc(&d_data, N * sizeof(float));
    
    h_size = new size_t(0);
    cudaMalloc(&d_size, sizeof(size_t));
    cudaMemcpy(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice);

    h_need_load = new bool(false);
    cudaMalloc(&d_need_load, sizeof(bool));
    cudaMemcpy(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice);
}

OffVector::~OffVector() {
    cudaFree(d_data);
    cudaFree(d_size);
    cudaFree(d_need_load);
    delete h_size;
    delete h_need_load;
}

// Host access function for `operator[]` that triggers loading if necessary
__host__ float& OffVector::host_access(size_t i) {
    if (i >= *h_size) {
        *h_need_load = true;
        check_and_load(); // Ensure data is loaded
    }
    return d_data[i];
}

// Device access function that assumes data is already loaded
__device__ float OffVector::device_access(size_t i) const {
    return d_data[i];
}

// Unified `operator[]` that switches between host and device access
__device__ __host__ float OffVector::operator[](size_t i) {
#ifdef __CUDA_ARCH__
    return device_access(i);
#else
    return host_access(i);
#endif
}

void OffVector::check_and_load() {
    bool need_load;
    cudaMemcpy(&need_load, d_need_load, sizeof(bool), cudaMemcpyDeviceToHost);
    if (need_load) {
        load_chunk(*h_size, 100); // Load next chunk
        *h_need_load = false;
        cudaMemcpy(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice);
    }
}

void OffVector::load_chunk(size_t start, size_t size){
    float* host_data = new float[size];

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streampos byte_offset = start * sizeof(float);
    file.seekg(byte_offset, std::ios::beg);
    if (file.fail()) {
        throw std::runtime_error("Failed to seek to position in file");
    }

    file.read(reinterpret_cast<char*>(host_data), size * sizeof(float));
    if (file.fail() && !file.eof()) {
        throw std::runtime_error("Error reading from file");
    }

    // Check how many elements were actually read
    std::streamsize elements_read = file.gcount() / sizeof(float);
    if (elements_read < size) {
        size = elements_read;
    }

    file.close();

    cudaMemcpy(d_data + start, host_data, size * sizeof(float), cudaMemcpyHostToDevice);

    *h_size = start + size;
}
