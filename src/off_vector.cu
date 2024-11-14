#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << #call << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

class OffVector
{
private:
    float *d_data;
    size_t size;    // real size of the vector
    size_t *d_size; // pointer to size of the vector on device
    size_t *h_size; // pointer to size of the vector on host
    std::string filename;
    bool *d_need_load; // flag to indicate if data needs to be loaded (on device)
    bool h_need_load; // copy

    __host__ void load_chunk(size_t start, size_t chunk_size);

public:
    OffVector(size_t N, std::string filename);
    ~OffVector();

    __device__ float& operator[](size_t i);
    __host__ void check_and_load();
};

OffVector::OffVector(size_t N, std::string filename) : size(N), filename(filename), h_need_load(false) {
    printf("Creating Vector of size %lu from file \"%s\"\n", N, filename.c_str());
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    h_size = new size_t(0);
    CHECK_CUDA(cudaMalloc(&d_size, sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_size, &h_size, sizeof(size_t), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_need_load, sizeof(bool)));
    CHECK_CUDA(cudaMemcpy(d_need_load, &h_need_load, sizeof(bool), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(&h_need_load, d_need_load, sizeof(bool), cudaMemcpyDeviceToHost));
    // printf("Init load: %d\n", *h_need_load);

    load_chunk(0, N); // Load first chunk
}

OffVector::~OffVector(){
    cudaFree(d_data);
    cudaFree(d_size);
    cudaFree(d_need_load);
    delete h_size;
}

__device__ float& OffVector::operator[](size_t i){
    printf("Accessing element %lu:", i);
    if (i >= *d_size){
        // auto grid = cooperative_groups::this_grid();
        // printf("\nNeed to load...\n");
        // *d_need_load = true;
        // grid.sync();

        // while (i >= *d_size){} // Wait for data to be loaded
    }
    return d_data[i % size];
}

__host__ void OffVector::check_and_load(){
    printf("Checking if we need to load...\n");

    // cudaMemcpy(&h_need_load, d_need_load, sizeof(bool), cudaMemcpyDeviceToHost);
    CHECK_CUDA(cudaMemcpy(&h_need_load, d_need_load, sizeof(bool), cudaMemcpyDeviceToHost)); // ERROR: invalid argument

    printf("need_load: %d\n", h_need_load);

    if (h_need_load)
    {
        load_chunk(*h_size, 10); // Load the next chunk
        h_need_load = false;
        CHECK_CUDA(cudaMemcpy(d_need_load, &h_need_load, sizeof(bool), cudaMemcpyHostToDevice));
    } else {
        printf("No load needed\n");
    }
}

__host__ void OffVector::load_chunk(size_t start, size_t size){
    printf("Loading chunk from %lu to %lu\n", start, start + size);
   
    float *host_data = new float[size];

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()){
        delete[] host_data;
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streampos byte_offset = start * sizeof(float);
    file.seekg(byte_offset, std::ios::beg);
    if (file.fail()){
        delete[] host_data;
        throw std::runtime_error("Failed to seek to position in file");
    }

    file.read(reinterpret_cast<char *>(host_data), size * sizeof(float));
    // Check how many elements were actually read
    std::streamsize elements_read = file.gcount() / sizeof(float);
    if (elements_read < size){
        size = elements_read;
    }

    file.close();

    CHECK_CUDA(cudaMemcpy(d_data + start, host_data, elements_read * sizeof(float), cudaMemcpyHostToDevice));
    *h_size += elements_read;
    CHECK_CUDA(cudaMemcpy(d_size, &h_size, sizeof(size_t), cudaMemcpyHostToDevice));

    delete[] host_data;
}