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
    std::string filename;

public:
    size_t size;    // real size of the vector
    size_t d_size; // pointer to size of the vector on device
    int d_need_load; // flag to indicate if data needs to be loaded (on device)
    int h_need_load; // copy
    OffVector(size_t N, std::string filename);
    ~OffVector();

    __device__ float& operator[](size_t i);
    // __host__ void check_and_load();
    __host__ void load_chunk(size_t chunk_size);
};

OffVector::OffVector(size_t N, std::string filename) : size(N), filename(filename), h_need_load(0), d_need_load(0), d_size(0) {
    printf("Creating Vector of size %lu from file \"%s\"\n", N, filename.c_str());
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    load_chunk(N); // Load first chunk
}

OffVector::~OffVector(){
    cudaFree(d_data);
}


__device__ float& OffVector::operator[](size_t i){
    printf("Accessing element %lu:", i);
    if (i >= d_size){
        auto grid = cooperative_groups::this_grid();
        printf("\nNeed to load...\n");
        d_need_load = 1;
        // grid.sync();
        printf("HELLO\n");
        // __threadfence();
        int looping = d_need_load;

        // while (i >= d_size){} // Wait for data to be loaded
        while (looping){
            looping = d_need_load;
            if (d_need_load == 0){
                break;
            }
            // printf("Looping: %d", looping);
        }
        
        printf("RELEASED\n");
        d_data = d_data;
        for (size_t i = 0; i <size; i++){
            printf("->>%f\n", d_data[i]);
        }
    }
    return d_data[i % size];
}


__host__ void OffVector::load_chunk(size_t size){
    size_t start = d_size;
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
    printf("Elements read: %d\n", elements_read);
    CHECK_CUDA(cudaMemcpy(d_data, host_data, elements_read * sizeof(float), cudaMemcpyHostToDevice));
    // cudaMemcpy(d_data, host_data, elements_read * sizeof(float), cudaMemcpyHostToDevice);
    d_size += elements_read;

    for (size_t i = 0; i <size; i++){
        printf("->%f\n", host_data[i]);
    }
    cudaDeviceSynchronize();

    delete[] host_data;
}