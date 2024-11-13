#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <string>
#include <fstream>
#include <stdexcept>

class OffVector
{
private:
    float *d_data;
    size_t size;    // real size of the vector
    size_t *d_size; // pointer to size of the vector on device
    size_t *h_size; // pointer to size of the vector on host
    std::string filename;
    bool *h_need_load; // flag to indicate if data needs to be loaded (on host)

    __host__ void load_chunk(size_t start, size_t chunk_size, cudaStream_t stream);

public:
    bool *d_need_load; // flag to indicate if data needs to be loaded (on device)
    cudaEvent_t needLoadEvent;
    OffVector(size_t N, std::string filename);
    ~OffVector();

    __device__ float& operator[](size_t i);
    __host__ void check_and_load(cudaStream_t stream);
};

OffVector::OffVector(size_t N, std::string filename) : size(N), filename(filename){
    printf("Creating Vector of size %lu from file \"%s\"\n", N, filename.c_str());
    cudaMalloc(&d_data, N * sizeof(float));

    h_size = new size_t(0);
    cudaMalloc(&d_size, sizeof(size_t));
    // cudaMemcpy(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice);

    h_need_load = new bool(false);
    cudaMalloc(&d_need_load, sizeof(bool));
    // cudaMemcpy(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice);

    // Initialize the CUDA event
    cudaEventCreate(&needLoadEvent);

    load_chunk(0, N, cudaStreamDefault); // Load first chunk
}

OffVector::~OffVector(){
    cudaFree(d_data);
    cudaFree(d_size);
    cudaFree(d_need_load);
    delete h_size;
    delete h_need_load;
    cudaEventDestroy(needLoadEvent);
}

__device__ float& OffVector::operator[](size_t i){
    printf("Accessing element %lu:", i);
    if (i >= *d_size){
        auto grid = cooperative_groups::this_grid();
        printf("\nNeed to load...\n");
        *d_need_load = true;
        // atomicExch((int*)d_need_load, 1);
        __threadfence_system();
        // grid.sync();

        // while (i >= *d_size){grid.sync();} // Wait for data to be loaded
        while (i >= *d_size){__threadfence_system();} // Wait for data to be loaded
    }
    return d_data[i % size];
}

__host__ void OffVector::check_and_load(cudaStream_t stream){
    printf("Checking if we need to load...\n");

    bool need_load;
    cudaMemcpyAsync(&need_load, d_need_load, sizeof(bool), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Ensure we have the updated flag

    if (need_load)
    {
        load_chunk(*h_size, 10, stream); // Load the next chunk
        *h_need_load = false;
        cudaMemcpyAsync(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice, stream);
        // cudaMemcpyAsync(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream); // Ensure we have the updated flag
    }
}

__host__ void OffVector::load_chunk(size_t start, size_t size, cudaStream_t stream){
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

    cudaMemcpyAsync(d_data + start, host_data, elements_read * sizeof(float), cudaMemcpyHostToDevice, stream);
    *h_size += elements_read;
    cudaMemcpyAsync(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice, stream);

    delete[] host_data;
}