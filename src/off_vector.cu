#include <cuda_runtime.h>
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
    bool *d_need_load; // flag to indicate if data needs to be loaded (on device)
    bool *h_need_load; // flag to indicate if data needs to be loaded (on host)

    __host__ void load_chunk(size_t start, size_t chunk_size);

public:
    cudaEvent_t needLoadEvent;
    OffVector(size_t N, std::string filename);
    ~OffVector();

    // __device__ float& operator[](size_t i);
    __host__ void check_and_load(cudaStream_t stream);
};

OffVector::OffVector(size_t N, std::string filename) : size(N), filename(filename)
{
    printf("Creating Vector of size %lu from file \"%s\"\n", N, filename.c_str());
    cudaMalloc(&d_data, N * sizeof(float));

    h_size = new size_t(0);
    cudaMalloc(&d_size, sizeof(size_t));
    cudaMemcpy(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice);

    h_need_load = new bool(false);
    cudaMalloc(&d_need_load, sizeof(bool));
    cudaMemcpy(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice);

    // Initialize the CUDA event
    cudaEventCreate(&needLoadEvent);

    load_chunk(0, N); // Load first chunk
}

OffVector::~OffVector()
{
    cudaFree(d_data);
    cudaFree(d_size);
    cudaFree(d_need_load);
    delete h_size;
    delete h_need_load;
    cudaEventDestroy(needLoadEvent);
}


__host__ void OffVector::check_and_load(cudaStream_t stream) {
    printf("Checking if we need to load...\n");

    bool need_load;
    cudaMemcpyAsync(&need_load, d_need_load, sizeof(bool), cudaMemcpyDeviceToHost, stream);
    printf("Need load: %d\n", need_load);
    cudaStreamSynchronize(stream);  // Ensure we have the updated flag
    printf("Need load: %d\n", need_load);

    if (need_load) {
        load_chunk(*h_size, 10);  // Load the next chunk
        *h_need_load = false;
        cudaMemcpyAsync(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice, stream);
    }
}

__host__ void OffVector::load_chunk(size_t start, size_t size)
{
    printf("Loading chunk from %lu to %lu\n", start, start + size);
    float *host_data = new float[size];

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streampos byte_offset = start * sizeof(float);
    file.seekg(byte_offset, std::ios::beg);
    if (file.fail()){
        throw std::runtime_error("Failed to seek to position in file");
    }

    file.read(reinterpret_cast<char *>(host_data), size * sizeof(float));
    if (file.fail() && !file.eof()){
        throw std::runtime_error("Error reading from file");
    }

    // Check how many elements were actually read
    std::streamsize elements_read = file.gcount() / sizeof(float);
    if (elements_read < size){
        size = elements_read;
    }

    file.close();

    cudaMemcpy(d_data, host_data, size * sizeof(float), cudaMemcpyHostToDevice);

    *h_size = size;
    cudaMemcpy(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice);

}