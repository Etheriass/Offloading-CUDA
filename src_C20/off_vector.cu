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
    OffVector(size_t N, std::string filename);
    ~OffVector();

    __device__ __host__ float& operator[](size_t i)
    {
        if (i >= *d_size)
        {
            *d_need_load = true;
            while (i >= *d_size){} // Wait for data to be loaded
        }
        return d_data[i];
    }
    __host__ void check_and_load();
};

OffVector::OffVector(size_t N, std::string filename) : size(N), filename(filename)
{
    cudaMalloc(&d_data, N * sizeof(float));

    h_size = new size_t(0);
    cudaMalloc(&d_size, sizeof(size_t));
    cudaMemcpy(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice);

    h_need_load = new bool(false);
    cudaMalloc(&d_need_load, sizeof(bool));
    cudaMemcpy(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice);
}

OffVector::~OffVector()
{
    cudaFree(d_data);
    cudaFree(d_size);
    cudaFree(d_need_load);
    delete h_size;
    delete h_need_load;
}

// __device__ __host__ float OffVector::operator[](size_t i) const {
//     if (i >= *d_size) {
//         *d_need_load = true;
//         while (i >= *d_size) { } // Wait for data to be loaded
//     }
//     return d_data[i];
// }

__host__ void OffVector::check_and_load()
{
    bool need_load;
    cudaMemcpy(&need_load, d_need_load, sizeof(bool), cudaMemcpyDeviceToHost);
    if (need_load)
    {
        load_chunk(*h_size, 100);
        *h_need_load = false;
        cudaMemcpy(d_need_load, h_need_load, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice);
    }
}

__host__ void OffVector::load_chunk(size_t start, size_t size)
{
    float *host_data = new float[size];

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streampos byte_offset = start * sizeof(float);
    file.seekg(byte_offset, std::ios::beg);
    if (file.fail())
    {
        throw std::runtime_error("Failed to seek to position in file");
    }

    file.read(reinterpret_cast<char *>(host_data), size * sizeof(float));
    if (file.fail() && !file.eof())
    {
        throw std::runtime_error("Error reading from file");
    }

    // Check how many elements were actually read
    std::streamsize elements_read = file.gcount() / sizeof(float);
    if (elements_read < size)
    {
        size = elements_read;
    }

    file.close();

    cudaMemcpy(d_data, host_data, size * sizeof(float), cudaMemcpyHostToDevice);

    *h_size = size;
}