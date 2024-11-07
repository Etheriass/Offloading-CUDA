#ifndef OFF_VECTOR_CUH
#define OFF_VECTOR_CUH

#include <string>
#include <atomic>

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
            while (i >= *d_size)
            {} // Wait for data to be loaded
        }
        return d_data[i];
    }
    __host__ void check_and_load();
};

#endif