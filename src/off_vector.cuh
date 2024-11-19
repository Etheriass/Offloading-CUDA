#ifndef OFF_VECTOR_CUH
#define OFF_VECTOR_CUH

#include <cuda_runtime.h>
#include <string>

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
    __host__ void load_chunk( size_t chunk_size);
};

#endif