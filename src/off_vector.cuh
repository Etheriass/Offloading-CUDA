#ifndef OFF_VECTOR_CUH
#define OFF_VECTOR_CUH

class OffVector {
    private:
        bfloat16_t* d_data; // data on device
        size_t size; // real size of the vector
        size_t d_size; // size of the vector on device
        std::string filename; // name of the file
    

    public:
        OffVector(size_t N, std::string filename);

        __device__ __host__ bfloat16_t& operator[](size_t i);

        void load_chunk(size_t i, size_t size);

}
