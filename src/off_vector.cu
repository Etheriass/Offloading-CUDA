#include <cuda_runtime.h>
#include <stdfloat> // for bfloat16_t
#include <string> // for string
#include <fstream> // for ifstream
#include <stdexcept> // for runtime_error


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

OffVector::OffVector(size_t N, std::string filename){
    this->size = N;
    this->filename = filename;
    this->d_size = 0;
    this->d_data = new bfloat16_t[N];
}

__device__ __host__ bfloat16_t& OffVector::operator[](size_t i){
    if (i >= size){
        bool is_loaded = false;
        load_chunk(i, 100, *is_loaded);
        if (!is_loaded){
            return 0;
        }
    }
    return d_data[i];
}

void OffVector::load_chunk(size_t i, size_t size){
    bfloat16_t* host_data = new bfloat16_t[size];

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streampos byte_offset = start_index * sizeof(bfloat16_t);
    file.seekg(byte_offset, std::ios::beg);
    if (file.fail()) {
        throw std::runtime_error("Failed to seek to position in file");
    }

    file.read(reinterpret_cast<char*>(host_data.data()), chunk_size * sizeof(float));
    if (file.fail() && !file.eof()) {
        throw std::runtime_error("Error reading from file");
    }

    // Check how many elements were actually read
    std::streamsize elements_read = file.gcount() / sizeof(float);
    if (elements_read < static_cast<std::streamsize>(chunk_size)) {
        host_data.resize(elements_read);
    }

    file.close();

    cudaMemcpy(d_data + start_index, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);

    loaded_size = start_index + host_data.size();
}