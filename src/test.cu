#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

class LayeredMatrix {
private:
    size_t total_layers;
    std::vector<size_t> layer_rows;
    std::vector<size_t> layer_cols;
    std::ifstream file;
    float* device_matrix;
    size_t current_layer;

public:
    LayeredMatrix(const std::string& filename, size_t total_layers, const std::vector<size_t>& layer_rows, const std::vector<size_t>& layer_cols)
        : total_layers(total_layers), layer_rows(layer_rows), layer_cols(layer_cols), current_layer(-1) {
        file.open(filename, std::ios::binary);
        if (!file.is_open()){
            throw std::runtime_error("Failed to open file: " + filename);
        }
        cudaMalloc(&device_matrix, getMaxLayerSize() * sizeof(float));
    }

    ~LayeredMatrix() {
        cudaFree(device_matrix);
        file.close();
    }

    float* getLayer(size_t layer_index) {
        if (layer_index != current_layer) {
            loadLayer(layer_index);
            current_layer = layer_index;
        }
        return device_matrix;
    }

    void loadLayer(size_t layer_index) {
        size_t offset = 0;
        for (size_t i = 0; i < layer_index; ++i) {
            offset += layer_rows[i] * layer_cols[i] * sizeof(float);
        }
        file.seekg(offset, std::ios::beg);
        if (file.fail()){
            throw std::runtime_error("Failed to seek to position in file");
        }
        size_t elements_to_read = layer_rows[layer_index] * layer_cols[layer_index];
        std::vector<float> host_matrix(elements_to_read);
        file.read(reinterpret_cast<char*>(host_matrix.data()), elements_to_read * sizeof(float));
        cudaMemcpy(device_matrix, host_matrix.data(), elements_to_read * sizeof(float), cudaMemcpyHostToDevice);
    }

    size_t getLayerRows(size_t layer_index) {
        return layer_rows[layer_index];
    }

    size_t getLayerCols(size_t layer_index) {
        return layer_cols[layer_index];
    }

    size_t getMaxLayerSize() {
        size_t max_size = 0;
        for (size_t i = 0; i < total_layers; ++i) {
            size_t size = layer_rows[i] * layer_cols[i];
            if (size > max_size) {
                max_size = size;
            }
        }
        return max_size;
    }
};

// Kernel for matrix-vector multiplication
__global__ void matVecMul(float* matrix, float* vector, float* result, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

int main() {
    const std::string filename = "matrix.bin";
    size_t total_layers = 2;    // Number of layers
    std::vector<size_t> layer_rows = {100, 100};   // Rows for each layer
    std::vector<size_t> layer_cols = {50, 50};  // Columns for each layer

    auto start = std::chrono::high_resolution_clock::now();

    // Initialize LayeredMatrix
    LayeredMatrix matrices(filename, total_layers, layer_rows, layer_cols);

    // Build a random input vector for the first layer
    std::vector<float> host_vector(layer_cols[0]);
    srand(static_cast<unsigned>(time(0)));
    for (size_t i = 0; i < layer_cols[0]; ++i) {
        host_vector[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory for vectors
    float* device_vector;
    float* device_result;
    cudaMalloc(&device_vector, layer_cols[0] * sizeof(float));
    cudaMalloc(&device_result, matrices.getMaxLayerSize() * sizeof(float));

    // Copy input vector to device
    cudaMemcpy(device_vector, host_vector.data(), layer_cols[0] * sizeof(float), cudaMemcpyHostToDevice);

    // Sequentially multiply with each layer
    for (size_t layer_index = 0; layer_index < total_layers; ++layer_index) {
        size_t rows = matrices.getLayerRows(layer_index);
        size_t cols = matrices.getLayerCols(layer_index);

        auto start_loading_layer = std::chrono::high_resolution_clock::now();
        float* device_matrix = matrices.getLayer(layer_index);
        auto end_loading_layer = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_loading_layer - start_loading_layer);
        std::cout << "Loading layer " << layer_index << " took:" << duration.count() << " ns" << std::endl;


        // Define grid and block dimensions
        size_t threads_per_block = 256;
        size_t blocks_per_grid = (rows + threads_per_block - 1) / threads_per_block;

        // Multiply current vector with the layer matrix
        matVecMul<<<blocks_per_grid, threads_per_block>>>(device_matrix, device_vector, device_result, rows, cols);
        cudaDeviceSynchronize();

        // Update the input vector for the next iteration
        cudaFree(device_vector);
        cudaMalloc(&device_vector, rows * sizeof(float));
        cudaMemcpy(device_vector, device_result, rows * sizeof(float), cudaMemcpyDeviceToDevice);

        // Update host vector size for final result
        if (layer_index == total_layers - 1) {
            host_vector.resize(rows);
        }
    }

    // Copy final result back to host
    cudaMemcpy(host_vector.data(), device_result, host_vector.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(device_vector);
    cudaFree(device_result);

    // Output the final result (optional)
    std::cout << "Final output vector:" << std::endl;
    for (size_t i = 0; i < host_vector.size(); ++i) {
        std::cout << host_vector[i] << " ";
    }
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference took: " << duration.count() << " ms" << std::endl;

    return 0;
}
