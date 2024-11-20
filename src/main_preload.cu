#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
// #include "OffLayer.cuh"

class OffLayer {
private:
    size_t total_layers;
    std::vector<size_t> layer_rows;
    std::vector<size_t> layer_cols;
    std::ifstream file;
    float* device_matrix;
    size_t current_layer;

public:
    OffLayer(const std::string& filename, size_t total_layers, const std::vector<size_t>& layer_rows, const std::vector<size_t>& layer_cols)
        : total_layers(total_layers), layer_rows(layer_rows), layer_cols(layer_cols), current_layer(1) {
        file.open(filename, std::ios::binary);
        if (!file.is_open()){
            throw std::runtime_error("Failed to open file: " + filename);
        }
        cudaMalloc(&device_matrix, getMaxLayerSize() * sizeof(float));
    }

    ~OffLayer() {
        cudaFree(device_matrix);
        file.close();
    }

    float* getLayer(size_t layer_index) {
        if (layer_index != current_layer) {
            current_layer = layer_index;
        }
        return device_matrix;
    }

    void preloadLayer(size_t layer_index, std::vector<float>& host_matrix) {
        size_t offset = 0;
        for (size_t i = 0; i < layer_index; ++i) {
            offset += layer_rows[i] * layer_cols[i] * sizeof(float);
        }
        file.seekg(offset, std::ios::beg);
        if (file.fail()){
            throw std::runtime_error("Failed to seek to position in file");
        }
        size_t elements_to_read = layer_rows[layer_index] * layer_cols[layer_index];
        host_matrix.resize(elements_to_read);
        file.read(reinterpret_cast<char*>(host_matrix.data()), elements_to_read * sizeof(float));
    }

    void loadPreloadedLayer(const std::vector<float>& host_matrix) {
        cudaMemcpy(device_matrix, host_matrix.data(), host_matrix.size() * sizeof(float), cudaMemcpyHostToDevice);
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

/* Kernel to multiply a matrix with a vector */
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

/* Function to run inference on a set of layers stored in a file */
void runInference(const std::string &filename, size_t total_layers, std::vector<size_t> layer_rows, std::vector<size_t> layer_cols) {
    auto start = std::chrono::high_resolution_clock::now();

    OffLayer matrices(filename, total_layers, layer_rows, layer_cols);

    // Build a random input vector for first layer
    std::vector<float> host_vector(layer_cols[0]);
    srand(static_cast<unsigned>(time(0)));
    for (size_t i = 0; i < layer_cols[0]; ++i) {
        host_vector[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* device_vector;
    float* device_result;
    cudaMalloc(&device_vector, layer_cols[0] * sizeof(float));
    cudaMalloc(&device_result, matrices.getMaxLayerSize() * sizeof(float));
    cudaMemcpy(device_vector, host_vector.data(), layer_cols[0] * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> preloaded_matrix;
    matrices.preloadLayer(0, preloaded_matrix);

    // Sequentially multiply with each layer
    for (size_t layer_index = 0; layer_index < total_layers; ++layer_index) {
        size_t rows = matrices.getLayerRows(layer_index);
        size_t cols = matrices.getLayerCols(layer_index);

        // Load the layer matrix
        auto start_loading_layer = std::chrono::high_resolution_clock::now();
        matrices.loadPreloadedLayer(preloaded_matrix);
        auto end_loading_layer = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_loading_layer - start_loading_layer);
        std::cout << "Loading layer " << layer_index << " took: " << duration.count() << " us" << std::endl;

        if (layer_index + 1 < total_layers) {
            matrices.preloadLayer(layer_index + 1, preloaded_matrix);
        }

        size_t threads_per_block = 256;
        size_t blocks_per_grid = (rows + threads_per_block - 1) / threads_per_block;

        // Multiply the matrix with the input vector
        auto start_layer_multiplication = std::chrono::high_resolution_clock::now();
        matVecMul<<<blocks_per_grid, threads_per_block>>>(matrices.getLayer(layer_index), device_vector, device_result, rows, cols);
        cudaDeviceSynchronize();
        auto end_layer_multiplication = std::chrono::high_resolution_clock::now();
        auto duration_layer_multiplication = std::chrono::duration_cast<std::chrono::microseconds>(end_layer_multiplication - start_layer_multiplication);
        std::cout << "Multiplying layer " << layer_index << " took: " << duration_layer_multiplication.count() << " us" << std::endl;

        // Update the input vector for the next iteration
        cudaFree(device_vector);
        cudaMalloc(&device_vector, rows * sizeof(float));
        cudaMemcpy(device_vector, device_result, rows * sizeof(float), cudaMemcpyDeviceToDevice);

        if (layer_index == total_layers - 1) {
            host_vector.resize(rows);
        }
    }

    cudaMemcpy(host_vector.data(), device_result, host_vector.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_vector);
    cudaFree(device_result);

    // std::cout << "Final output vector:" << std::endl;
    // for (size_t i = 0; i < host_vector.size(); ++i) {
    //     std::cout << host_vector[i] << " ";
    // }
    // std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference took: " << duration.count() << " ms" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <total_layers>" << std::endl;
        return 1;
    }

    const std::string &filename = argv[1];
    size_t total_layers = std::stoi(argv[2]);
    std::vector<size_t> layer_rows = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000};
    std::vector<size_t> layer_cols = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000};

    runInference(filename, total_layers, layer_rows, layer_cols);
    return 0;
}
