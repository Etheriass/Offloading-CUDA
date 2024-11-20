#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "OffLayer.cuh"

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

    // Sequentially multiply with each layer
    for (size_t layer_index = 0; layer_index < total_layers; ++layer_index) {
        size_t rows = matrices.getLayerRows(layer_index);
        size_t cols = matrices.getLayerCols(layer_index);

        // Load the layer matrix
        auto start_loading_layer = std::chrono::high_resolution_clock::now();
        float* device_matrix = matrices.getLayer(layer_index);
        auto end_loading_layer = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_loading_layer - start_loading_layer);
        std::cout << "Loading layer " << layer_index << " took: " << duration.count() << " us" << std::endl;

        size_t threads_per_block = 128;
        size_t blocks_per_grid = (rows + threads_per_block - 1) / threads_per_block;

        // Multiply the matrix with the input vector
        auto start_layer_multiplication = std::chrono::high_resolution_clock::now();
        matVecMul<<<blocks_per_grid, threads_per_block>>>(device_matrix, device_vector, device_result, rows, cols);
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
