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

    // Use double buffering for host matrices
    auto start_host_matrix_allocation = std::chrono::high_resolution_clock::now();
    float* host_matrix[2]; // allocate host_matrix using cudaMallocHost to get pinned host memory, allowing cudaMemcpyAsync to be asynchronous.
    for (int i = 0; i < 2; ++i) {
        cudaMallocHost(&host_matrix[i], matrices.getMaxLayerSize() * sizeof(float)); // cudaMallocHost is slow because it has to manipulate the page table
    }
    auto end_host_matrix_allocation = std::chrono::high_resolution_clock::now();
    auto duration_host_matrix_allocation = std::chrono::duration_cast<std::chrono::microseconds>(end_host_matrix_allocation - start_host_matrix_allocation);
    std::cout << "Allocating pinned host matrices took: " << duration_host_matrix_allocation.count() << " us" << std::endl;

    size_t current_buffer = 0;


    // CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Preload the first layer
    matrices.preloadLayer(0, host_matrix[current_buffer]);

    // Sequentially process each layer
    for (size_t layer_index = 0; layer_index < total_layers; ++layer_index) {
        size_t rows = matrices.getLayerRows(layer_index);
        size_t cols = matrices.getLayerCols(layer_index);


        matrices.loadPreloadedLayer(host_matrix[current_buffer], host_matrix[(current_buffer+1)%2], layer_index, stream);

        size_t threads_per_block = 256;
        size_t blocks_per_grid = (rows + threads_per_block - 1) / threads_per_block;
        auto start_layer_multiplication = std::chrono::high_resolution_clock::now();
        matVecMul<<<blocks_per_grid, threads_per_block>>>(matrices.getLayer(), device_vector, device_result, rows, cols);
        cudaDeviceSynchronize();
        auto end_layer_multiplication = std::chrono::high_resolution_clock::now();
        auto duration_layer_multiplication = std::chrono::duration_cast<std::chrono::microseconds>(end_layer_multiplication - start_layer_multiplication);
        std::cout << "Multiplying layer " << layer_index << " took: " << duration_layer_multiplication.count() << " us" << std::endl;

        // Swap buffers
        current_buffer = (current_buffer + 1) % 2;

        // Update the input vector for the next iteration
        auto start_save_intermediate_result = std::chrono::high_resolution_clock::now();
        cudaMemcpy(device_vector, device_result, rows * sizeof(float), cudaMemcpyDeviceToDevice);
        auto end_save_intermediate_result = std::chrono::high_resolution_clock::now();
        auto duration_save_intermediate_result = std::chrono::duration_cast<std::chrono::microseconds>(end_save_intermediate_result - start_save_intermediate_result);
        std::cout << "Saving intermediate result of layer " << layer_index << " took: " << duration_save_intermediate_result.count() << " us" << std::endl;
    }


    auto start_copy_result = std::chrono::high_resolution_clock::now();
    cudaMemcpy(host_vector.data(), device_result, host_vector.size() * sizeof(float), cudaMemcpyDeviceToHost);
    auto end_copy_result = std::chrono::high_resolution_clock::now();
    auto duration_copy_result = std::chrono::duration_cast<std::chrono::microseconds>(end_copy_result - start_copy_result);
    std::cout << "Copying result back to host took: " << duration_copy_result.count() << " us" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference took: " << duration.count() << " ms" << std::endl;

    // Destroy the stream
    cudaStreamDestroy(stream);

    cudaFree(device_vector);
    cudaFree(device_result);

    // std::cout << "Final output vector:" << std::endl;
    // for (size_t i = 0; i < host_vector.size(); ++i) {
    //     std::cout << host_vector[i] << " ";
    // }
    // std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <filename> <total_layers> <layers_dim>" << std::endl;
        return 1;
    }

    const std::string &filename = argv[1];
    size_t total_layers = std::stoi(argv[2]);
    size_t layers_dim = std::stoi(argv[3]); // We assume all layers are square and have the same dimensions
    std::vector<size_t> layer_rows;
    std::vector<size_t> layer_cols;
    for (size_t i = 0; i < total_layers; ++i) {
        layer_rows.push_back(layers_dim);
        layer_cols.push_back(layers_dim);
    }

    runInference(filename, total_layers, layer_rows, layer_cols);
    return 0;
}
