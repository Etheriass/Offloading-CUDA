#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void tensorOperation(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 512; // Matrix size
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize host data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / float(RAND_MAX);
        h_B[i] = rand() / float(RAND_MAX);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Measure memory transfer time from host to device (for A)
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Memory Transfer Time (Host to Device for A): " << duration.count() << " ms\n";

    // Measure memory transfer time from host to device (for B)
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Memory Transfer Time (Host to Device for B): " << duration.count() << " ms\n";

    // Measure kernel execution time (matrix multiplication)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    start = std::chrono::high_resolution_clock::now();
    tensorOperation<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Kernel Execution Time: " << duration.count() << " ms\n";

    // Simulate page faults and measure the time for memory prefetching
    for (int i = 0; i < 5; i++) {
        start = std::chrono::high_resolution_clock::now();
        cudaMemPrefetchAsync(d_A, size, cudaCpuDeviceId);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "Page fault simulation #" << i + 1 << ": " << duration.count() << " ms\n";
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

