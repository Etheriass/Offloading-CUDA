#include <cstdlib>
#include <iostream>
#include "add.cuh"


__global__ void add_kernel(int A[], int B[], int C[], int N){
    // Get thread ID.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N){
        C[tid] = A[tid] + B[tid];
    }   
}


void add(int A[], int B[], int C[], int N){

    // Intialize device pointers
    int *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    // Transfer Arrays to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Config threads
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blocks_per_grid = deviceProp.multiProcessorCount;
    int threads_per_blocks = 128; // Adjust if needed

    std::cout << "Number of threads:" << blocks_per_grid*threads_per_blocks << std::endl;

    add_kernel<<<blocks_per_grid, threads_per_blocks>>>(d_A, d_B, d_C, N);

    // Copy the result
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
}