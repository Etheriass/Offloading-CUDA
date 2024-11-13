
#include <iostream>
#include "off_vector.cuh"

__global__ void kernel(OffVector M){

    for (int i = 0; i < 11; i++){
        printf("%f\n", M[i]);
    }
}


int main() {
    std::cout << "Starting" << std::endl;
    bool finished = false;

    int max_capability = 10;
    OffVector M(max_capability, "matrix.bin");

    // Create CUDA streams
    cudaStream_t kernelStream, loadStream;
    cudaStreamCreate(&kernelStream);
    cudaStreamCreate(&loadStream);

    // Create event to monitor kernel completion
    cudaEvent_t kernelDoneEvent;
    cudaEventCreate(&kernelDoneEvent);

    // Launch kernel in kernelStream
    kernel<<<1, 1, 0, kernelStream>>>(M);
    cudaEventRecord(kernelDoneEvent, kernelStream); // Record completion event
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    while (!finished) {
        std::cout << "Waiting for event..." << std::endl;
        
        // Check if kernel has finished
        cudaError_t status = cudaStreamQuery(kernelStream);
        if (status == cudaSuccess) {
            finished = true;
        } else if (status == cudaErrorNotReady) {
            bool need_load;
            cudaMemcpyAsync(&need_load, M.d_need_load, sizeof(bool), cudaMemcpyDeviceToHost, loadStream);
            cudaStreamSynchronize(loadStream);
            if (need_load) {
                M.check_and_load(loadStream);
                cudaStreamSynchronize(loadStream);
            }
        } else {
            std::cerr << "Error in kernel execution: " << cudaGetErrorString(status) << std::endl;
            finished = true;
        }
    }

    // Cleanup
    cudaStreamDestroy(kernelStream);
    cudaStreamDestroy(loadStream);
    cudaEventDestroy(kernelDoneEvent);

    return 0;
}