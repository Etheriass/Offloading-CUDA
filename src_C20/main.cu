
#include <iostream>
#include "off_vector.cuh"

__global__ void kernel(OffVector M){

    for (int i = 0; i < 10; i++){
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

    // Launch kernel in kernelStream
    kernel<<<1, 1, 0, kernelStream>>>(M);

    int i = 0;
    while (!finished) {
        i++;
        std::cout << "Waiting for event..." << std::endl;
        printf("i: %d\n", i);
        // Wait for the "need load" event
        cudaError_t eventStatus = cudaEventQuery(M.needLoadEvent);
        if (eventStatus == cudaSuccess) {
            printf("Event triggered\n");
            // Event triggered: load next chunk in loadStream
            M.check_and_load(loadStream);
            cudaStreamSynchronize(loadStream);

            // Reset the event for the next load
            cudaEventRecord(M.needLoadEvent, kernelStream);
        } else if (eventStatus != cudaErrorNotReady) {
            // Check if there was an error other than "not ready"
            std::cerr << "Error waiting for event: " << cudaGetErrorString(eventStatus) << std::endl;
            finished = true;
        } else {
            // Event not ready: continue processing
            std::cout << "Event not ready" << std::endl;
        }
    }

    // Cleanup
    cudaStreamDestroy(kernelStream);
    cudaStreamDestroy(loadStream);

    return 0;
}