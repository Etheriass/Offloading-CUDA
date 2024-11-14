
#include <iostream>
#include "off_vector.cuh"

__global__ void kernel(OffVector M){

    for (int i = 0; i < 11; i++){
        printf("%f\n", M[i]);
    }
}


int main() {
    std::cout << "Starting" << std::endl;
    // bool finished = false;
    int it = 0;

    int max_capability = 10;
    OffVector M(max_capability, "matrix.bin");

    kernel<<<1, 1>>>(M);

    while (it != 10) {
        cudaDeviceSynchronize();
        std::cout << "Waiting for event..." << std::endl;
        
        M.check_and_load();
        it += 2; 
    }

    return 0;
}