
#include <iostream>
#include "off_vector.cuh"

__global__ void kernel(OffVector* M){
    float c;
    for (int i = 0; i < 11; i++){
        c = (*M)[i];
        printf("%f\n", c);
    }
}


void check_and_load(OffVector* h_M, OffVector* d_M){
    // printf("Checking if we need to load...\n");
    int need_load = *&d_M->d_need_load;
    // printf("need_load: %d\n", need_load);

    if (need_load)
    {
        printf("\nNEED LOAD\n\n");
        h_M->load_chunk(10);
        printf("\nLOADED\n\n");
        *&d_M->d_need_load = 0;
        *&h_M->d_need_load = 0;
        cudaDeviceSynchronize();
    } else {
        // printf("No load needed\n");
    }
}


int main() {
    std::cout << "Starting" << std::endl;
    int it = 0;

    int max_capability = 10;
    OffVector h_M(max_capability, "matrix.bin");
    OffVector* d_M;
    cudaMallocManaged(&d_M, sizeof(OffVector));
    // cudaMemcpy(d_M, &h_M, sizeof(OffVector), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_M);

    while (true) {
        // std::cout << "Waiting for event..." << std::endl;
        
        check_and_load(&h_M, d_M);
        it += 1; 
    }

    cudaFree(d_M); 

    return 0;
}