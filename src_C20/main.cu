#include <stdio.h>
#include "off_vector.cuh"

__global__ void kernel(OffVector M){
    printf("%f\n", M[0]);
}

int main(){
    int max_capability = 10;
    OffVector M(max_capability, "matrix.bin");

    kernel<<<1, 1>>>(M);

    while (1)
    {
        M.check_and_load();
    }
    


    return 0;
}