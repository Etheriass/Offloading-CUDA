#include <stdio.h>
#include "off_vector.cuh"

#define SIZE 10

int main(){

    OffVector M(SIZE, "matrix.bin");

    M.load_chunk(0, 10);

    for (int i = 0; i < SIZE; i++){
        printf("%f\n", M[i]);
    }


    return 0;
}