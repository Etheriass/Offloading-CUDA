// #include <cstdio>


// __global__ void kernel(){
//     printf("Begin counting");
//     for (long long i = 0; i < 999999999999999999; i++){
//         printf("%lu", i);
//     }
    
//     printf("Counting finished\n");
// }

// int main(){

//     printf("Starting\n");

//     kernel<<<1,10>>>();

//     printf("CPU\n");
// }

// hello_world.cu

#include <cstdio>
__global__ void hello_world() {
    int i = threadIdx.x;
    printf("hello world from thread %d\n", i);
}

int main() {
    hello_world<<<1, 10>>>();
    // cudaDeviceSynchronize();
    printf("Execution ends\n");
}