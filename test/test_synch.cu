#include <cstdio>
#include <iostream>
#include <vector>
#include <unistd.h>

__global__ void kernel(int* flag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (true) {
        if (idx == 0) {
            *flag = 1;
        }
        __nanosleep(1000); // Simulate some work with a sleep
    }
}

int main() {
    printf("Starting\n");

    // Allocate Unified Memory for d_flag
    int* d_flag;
    cudaMallocManaged(&d_flag, sizeof(int));
    *d_flag = 0; // Initialize the flag

    // Create a stream for the kernel
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

    // Launch the kernel in the stream
    kernel<<<1, 2>>>(d_flag);

    while (true) {
        printf("Looping\n");

        // Since d_flag is in Unified Memory, we can access it directly from the host
        int h_flag = *d_flag;
        printf("h_flag: %d\n", h_flag);

        // Sleep for a short period to avoid busy-waiting
        sleep(1);
    }

    // Clean up
    // cudaStreamDestroy(stream);
    cudaFree(d_flag);
    return 0;
}




// __global__ void kernel(){
//     while (true){
//         printf("Kernel loop");
//         for (unsigned long long i = 0; i < 99999999999999999; i++){
//             int c;
//             c = i/2;
//         }
//     }
// }

// int main (){
//     printf("Starting\n");
//     kernel<<<1,1>>>();
//     while (true){   
//         printf("Host loop\n");
//         for (unsigned long long i = 0; i < 999999; i++){
//             int c;
//             c = i/2;
//         }
//     }
// }


// class obj {
// public:
//     float a;
//     bool d_need_load;

//     obj(float a) : a(a) , d_need_load(false) {}

//     // __device__ void set_a() {
//     //     a = 13;
//     // }

//     __device__ float& operator[](size_t i){
//         a += 1;


//         return a;
//     }
// };

// void copy_a_to_host(obj* h_obj, obj* d_obj) {
//     cudaMemcpy(&h_obj->a, &d_obj->a, sizeof(int), cudaMemcpyDeviceToHost);
// }

// __global__ void kernel(obj* o) {
//     // o->set_a();

//     for (int i = 0; i < 100; i++){
//         float c = (*o)[0];
//         printf("%f\n", c);
//     }
// }

// int main() {
//     obj h_o(12);
//     obj* d_o;
//     cudaMalloc(&d_o, sizeof(obj));
//     cudaMemcpy(d_o, &h_o, sizeof(obj), cudaMemcpyHostToDevice);

//     kernel<<<1, 1>>>(d_o);

//     copy_a_to_host(&h_o, d_o);

//     printf("%f\n", h_o.a);

//     cudaFree(d_o);
// }


// #include <cstdio>
// __global__ void hello_world() {
//     int i = threadIdx.x;
//     printf("hello world from thread %d\n", i);
// }

// int main() {
//     hello_world<<<1, 10>>>();
//     cudaDeviceSynchronize();
//     printf("Execution ends\n");
// }