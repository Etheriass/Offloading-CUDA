#include <fstream>
#include <vector>
#include <iostream>

#define SIZE 100

int main(int argc, char* agrv[]){
    size_t N = SIZE;

    // Open the binary file for reading
    std::ifstream file("matrix.bin", std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading");
    }

    // Container for the matrix
    std::vector<float> matrix(N);

    // Load the file in the matix
    file.read(reinterpret_cast<char*>(matrix.data()), 4*N); // One char is one byte, bfloat16 is two bytes, float is four bytes

    // Close the file
    file.close();
    if (!file.good()) {
        throw std::runtime_error("Error occurred while reading the file");
    }

    for (float c : matrix){
        std::cout << c << std::endl;
    }


    return 0;
}