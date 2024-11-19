// #include <string>
#include <iostream> // for cout
#include <random> // fir random, gen..
#include <fstream> // for ofstream, file

#define SIZE 100


void generate(const std::string& filename, size_t N){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f,1.0f);

    // init and fill matrix
    std::vector<float> M(N * N);
    std::cout << M.max_size() << std::endl;
    if (M.max_size() < N*N){
        throw std::runtime_error("TOO BIG");
    }

    for (size_t i = 0; i < N*N; i++){
        float c = dis(gen);
        float b = static_cast<float>(c);
        // std::cout << "float: " << c << ", bfloat: " << b << std::endl;
        std::cout << b << std::endl;
        M[i] = b;
    }

    // create file
    std::ofstream file(filename, std::ios::binary);
    if (!file){
        throw std::runtime_error("Failed to open file");
    }

    // write to file
    file.write(reinterpret_cast<const char*>(M.data()), M.size() * sizeof(float));

    file.close();
    if (!file.good()){
        throw std::runtime_error("Error occurred while writing the file");
    } else {
        std::cout << "File generated: " << filename << std::endl;
    }

}

int main(int argc, char* argv[]){
    // if (argc < 2){
    //     std::cerr << "Usage: " << argv[0] << " N" << std::endl;
    //     return 1;
    // }
    // int N = std::stoi(argv[1]);
    std::cout << "C++ version: " << __cplusplus << std::endl;

    size_t N = SIZE;
    std::cout << "Generation of " << N << "x" << N << " random float matrice"<< std::endl;

    std::string filename = "matrix.bin";

    try{
        generate(filename, N);
    } catch(const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }


    return 0;
}