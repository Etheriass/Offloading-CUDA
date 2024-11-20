#include <iostream>
#include <random>
#include <fstream>

// #define SIZE 1000000000

void generate(const std::string& filename, size_t N){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f,1.0f);

    // init and fill matrix
    std::vector<float> M(N);
    // std::cout << M.max_size() << std::endl;
    if (M.max_size() < N){
        throw std::runtime_error("TOO BIG");
    }

    for (size_t i = 0; i < N; i++){
        float c = dis(gen);
        float b = static_cast<float>(c);
        // std::cout << b << std::endl;
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
    if (argc != 3){
        std::cerr << "Usage: " << argv[0] << " <filename> <N>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    size_t N = std::stoi(argv[2]);
    std::cout << "Generation of " << N << " random float matrice"<< std::endl;

    try{
        generate(filename, N);
    } catch(const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}