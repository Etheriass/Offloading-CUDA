#include <cstdlib>

#include <vector>

using Matrix = std::vector<std::vector<double>>;

Matrix matrixMultiply(const Matrix& A, const Matrix& B) {
    size_t n = A.size();
    size_t m = B[0].size();
    size_t k = B.size();

    Matrix result(n, std::vector<double>(m, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            for (size_t l = 0; l < k; ++l) {
                sum += A[i][l] * B[l][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}
