#ifndef OFFLAYER_CUH
#define OFFLAYER_CUH

#include <iostream>
#include <fstream>
#include <vector>

/* OffLayer class 
*  This class is used to load layers of a simulated LLM reduce to a set of matrices
*  The matrices are stored in a binary file
*/
class OffLayer {
private:
    size_t total_layers;
    std::vector<size_t> layer_rows;
    std::vector<size_t> layer_cols;
    std::ifstream file;
    float* device_matrix;
    size_t current_layer;

public:
    OffLayer(const std::string& filename, size_t total_layers, const std::vector<size_t>& layer_rows, const std::vector<size_t>& layer_cols);

    ~OffLayer();

    float* getLayer(size_t layer_index);

    void loadLayer(size_t layer_index);

    size_t getLayerRows(size_t layer_index);
    size_t getLayerCols(size_t layer_index);

    size_t getMaxLayerSize();
};

#endif