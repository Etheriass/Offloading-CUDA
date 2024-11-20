#include "OffLayer.cuh"
#include <iostream>
#include <fstream>
#include <vector>

OffLayer::OffLayer(const std::string &filename, size_t total_layers, const std::vector<size_t> &layer_rows, const std::vector<size_t> &layer_cols)
    : total_layers(total_layers), layer_rows(layer_rows), layer_cols(layer_cols), current_layer(1)
{
    file.open(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    cudaMalloc(&device_matrix, getMaxLayerSize() * sizeof(float));
}

OffLayer::~OffLayer()
{
    cudaFree(device_matrix);
    file.close();
}

float* OffLayer::getLayer(size_t layer_index)
{
    if (layer_index != current_layer)
    {
        loadLayer(layer_index);
        current_layer = layer_index;
    }
    return device_matrix;
}

void OffLayer::loadLayer(size_t layer_index)
{
    size_t offset = 0;
    for (size_t i = 0; i < layer_index; ++i)
    {
        offset += layer_rows[i] * layer_cols[i] * sizeof(float);
    }
    file.seekg(offset, std::ios::beg);
    if (file.fail())
    {
        throw std::runtime_error("Failed to seek to position in file");
    }
    size_t elements_to_read = layer_rows[layer_index] * layer_cols[layer_index];
    std::vector<float> host_matrix(elements_to_read);
    file.read(reinterpret_cast<char *>(host_matrix.data()), elements_to_read * sizeof(float));
    cudaMemcpy(device_matrix, host_matrix.data(), elements_to_read * sizeof(float), cudaMemcpyHostToDevice);
}

size_t OffLayer::getLayerRows(size_t layer_index)
{
    return layer_rows[layer_index];
}

size_t OffLayer::getLayerCols(size_t layer_index)
{
    return layer_cols[layer_index];
}

size_t OffLayer::getMaxLayerSize()
{
    size_t max_size = 0;
    for (size_t i = 0; i < total_layers; ++i)
    {
        size_t size = layer_rows[i] * layer_cols[i];
        if (size > max_size)
        {
            max_size = size;
        }
    }
    return max_size;
}