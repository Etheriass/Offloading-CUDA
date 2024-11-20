# CS546 Project - Enlarging Effective GPU Memory for AI Tensors

As LLM are becoming very large, it is important to have a way to store them in the GPU memory. This project aims to implement a way to store large tensors in the GPU memory by using the host memory to store the data that doesn't fit in the GPU memory.

We simulate a LLM by a set of matrix representing the layers of the model. 

Each matrix is dynamically allocated in the host memory and the data is copied to the GPU memory when needed.

## Execution

To run the program use:
```bash 
./main <filename> <number of layers>
``` 
Example wit an LLM of 2 layers: `./main matrix.bin 2`

## Compile 
To compile directly use:
```bash
 nvcc -o main main.cu OffLayer.cu 
```

## CMAKE

Need boost:
```
sudo apt-get install libboost-all-dev
```

```bash
mkdir build
cd build
cmake ../
make -j8
```

Test (doesn't work):
```bash
ctest -VV
```