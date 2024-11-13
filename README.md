# CS546_Project
Enlarging Effective GPU Memory for AI Tensors


## Compile 
To compile directly use:
```bash
vcc off_vector.cu -o main main.cu -lcudart -rdc=true 
```
- -lcudart: link the CUDA runtime libraries
- -rdc=true: enable relocatable device code

## CMAKE

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