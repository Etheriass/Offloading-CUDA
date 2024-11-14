# CS546_Project
Enlarging Effective GPU Memory for AI Tensors


## Compile 
To compile directly use:
```bash
nvcc off_vector.cu -o main main.cu -rdc=true 
```
- -rdc=true: enable relocatable device code

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