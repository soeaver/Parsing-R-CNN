#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling pool points interp kernels by nvcc..."
nvcc -c -o pool_points_interp_kernel.cu.o pool_points_interp_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python3 build.py
