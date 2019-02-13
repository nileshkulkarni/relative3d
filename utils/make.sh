#!/usr/bin/env bash

#CUDA_PATH=/usr/local/cuda/
#CUDA_PATH=/usr/local/cuda-8.0/
#CUDA_PATH=/opt/cuda/8.0/
export CPATH=/opt/cuda/8.0/include/

cd roi_pooling/src

echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_60

#g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
#	roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
cd ../
python build.py
