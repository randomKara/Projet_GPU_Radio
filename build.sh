#!/bin/bash

CUDA_PATH=/usr/local/cuda-13.0
export CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH

rm -rf build
mkdir build
cd build

cmake -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
      -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
      ..

make

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo "To run with satellite data, use: ./process_qb50_segments.sh"
    echo ""
else
    echo "Build failed!"
    exit 1
fi
