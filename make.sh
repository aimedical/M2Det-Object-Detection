#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda-10.1/

python build.py build_ext --inplace

cd ..
