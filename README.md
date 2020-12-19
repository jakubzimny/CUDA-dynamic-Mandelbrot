# CUDA-dynamic-Mandelbrot

## Overview
Application generating Mandelbrot set visualization using CUDA dynamic parallelism. This project was based on the idea presented in [this NVIDIA Developer blog article](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/) and their solution is probably better and more optimized one.

This repository contains a regular CUDA implementation of Mandelbrot set visualization (mandel.cu file) and one with dynamic parallelism implemented (dynamic_mandel.cu file) and its main purpose was to compare performance improvements between those two versions. 

## Compilation
To compile and run those programs you need to have a suitable NVIDIA GPU and CUDA development enviroment configured.

Regular version:
`nvcc mandel.cu -lpng -o mandel`

Dynamic version (keep in mind that you need to have a GPU with Compute Capability 3.5 or higher): `nvcc -lpng dynamic_mandel.cu -o dyn_mandel -arch=sm_35 -rdc=true -lcudadevrt` or use `dyn_compile.sh` script.

## Runtime parameters
Available parameters (all are optional):
  * w \<int> - specifies width of the final image in pixels, default - 1024
  * h \<int> - specifies height of the final image in pixels, default - 1024
  * iter \<int> - specifies number of iterations that is used to decide whether point is inside Mandelbrot set or not (basically the bigger the value, the more precise image is), default - 1000
  * o \<string> - name of output file, default - Mandelbrot.png 
