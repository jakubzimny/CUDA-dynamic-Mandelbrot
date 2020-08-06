nvcc -lpng dynamic_mandel.cu -o dyn_mandel -arch=sm_35 -rdc=true -lcudadevrt
