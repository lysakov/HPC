#!/bin/sh
nvcc -Iinc -O5 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -Xptxas -dlcm=cg -c src/GPUSolver.cu
mpixlC -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ src/main.cpp src/SerialSolver.cpp src/ParallelSolver.cpp src/MultiThreadSolver.cpp src/Interactor.cpp GPUSolver.o -lcudart -Iinc -o PoissonEquation -O5 -qsmp=omp -std=c++11
rm -f *.o
