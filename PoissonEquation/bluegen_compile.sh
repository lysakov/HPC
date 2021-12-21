#!/bin/sh
mpixlcxx_r src/main.cpp src/SerialSolver.cpp src/ParallelSolver.cpp src/MultiThreadSolver.cpp src/Interactor.cpp src/GPUSolverMock.cpp -Iinc -o PoissonEquation -O5 -qsmp=omp