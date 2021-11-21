#!/bin/sh
mpixlC src/main.cpp src/SerialSolver.cpp src/ParallelSolver.cpp src/MultiThreadSolver.cpp src/Interactor.cpp -Iinc -o PoissonEquation -O5 -qsmp=omp
