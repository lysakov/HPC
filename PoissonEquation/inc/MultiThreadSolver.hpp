#ifndef MULTITHREADSOLVER
#define MULTITHREADSOLVER

#include "ISolver.hpp"
#include "ParallelSolver.hpp"

#include <cstdlib>

class MultiThreadSolver : public ParallelSolver
{

public:
    MultiThreadSolver(Context *ctx, int rank, int size) : 
        ParallelSolver(ctx, NULL, rank, size) {}
    virtual void solve(double error);
    virtual double getError(double (*u)(double, double));

};

#endif