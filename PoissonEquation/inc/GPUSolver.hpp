#ifndef GPUSOLVER
#define GPUSOLVER

#include "ParallelSolver.hpp"

class GPUSolver: public ParallelSolver
{

public:
    GPUSolver(Context *ctx, int rank, int size): ParallelSolver(ctx, NULL, rank, size) {}
    virtual void solve(double error);
    virtual double getError(double (*u)(double, double));

private:
    void modifyState();
    void restoreState();

    IndexRange realRange;
    int realM, realN;
    Domain realDomain;

};

#endif