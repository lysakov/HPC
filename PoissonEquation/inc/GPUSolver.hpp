#ifndef GPUSOLVER
#define GPUSOLVER

#include "ParallelSolver.hpp"

#include <cstdlib>
#include <cuda_runtime.h>

class GPUSolver: public ParallelSolver
{

public:
    GPUSolver(Context *ctx, int rank, int size): ParallelSolver(ctx, NULL, rank, size) 
    {
        d_w = NULL;
        d_B = NULL;
        d_Aw = NULL;
        d_r = NULL;
        d_Ar = NULL;
        d_buf = NULL;
        d_dotBuf1 = NULL;
        d_dotBuf2 = NULL;
        d_topNodes = NULL;
        d_bottomNodes = NULL;
        d_rightNodes = NULL;
        d_leftNodes = NULL;
    }
    virtual void solve(double error);
    virtual double getError(double (*u)(double, double));

private:
    void modifyState();
    void restoreState();
    void initDeviceMemory();
    void freeDeviceMemory();
    void sendNodes(double *w, const ProcessorCoordinates &coord, 
        const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf);
    void receiveNodes(double *w, const ProcessorCoordinates &coord, 
        const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf,
        const dim3 &blocksPerGrin, const dim3 &threadsPerBlock);

    IndexRange realRange;
    int realM, realN;
    Domain realDomain;

    double *d_w;
    double *d_B;
    double *d_Aw;
    double *d_r;
    double *d_Ar;
    double *d_buf;
    double *d_dotBuf1;
    double *d_dotBuf2;
    double *d_topNodes;
    double *d_bottomNodes;
    double *d_rightNodes;
    double *d_leftNodes;
    double *d_nodes;
    double *r_topNodes;
    double *r_bottomNodes;
    double *r_rightNodes;
    double *r_leftNodes;

};

#endif
