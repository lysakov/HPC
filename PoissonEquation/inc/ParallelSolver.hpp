#ifndef PARALLEL_SOLVER
#define PARALLEL_SOLVER

#include "ISolver.hpp"

#include <utility>

class ParallelAlgebra : public AbstractLinearAlgebra
{

public:
    ParallelAlgebra(int M, int N, Domain domain) : M(M), N(N), domain(domain)
    {
        h1 = (domain.x2 - domain.x1) / M;
        h2 = (domain.y2 - domain.y1) / N;
    }
    virtual double dot(double *u, double *v, const IndexRange &range);
    virtual void mult(double *r, double alpha, double *u, const IndexRange &range);
    virtual void subs(double *r, double *u, double *v, const IndexRange &range);
    virtual void A(double *r, double *u, const IndexRange &range,
        double (*k)(double, double), double (*q)(double, double));

private:
    int M, N;
    double h1, h2;
    Domain domain;

};

class ParallelSolver: public ISolver
{

public:
    ParallelSolver(Context *ctx, AbstractLinearAlgebra *engine, int rank, int size) : 
        ctx(ctx), engine(engine), rank(rank), size(size) 
    {
        splitGrid(rank, size, range);
        this->range = range;
        sendBuf = new double[range.x2 - range.x1 + 1];
        recvBuf = new double[range.x2 - range.x1 + 1 > range.y2 - range.y1 + 1 ? range.x2 - range.x1 + 1 : range.y2 - range.y1 + 1];
        pGrigSize = getProcessorGridSize(size);
        coord = toCoordinates(rank, pGrigSize);
    }
    virtual void solve(double error);
    virtual double* getSolution();
    void step(const IndexRange &range);
    void splitGrid(int rank, int size, IndexRange &range);
    virtual double getError(double (*u)(double, double));
    ~ParallelSolver()
    {
        delete[] sendBuf;
        delete[] recvBuf;
    }

protected:
    struct ProcessorCoordinates
    {
        int x;
        int y;
        ProcessorCoordinates(int x = 0, int y = 0) : x(x), y(y) {}
    };
    
    std::pair<int, int> getProcessorGridSize(int size);
    ProcessorCoordinates toCoordinates(int rank, std::pair<int, int> pGridSize);
    void sendNodes(double *w, const ProcessorCoordinates &coord, 
        const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf);
    void receiveNodes(double *w, const ProcessorCoordinates &coord, 
        const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf);
    void copyW();

    Context *ctx;
    AbstractLinearAlgebra *engine;
    int rank, size;
    IndexRange range;
    double *sendBuf;
    double *recvBuf;
    std::pair<int, int> pGrigSize;
    ProcessorCoordinates coord;


};

#endif