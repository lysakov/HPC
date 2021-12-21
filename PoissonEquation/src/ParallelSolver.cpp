#include "ParallelSolver.hpp"

#include <stdexcept>
#include <iostream>
#include <cstring>

#include <mpi.h>
#include <omp.h>

#include "macros.hpp"

enum NodesType {
    TOP_NODES,
    BOTTOM_NODES,
    LEFT_NODES,
    RIGHT_NODES
};

static int log2(int n)
{

    int x = 1;
    int pow = 0;
    while (x != n) {
        x *= 2;
        ++pow;
        if (pow > 64) {
            throw std::runtime_error("Processor number should be a power of 2 less then 64");
        }
    }

    return pow;

}

int myPow(int x, int p)
{

    if (p == 0) 
        return 1;
    if (p == 1) 
        return x;
  
    int tmp = myPow(x, p/2);
    if (p % 2 == 0) 
        return tmp*tmp;
    else 
        return x*tmp*tmp;

}

double ParallelAlgebra::dot(double *u, double *v, const IndexRange &range) 
{

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            sum += h1*h2*u[i*(N + 1) + j]*v[i*(N + 1) + j];
        }
    }

    return sum;

}

void ParallelAlgebra::mult(double *r, double alpha, double *u, const IndexRange &range)
{

    #pragma omp parallel for
    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i*(N + 1) + j] = alpha*u[i*(N + 1) + j];
        }
    }

}

void ParallelAlgebra::subs(double *r, double *u, double *v, const IndexRange &range)
{

    #pragma omp parallel for
    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i*(N + 1) + j] = u[i*(N + 1) + j] - v[i*(N + 1) + j];
        }
    }

}

void ParallelAlgebra::A(double *r, double *u, const IndexRange &range,
    double (*k)(double, double), double (*q)(double, double))
{

    #pragma omp parallel for
    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i*(N + 1) + j] = -dx(u, k) - dy(u, k) + q(x(i), y(j))*u[i*(N + 1) + j];
        }        
    }

}

std::pair<int, int> ParallelSolver::getProcessorGridSize(int size)
{

    if (size == 2) {
        return std::pair<int, int>(1, 2);
    }

    int n = log2(size);
    int n_x = myPow(2, n/2);
    int n_y = myPow(2, n % 2 ? n/2 + 1 : n/2);

    return std::pair<int, int>(n_x, n_y);

}

ParallelSolver::ProcessorCoordinates ParallelSolver::toCoordinates
    (int rank, std::pair<int, int> pGridSize)
{

    int x = rank % pGridSize.first;
    int y = rank / pGridSize.first;

    return ParallelSolver::ProcessorCoordinates(x, y);

}


void ParallelSolver::splitGrid(int rank, int size, IndexRange &range)
{

    std::pair<int, int> gridSize = getProcessorGridSize(size);
    int n_x = gridSize.first;
    int n_y = gridSize.second;

    if (n_x >= ctx->M - 1) {
        range.x1 = rank > ctx->M - 1 ? ctx->M - 1 : rank;
        range.x2 = range.x1;
    }
    else {
        int k_x = (ctx->M - 1) / n_x;
        int r_x = (ctx->M - 1) % n_x;
        int rank_x = rank % n_x;
        range.x1 = 1 + (rank_x < r_x ? (k_x + 1)*rank_x : (k_x + 1)*r_x + k_x*(rank_x - r_x));
        range.x2 = (rank_x < r_x ? range.x1 + k_x : range.x1 + k_x - 1);
    }

    if (n_y >= ctx->N - 1) {
        range.y1 = rank > ctx->N - 1? ctx->N - 1: rank;
        range.y2 = range.y1;
    }
    else {
        int k_y = (ctx->N - 1) / n_y;
        int r_y = (ctx->N - 1) % n_y;
        int rank_y = rank / n_x;
        range.y1 = 1 + (rank_y < r_y ? (k_y + 1)*rank_y : (k_y + 1)*r_y + k_y*(rank_y - r_y));
        range.y2 = (rank_y < r_y ? range.y1 + k_y : range.y1 + k_y - 1);
    }

}

void ParallelSolver::sendNodes(double *w, const ProcessorCoordinates &coord, 
    const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf)
{

    if (coord.x != 0) {
        MPI_Send(w + range.x1*(ctx->N + 1) + range.y1, 
            range.y2 - range.y1 + 1, MPI_DOUBLE, coord.y*pGridSize.first + coord.x - 1, 
            RIGHT_NODES, MPI_COMM_WORLD);
    }
    if (coord.x != pGridSize.first - 1) {
        MPI_Send(w + range.x2*(ctx->N + 1) + range.y1, 
            range.y2 - range.y1 + 1, MPI_DOUBLE, coord.y*pGridSize.first + coord.x + 1, 
            LEFT_NODES, MPI_COMM_WORLD);
    }
    if (coord.y != 0) {
        for (int i = 0; i <= range.x2 - range.x1; ++i) {
            buf[i] = w[(range.x1 + i)*(ctx->N + 1) + range.y1];
        }
        MPI_Send(buf, range.x2 - range.x1 + 1, MPI_DOUBLE, (coord.y - 1)*pGridSize.first + coord.x, TOP_NODES, MPI_COMM_WORLD);
    }
    if (coord.y != pGridSize.second - 1) {
        for (int i = 0; i <= range.x2 - range.x1; ++i) {
            buf[i] = w[(range.x1 + i)*(ctx->N + 1) + range.y2];
        }
        MPI_Send(buf, range.x2 - range.x1 + 1, MPI_DOUBLE, (coord.y + 1)*pGridSize.first + coord.x, BOTTOM_NODES, MPI_COMM_WORLD);
    }

}

void ParallelSolver::receiveNodes(double *w, const ProcessorCoordinates &coord, 
    const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf)
{

    int msgToReceive = 4;
    if (coord.x == 0 || coord.x == pGridSize.first - 1)
        msgToReceive--;
    if (coord.y == 0 || coord.y == pGridSize.second - 1)
        msgToReceive--;
    if (size == 2)
        msgToReceive = 1;

    int maxMsgLen = range.x2 - range.x1 + 1 > range.y2 - range.y1 + 1 ? range.x2 - range.x1 + 1 : range.y2 - range.y1 + 1;

    for (int i = 0; i < msgToReceive; ++i) {
        MPI_Status status;
        MPI_Recv(buf, maxMsgLen, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int count = 0;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        
        switch (static_cast<NodesType>(status.MPI_TAG)) {
            case LEFT_NODES:
                std::memcpy(w + (range.x1 - 1)*(ctx->N + 1) + range.y1, buf, count*sizeof(double));
                break;
            case RIGHT_NODES:
                std::memcpy(w + (range.x2 + 1)*(ctx->N + 1) + range.y1, buf, count*sizeof(double));
                break;
            case BOTTOM_NODES:
                for (int i = 0; i < count; ++i) {
                    w[(range.x1 + i)*(ctx->N + 1) + range.y1 - 1] = buf[i];
                }
                break;
            case TOP_NODES:
                for (int i = 0; i < count; ++i) {
                    w[(range.x1 + i)*(ctx->N + 1) + range.y2 + 1] = buf[i];
                }
                break;
            default:
                break;
        }
    }

}

void ParallelSolver::solve(double error)
{

    double squaredNorm = 0.0;
    do {
        copyW();
        step(range);
        engine->subs(ctx->buf, ctx->w, ctx->buf, range);
        squaredNorm = engine->dot(ctx->buf, ctx->buf, range);
        MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } while(sqrt(squaredNorm) > error);

}

void ParallelSolver::step(const IndexRange &range)
{

    engine->A(ctx->Aw, ctx->w, range, ctx->k, ctx->q);
    engine->subs(ctx->r, ctx->Aw, ctx->B, range);
    sendNodes(ctx->r, coord, range, pGrigSize, sendBuf);
    receiveNodes(ctx->r, coord, range, pGrigSize, recvBuf);

    engine->A(ctx->Ar, ctx->r, range, ctx->k, ctx->q);
    double squaredNorm = engine->dot(ctx->Ar, ctx->Ar, range);
    MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double tau = engine->dot(ctx->Ar, ctx->r, range) / squaredNorm;
    MPI_Allreduce(MPI_IN_PLACE, &tau, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    engine->mult(ctx->r, tau, ctx->r, range);
    engine->subs(ctx->w, ctx->w, ctx->r, range);
    sendNodes(ctx->w, coord, range, pGrigSize, sendBuf);
    receiveNodes(ctx->w, coord, range, pGrigSize, recvBuf);

}

double ParallelSolver::getError(double (*u)(double, double))
{

    double *U = new double[(ctx->M + 1)*(ctx->N + 1)];

    for (int i = 0; i <= ctx->M; ++i)
        for (int j = 0; j <= ctx->N; ++j)
            U[i*(ctx->N + 1) + j] = u(ctx->domain.x1 + i*ctx->h1, ctx->domain.y1 + j*ctx->h2);

    engine->subs(U, U, ctx->w, range);
    double norm = engine->dot(U, U, range);
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] U;

    return sqrt(norm);

}

double* ParallelSolver::getSolution()
{

    ctx->finalize();
    for (int i = 0; i < size; ++i) {
        IndexRange r;
        if (i == rank) {
            r = this->range; 
        }
        MPI_Bcast(&r, 4, MPI_INTEGER, i, MPI_COMM_WORLD);
        for (int j = 0; j <= r.x2 - r.x1; ++j) {
            MPI_Bcast(ctx->w + (r.x1 + j)*(ctx->N + 1) + r.y1, r.y2 - r.y1 + 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
        }
    }

    return ctx->w;

}


void ParallelSolver::copyW()
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            ctx->buf[i*(ctx->N + 1) + j] = ctx->w[i*(ctx->N + 1) + j];
        }
    }

}
