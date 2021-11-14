#include "ParallelSolver.hpp"

#include <stdexcept>
#include <iostream>
#include <cstring>

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

std::pair<int, int> ParallelSolver::getProcessorGridSize(int size)
{

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

void ParallelSolver::sendNodes(const double **w, const ProcessorCoordinates &coord, 
    const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf)
{

    if (coord.x != 0) {
        MPI_Send(w[range.x1] + range.y1, range.y2 - range.y1 + 1, MPI_DOUBLE, coord.y*pGridSize.first + coord.x - 1, RIGHT_NODES, MPI_COMM_WORLD);
    }
    if (coord.x != pGridSize.first - 1) {
        MPI_Send(w[range.x2] + range.y1, range.y2 - range.y1 + 1, MPI_DOUBLE, coord.y*pGridSize.first + coord.x + 1, LEFT_NODES, MPI_COMM_WORLD);
    }
    if (coord.y != 0) {
        for (int i = 0; i <= range.x2 - range.x1; ++i) {
            buf[i] = w[range.x1 + i][range.y1];
        }
        MPI_Send(buf, range.x2 - range.x1 + 1, MPI_DOUBLE, (coord.y - 1)*pGridSize.first + coord.x, TOP_NODES, MPI_COMM_WORLD);
    }
    if (coord.y != pGridSize.second - 1) {
        for (int i = 0; i <= range.x2 - range.x1; ++i) {
            buf[i] = w[range.x1 + i][range.y2];
        }
        MPI_Send(buf, range.x2 - range.x1 + 1, MPI_DOUBLE, (coord.y + 1)*pGridSize.first + coord.x, BOTTOM_NODES, MPI_COMM_WORLD);
    }

}

void ParallelSolver::receiveNodes(double **w, const ProcessorCoordinates &coord, 
    const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf)
{

    int msgToReceive = 4;
    if (coord.x == 0 || coord.x == pGridSize.first - 1)
        msgToReceive--;
    if (coord.y == 0 || coord.y == pGridSize.second - 1)
        msgToReceive--;
    //std::cout << msgToReceive << "\n";
    int maxMsgLen = range.x2 - range.x1 + 1 > range.y2 - range.y1 + 1 ? range.x2 - range.x1 + 1 : range.y2 - range.y1 + 1;

    for (int i = 0; i < msgToReceive; ++i) {
        MPI_Status status;
        MPI_Recv(buf, maxMsgLen, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int count = 0;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        //if (rank == 12)
        //std::cout << rank << "-" << coord.x << " " << coord.y << "-" << count  << " | " << status.MPI_SOURCE << " " << status.MPI_TAG << '\n';

        switch (static_cast<NodesType>(status.MPI_TAG)) {
            case LEFT_NODES:
                std::memcpy(w[range.x1 - 1] + range.y1, buf, count*sizeof(double));
                break;
            case RIGHT_NODES:
                std::memcpy(w[range.x2 + 1] + range.y1, buf, count*sizeof(double));
                break;
            case BOTTOM_NODES:
                for (int i = 0; i < count; ++i) {
                    w[range.x1 + i][range.y1 - 1] = buf[i];
                }
                break;
            case TOP_NODES:
                for (int i = 0; i < count; ++i) {
                    w[range.x1 + i][range.y2 + 1] = buf[i];
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
        //std::cout << "1) " << locSquaredNorm << "\n";
        MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //std::cout << "2) " << squaredNorm << "\n";
    } while(sqrt(squaredNorm) > error);

}

void ParallelSolver::step(const IndexRange &range)
{

    engine->A(ctx->curF, ctx->w, range, ctx->k, ctx->q);
    engine->subs(ctx->r, ctx->curF, ctx->B, range);
    sendNodes(const_cast<const double**>(ctx->r), coord, range, pGrigSize, sendBuf);
    receiveNodes(ctx->r, coord, range, pGrigSize, recvBuf);

    engine->A(ctx->Ar, ctx->r, range, ctx->k, ctx->q);
    double squaredNorm = engine->dot(ctx->Ar, ctx->Ar, range);
    MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double tau = engine->dot(ctx->Ar, ctx->r, range) / squaredNorm;
    MPI_Allreduce(MPI_IN_PLACE, &tau, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    engine->mult(ctx->r, tau, ctx->r, range);
    engine->subs(ctx->w, ctx->w, ctx->r, range);
    sendNodes(const_cast<const double**>(ctx->w), coord, range, pGrigSize, sendBuf);
    receiveNodes(ctx->w, coord, range, pGrigSize, recvBuf);

}

double ParallelSolver::getError(const Domain &domain, double (*u)(double, double))
{

    double **U = new double*[ctx->M + 1];
    for (int i = 0; i < ctx->M + 1; ++i) {
        U[i] = new double[ctx->N + 1];
    }

    double h1 = (domain.x2 - domain.x1) / ctx->M;
    double h2 = (domain.y2 - domain.y1) / ctx->N;
    for (int j = ctx->N; j >= 0; --j)
        for (int i = 0; i < ctx->M + 1; ++i)
            U[i][j] = u(domain.x1 + i*h1, domain.y1 + j*h2);

    engine->subs(U, U, ctx->w, range);
    double norm = engine->dot(U, U, range);
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < ctx->M + 1; ++i) {
        delete[] U[i];
    }
    delete[] U;

    return sqrt(norm);

}

void ParallelSolver::copyW()
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            ctx->buf[i][j] = ctx->w[i][j];
        }
    }

}