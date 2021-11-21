#include "MultiThreadSolver.hpp"

#include <omp.h>
#include <mpi.h>

#define x(i) (ctx->domain.x1 + i*(ctx->h1))
#define y(i) (ctx->domain.y1 + i*(ctx->h2))
#define d_rx(w) (w[i + 1][j] - w[i][j])/(ctx->h1)
#define d_ry(w) (w[i][j + 1] - w[i][j])/(ctx->h2)
#define d_lx(w) (w[i][j] - w[i - 1][j])/(ctx->h1)
#define d_ly(w) (w[i][j] - w[i][j - 1])/(ctx->h2)
#define dx(w, k) (k(x(i) + 0.5*(ctx->h1), y(j))*d_rx(w) - k(x(i) - 0.5*(ctx->h1), y(j))*d_lx(w))/(ctx->h1)
#define dy(w, k) (k(x(i), y(j) + 0.5*(ctx->h2))*d_ry(w) - k(x(i), y(j) - 0.5*(ctx->h2))*d_ly(w))/(ctx->h2)

void MultiThreadSolver::solve(double error)
{

    double stepDiff = 0.0;
    do {
        #pragma omp parallel for
        for (int i = range.x1; i <= range.x2; ++i) {
            for (int j = range.y1; j <= range.y2; ++j) {
                ctx->buf[i][j] = ctx->w[i][j];
                ctx->curF[i][j] = -dx(ctx->w, ctx->k) - dy(ctx->w, ctx->k) + ctx->q(x(i), y(j))*(ctx->w[i][j]);
                ctx->r[i][j] = ctx->curF[i][j] - ctx->B[i][j];
            }        
        }

        sendNodes(ctx->r, coord, range, pGrigSize, sendBuf);
        receiveNodes(ctx->r, coord, range, pGrigSize, recvBuf);

        double squaredNorm = 0.0;
        double tau = 0.0;
        #pragma omp parallel for reduction(+:squaredNorm, tau)
        for (int i = range.x1; i <= range.x2; ++i) {
            for (int j = range.y1; j <= range.y2; ++j) {
                ctx->Ar[i][j] = -dx(ctx->r, ctx->k) - dy(ctx->r, ctx->k) + ctx->q(x(i), y(j))*(ctx->r[i][j]);
                squaredNorm += (ctx->h1)*(ctx->h2)*(ctx->Ar[i][j])*(ctx->Ar[i][j]);
                tau += (ctx->h1)*(ctx->h2)*(ctx->Ar[i][j])*(ctx->r[i][j]);
            }        
        }

        MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &tau, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau /= squaredNorm;

        stepDiff = 0.0;
        #pragma omp parallel for reduction(+:stepDiff)
        for (int i = range.x1; i <= range.x2; ++i) {
            for (int j = range.y1; j <= range.y2; ++j) {
                ctx->w[i][j] -= tau*(ctx->r[i][j]);
                ctx->buf[i][j] = (ctx->w[i][j]) - (ctx->buf[i][j]);
                stepDiff += (ctx->h1)*(ctx->h2)*(ctx->buf[i][j])*(ctx->buf[i][j]);
            }        
        }

        sendNodes(ctx->w, coord, range, pGrigSize, sendBuf);
        receiveNodes(ctx->w, coord, range, pGrigSize, recvBuf);
        MPI_Allreduce(MPI_IN_PLACE, &stepDiff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } while (sqrt(stepDiff) > error);

}

double MultiThreadSolver::getError(double (*u)(double, double))
{

    double **U = new double*[ctx->M + 1];
    for (int i = 0; i < ctx->M + 1; ++i) {
        U[i] = new double[ctx->N + 1];
    }

    for (int j = ctx->N; j >= 0; --j)
        for (int i = 0; i < ctx->M + 1; ++i)
            U[i][j] = u(ctx->domain.x1 + i*ctx->h1, ctx->domain.y1 + j*ctx->h2);

    double squaredNorm = 0.0;
    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            U[i][j] -= ctx->w[i][j];
            squaredNorm += (ctx->h1)*(ctx->h2)*U[i][j]*U[i][j];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < ctx->M + 1; ++i) {
        delete[] U[i];
    }
    delete[] U;

    return sqrt(squaredNorm);

}