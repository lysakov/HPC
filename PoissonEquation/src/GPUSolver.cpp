#include "GPUSolver.hpp"

#include <cstring>
#include <mpi.h>

#define x(i) (ctx->domain.x1 + (i)*(ctx->h1))
#define y(i) (ctx->domain.y1 + (i)*(ctx->h2))
#define d_rx(w) (w[(i + 1)*(ctx->N + 1) + j] - w[i*(ctx->N + 1) + j])/(ctx->h1)
#define d_ry(w) (w[i*(ctx->N + 1) + j + 1] - w[i*(ctx->N + 1) + j])/(ctx->h2)
#define d_lx(w) (w[i*(ctx->N + 1) + j] - w[(i - 1)*(ctx->N + 1) + j])/(ctx->h1)
#define d_ly(w) (w[i*(ctx->N + 1) + j] - w[i*(ctx->N + 1) + j - 1])/(ctx->h2)
#define dx(w, k) (k(x(i) + 0.5*(ctx->h1), y(j))*d_rx(w) - k(x(i) - 0.5*(ctx->h1), y(j))*d_lx(w))/(ctx->h1)
#define dy(w, k) (k(x(i), y(j) + 0.5*(ctx->h2))*d_ry(w) - k(x(i), y(j) - 0.5*(ctx->h2))*d_ly(w))/(ctx->h2)

void GPUSolver::solve(double error)
{

    modifyState();
    
    double stepDiff = 0.0;
    do {
        for (int i = range.x1; i <= range.x2; ++i) {
            for (int j = range.y1; j <= range.y2; ++j) {
                int ind = i*(ctx->N + 1) + j;
                ctx->buf[ind] = ctx->w[ind];
                ctx->Aw[ind] = -dx(ctx->w, ctx->k) - dy(ctx->w, ctx->k) + ctx->q(x(i), y(j))*(ctx->w[ind]);
                ctx->r[ind] = ctx->Aw[ind] - ctx->B[ind];
            }        
        }

        sendNodes(ctx->r, coord, range, pGrigSize, sendBuf);
        receiveNodes(ctx->r, coord, range, pGrigSize, recvBuf);

        double squaredNorm = 0.0;
        double tau = 0.0;
        for (int i = range.x1; i <= range.x2; ++i) {
            for (int j = range.y1; j <= range.y2; ++j) {
                int ind = i*(ctx->N + 1) + j;
                ctx->Ar[ind] = -dx(ctx->r, ctx->k) - dy(ctx->r, ctx->k) + ctx->q(x(i), y(j))*(ctx->r[ind]);
                squaredNorm += (ctx->h1)*(ctx->h2)*(ctx->Ar[ind])*(ctx->Ar[ind]);
                tau += (ctx->h1)*(ctx->h2)*(ctx->Ar[ind])*(ctx->r[ind]);
            }        
        }

        MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &tau, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau /= squaredNorm;

        stepDiff = 0.0;
        for (int i = range.x1; i <= range.x2; ++i) {
            for (int j = range.y1; j <= range.y2; ++j) {
                int ind = i*(ctx->N + 1) + j;
                ctx->w[ind] -= tau*(ctx->r[ind]);
                ctx->buf[ind] = (ctx->w[ind]) - (ctx->buf[ind]);
                stepDiff += (ctx->h1)*(ctx->h2)*(ctx->buf[ind])*(ctx->buf[ind]);
            }        
        }

        sendNodes(ctx->w, coord, range, pGrigSize, sendBuf);
        receiveNodes(ctx->w, coord, range, pGrigSize, recvBuf);
        MPI_Allreduce(MPI_IN_PLACE, &stepDiff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } while (sqrt(stepDiff) > error);

    restoreState();

}

double GPUSolver::getError(double (*u)(double, double))
{

    double *U = new double[(ctx->M + 1)*(ctx->N + 1)];

    for (int i = 0; i <= ctx->M; ++i)
        for (int j = 0; j <= ctx->N; ++j)
            U[i*(ctx->N + 1) + j] = u(ctx->domain.x1 + i*ctx->h1, ctx->domain.y1 + j*ctx->h2);

    double squaredNorm = 0.0;
    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            U[i*(ctx->N + 1) + j] -= ctx->w[i*(ctx->N + 1) + j];
            squaredNorm += (ctx->h1)*(ctx->h2)*U[i*(ctx->N + 1) + j]*U[i*(ctx->N + 1) + j];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] U;

    return sqrt(squaredNorm);

}

void GPUSolver::modifyState()
{

    realM = ctx->M;
    realN = ctx->N;
    realRange = range;
    realDomain = ctx->domain;

    ctx->domain.x1 = x(range.x1 - 1);
    ctx->domain.y1 = y(range.y1 - 1);
    ctx->M = range.x2 - range.x1 + 2;
    ctx->N = range.y2 - range.y1 + 2;
    range = IndexRange(1, ctx->M - 1, 1, ctx->N - 1);

    double *buf = new double[(ctx->M + 1)*(ctx->N + 1)];
    for (int i = 0; i <= ctx->M; ++i) {
        std::memcpy(buf + i*(ctx->N + 1), ctx->B + (i + realRange.x1 - 1)*(realN + 1) + realRange.y1 - 1, (ctx->N + 1)*sizeof(double));
    }
    std::memcpy(ctx->B, buf, sizeof(double)*(ctx->M + 1)*(ctx->N + 1));
    delete[] buf;

}

void GPUSolver::restoreState()
{

    double *buf = new double[(ctx->M + 1)*(ctx->N + 1)];
    std::memcpy(buf, ctx->w, sizeof(double)*(ctx->M + 1)*(ctx->N + 1));
    for (int i = 0; i <= ctx->M; ++i) {
        std::memcpy(ctx->w + (i + realRange.x1 - 1)*(realN + 1) + realRange.y1 - 1, buf + i*(ctx->N + 1), (ctx->N + 1)*sizeof(double));
    }
    range = realRange;
    ctx->M = realM;
    ctx->N = realN;
    ctx->domain = realDomain;
    delete[] buf;

}