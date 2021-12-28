#include "GPUSolver.hpp"

#include <cstring>
#include <mpi.h>
#include <iostream>

#include <deviceFunctions.hpp>

#define x(i) (x1 + i*h1)
#define y(i) (y1 + i*h2)
#define d_rx(w) (w[(i + 1)*(N + 1) + j] - w[i*(N + 1) + j])/(h1)
#define d_ry(w) (w[i*(N + 1) + j + 1] - w[i*(N + 1) + j])/(h2)
#define d_lx(w) (w[i*(N + 1) + j] - w[(i - 1)*(N + 1) + j])/(h1)
#define d_ly(w) (w[i*(N + 1) + j] - w[i*(N + 1) + j - 1])/(h2)
#define dx(w, k) (k(x(i) + 0.5*(h1), y(j))*d_rx(w) - k(x(i) - 0.5*(h1), y(j))*d_lx(w))/(h1)
#define dy(w, k) (k(x(i), y(j) + 0.5*(h2))*d_ry(w) - k(x(i), y(j) - 0.5*(h2))*d_ly(w))/(h2)

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

enum NodesType {
    TOP_NODES,
    BOTTOM_NODES,
    LEFT_NODES,
    RIGHT_NODES
};

__global__ void step_1(int M, int N, double x1, double y1, double h1, double h2, 
    double *w, double *B, double *Aw, double *r, double *buf,
    double *topNodes, double *bottomNodes, double *rightNodes, double *leftNodes)
{

    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    if (i > 0 && i < M && j > 0 && j < N) {
        int ind = i*(N + 1) + j;
        buf[ind] = w[ind];
        Aw[ind] = -dx(w, d_k) - dy(w, d_k) + d_q(x(i), y(j))*(w[ind]);
        r[ind] = Aw[ind] - B[ind];
    }

}

__global__ void step_2(int M, int N, double x1, double y1, double h1, double h2, 
     double *r, double *Ar, double *dotBuf, double *normBuf)
{

    __shared__ double cacheDot[THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y];
    __shared__ double cacheNorm[THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y];

    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int threadIndex = threadIdx.y*blockDim.x + threadIdx.x;
    int blockIndex = blockIdx.y*gridDim.x + blockIdx.x;

    cacheDot[threadIndex] = 0.0;
    cacheNorm[threadIndex] = 0.0;

    if (i > 0 && i < M && j > 0 && j < N) {
        int ind = i*(N + 1) + j;
        Ar[ind] = -dx(r, d_k) - dy(r, d_k) + d_q(x(i), y(j))*(r[ind]);
        cacheNorm[threadIndex] = (h1)*(h2)*(Ar[ind])*(Ar[ind]);
        cacheDot[threadIndex] = (h1)*(h2)*(Ar[ind])*(r[ind]);
    }
    __syncthreads();

    i = THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y/2;
    while (i != 0) {
        if (threadIndex < i) {
            cacheNorm[threadIndex] += cacheNorm[threadIndex + i];
            cacheDot[threadIndex] += cacheDot[threadIndex + i];
        } 
        __syncthreads();
        i /= 2;
    }

    if (threadIndex == 0) {
        normBuf[blockIndex] = cacheNorm[0];
        dotBuf[blockIndex] = cacheDot[0];
    }

}

__global__ void step_3(int M, int N, double x1, double y1, double h1, double h2, 
    double tau, double *w, double *r, double *buf, double *dotBuf,
    double *topNodes, double *bottomNodes, double *rightNodes, double *leftNodes)
{

    __shared__ double cache[THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y];

    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int threadIndex = threadIdx.y*blockDim.x + threadIdx.x;
    int blockIndex = blockIdx.y*gridDim.x + blockIdx.x;

    cache[threadIndex] = 0.0;

    if (i > 0 && i < M && j > 0 && j < N) {
        int ind = i*(N + 1) + j;
        w[ind] -= tau*r[ind];
        buf[ind] = w[ind] - buf[ind];
        cache[threadIndex] = h1*h2*buf[ind]*buf[ind];
    }
    __syncthreads();

    i = THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y/2;
    while (i != 0) {
        if (threadIndex < i) {
            cache[threadIndex] += cache[threadIndex + i];
        } 
        __syncthreads();
        i /= 2;
    }

    if (threadIndex == 0) {
        dotBuf[blockIndex] = cache[0];
    }

}

__global__ void getNodes(int M, int N, double *w,
    double *topNodes, double *bottomNodes, double *rightNodes, double *leftNodes)
{

    int j = blockIdx.x*blockDim.x + threadIdx.x + 1;

    if (j > 0 && j < N) {
        leftNodes[j - 1] = w[N + 1 + j];
        rightNodes[j - 1] = w[(M - 1)*(N + 1) + j];
    }
    if (j > 0 && j < M) {
        bottomNodes[j - 1] = w[j*(N + 1) + 1];
        topNodes[j - 1] = w[j*(N + 1) + N - 1];
    }

}

__global__ void loadNodes(int M, int N, double *dest,
    double *topNodes, double *bottomNodes, double *rightNodes, double *leftNodes)
{

    int j = blockIdx.x*blockDim.x + threadIdx.x + 1;

    if (leftNodes != NULL && j > 0 && j < N) {
        dest[j] = leftNodes[j - 1];
    }
    if (rightNodes != NULL && j > 0 && j < N) {
        dest[M*(N + 1) + j] = rightNodes[j - 1];
    }
    if (bottomNodes != NULL && j > 0 && j < M) {
        dest[j*(N + 1)] = bottomNodes[j - 1];
    }
    if (topNodes != NULL && j > 0 && j < M) {
        dest[j*(N + 1) + N] = topNodes[j - 1];
    }

}

void GPUSolver::solve(double error)
{

    modifyState();
    initDeviceMemory();
    
    double stepDiff = 0.0;
    int blockNum = (ctx->M/32 + 1)*(ctx->N/32 + 1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 blocksPerGrid(ctx->N/32 + 1, ctx->M/32 + 1);

    double *dotBuf1 = new double[blockNum];
    double *dotBuf2 = new double[blockNum];
    delete[] sendBuf;
    sendBuf = new double[2*(ctx->N - 1 + ctx->M - 1)];
    delete[] recvBuf;
    recvBuf = new double[2*(ctx->N - 1 + ctx->M - 1)];
    do {
        step_1<<<blocksPerGrid, threadsPerBlock>>>(ctx->M, ctx->N, ctx->domain.x1, ctx->domain.y1, 
            ctx->h1, ctx->h2, d_w, d_B, d_Aw, d_r, d_buf, d_topNodes, d_bottomNodes, d_rightNodes, d_leftNodes);

        if (size != 1) {
            sendNodes(d_r, coord, range, pGrigSize, sendBuf);
            receiveNodes(d_r, coord, range, pGrigSize, recvBuf, blocksPerGrid, threadsPerBlock);
        }

        double squaredNorm = 0.0;
        double tau = 0.0;
        step_2<<<blocksPerGrid, threadsPerBlock>>>(ctx->M, ctx->N, ctx->domain.x1, ctx->domain.y1, 
            ctx->h1, ctx->h2, d_r, d_Ar, d_dotBuf1, d_dotBuf2);
        cudaMemcpy(dotBuf1, d_dotBuf1, blockNum*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(dotBuf2, d_dotBuf2, blockNum*sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < blockNum; ++i) {
            tau += dotBuf1[i];
            squaredNorm += dotBuf2[i];
        }
        if (size != 1) {
            MPI_Allreduce(MPI_IN_PLACE, &squaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &tau, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        tau /= squaredNorm;
        
        step_3<<<blocksPerGrid, threadsPerBlock>>>(ctx->M, ctx->N, ctx->domain.x1, ctx->domain.y1, 
            ctx->h1, ctx->h2, tau, d_w, d_r, d_buf, d_dotBuf1, d_topNodes, d_bottomNodes, d_rightNodes, d_leftNodes);
        cudaMemcpy(dotBuf1, d_dotBuf1, blockNum*sizeof(double), cudaMemcpyDeviceToHost);
        stepDiff = 0.0;
        for (int i = 0; i < blockNum; ++i) {
            stepDiff += dotBuf1[i];
        }
        if (size != 1) {
            MPI_Allreduce(MPI_IN_PLACE, &stepDiff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            sendNodes(d_w, coord, range, pGrigSize, sendBuf);
            receiveNodes(d_w, coord, range, pGrigSize, recvBuf, blocksPerGrid, threadsPerBlock);
        }
    } while (sqrt(stepDiff) > error);
    cudaDeviceSynchronize();
    cudaMemcpy(ctx->w, d_w, (ctx->M + 1)*(ctx->N + 1)*sizeof(double), cudaMemcpyDeviceToHost);

    delete[] dotBuf1;
    delete[] dotBuf2;
    freeDeviceMemory();
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

    ctx->domain.x1 = ctx->domain.x1 + (range.x1 - 1)*ctx->h1;
    ctx->domain.y1 = ctx->domain.y1 + (range.y1 - 1)*ctx->h2;
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

void GPUSolver::sendNodes(double *w, const ProcessorCoordinates &coord, 
    const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf)
{

    int gridSize = ((ctx->M > ctx->N ? ctx->M : ctx->N) - 2)/1024 + 1;
    int blockSize = 1024;

    getNodes<<<gridSize, blockSize>>>(ctx->M, ctx->N, w, d_topNodes, d_bottomNodes, d_rightNodes, d_leftNodes);
    cudaDeviceSynchronize();

    if (coord.x != 0) {
        MPI_Send(d_leftNodes, ctx->N - 1, MPI_DOUBLE, coord.y*pGridSize.first + coord.x - 1, 
            RIGHT_NODES, MPI_COMM_WORLD);
    }
    if (coord.x != pGridSize.first - 1) {
        MPI_Send(d_rightNodes, ctx->N - 1, MPI_DOUBLE, coord.y*pGridSize.first + coord.x + 1, 
            LEFT_NODES, MPI_COMM_WORLD);
    }
    if (coord.y != 0) {
        MPI_Send(d_bottomNodes, ctx->M - 1, MPI_DOUBLE, (coord.y - 1)*pGridSize.first + coord.x, TOP_NODES, MPI_COMM_WORLD);
    }
    if (coord.y != pGridSize.second - 1) {
        MPI_Send(d_topNodes, ctx->M - 1, MPI_DOUBLE, (coord.y + 1)*pGridSize.first + coord.x, BOTTOM_NODES, MPI_COMM_WORLD);
    }

}

void GPUSolver::receiveNodes(double *w, const ProcessorCoordinates &coord, 
    const IndexRange &range, const std::pair<int, int> &pGridSize, double *buf,
    const dim3 &blocksPerGrid, const dim3 &threadsPerBlock)
{

    int gridSize = ((ctx->M > ctx->N ? ctx->M : ctx->N) - 2)/1024 + 1;
    int blockSize = 1024;

    double *topNodes = NULL;
    double *bottomNodes = NULL;
    double *rightNodes = NULL;
    double *leftNodes = NULL;

    MPI_Status status;

    if (coord.x != 0) {
        MPI_Recv(r_leftNodes, ctx->N - 1, MPI_DOUBLE, MPI_ANY_SOURCE, LEFT_NODES, MPI_COMM_WORLD, &status);
        leftNodes = r_leftNodes;
    }
    if (coord.x != pGridSize.first - 1) {
        MPI_Recv(r_rightNodes, ctx->N - 1, MPI_DOUBLE, MPI_ANY_SOURCE, RIGHT_NODES, MPI_COMM_WORLD, &status);
        rightNodes = r_rightNodes;
    }
    if (coord.y != 0) {
        MPI_Recv(r_bottomNodes, ctx->M - 1, MPI_DOUBLE, MPI_ANY_SOURCE, BOTTOM_NODES, MPI_COMM_WORLD, &status);
        bottomNodes = r_bottomNodes;
    }
    if (coord.y != pGridSize.second - 1) {
        MPI_Recv(r_topNodes, ctx->M - 1, MPI_DOUBLE, MPI_ANY_SOURCE, TOP_NODES, MPI_COMM_WORLD, &status);
        topNodes = r_topNodes;
    }

    cudaDeviceSynchronize();
    loadNodes<<<gridSize, blockSize>>>(ctx->M, ctx->N, w, topNodes, bottomNodes, rightNodes, leftNodes);

}

void GPUSolver::initDeviceMemory()
{

    cudaMalloc(&d_w, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));
    cudaMemcpy(d_w, ctx->w, (ctx->M + 1)*(ctx->N + 1)*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_B, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));
    cudaMemcpy(d_B, ctx->B, (ctx->M + 1)*(ctx->N + 1)*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Aw, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));
    cudaMemcpy(d_Aw, ctx->Aw, (ctx->M + 1)*(ctx->N + 1)*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_r, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));
    cudaMemcpy(d_r, ctx->r, (ctx->M + 1)*(ctx->N + 1)*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Ar, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));
    cudaMemcpy(d_Ar, ctx->Ar, (ctx->M + 1)*(ctx->N + 1)*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_buf, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));
    cudaMalloc(&d_dotBuf1, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));
    cudaMalloc(&d_dotBuf2, (ctx->M + 1)*(ctx->N + 1)*sizeof(double));

    cudaHostAlloc(&d_topNodes, (ctx->M - 1)*sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc(&d_bottomNodes, (ctx->M - 1)*sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc(&d_rightNodes, (ctx->N - 1)*sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc(&d_leftNodes, (ctx->N - 1)*sizeof(double), cudaHostAllocMapped);

    cudaHostAlloc(&r_topNodes, (ctx->M - 1)*sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc(&r_bottomNodes, (ctx->M - 1)*sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc(&r_rightNodes, (ctx->N - 1)*sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc(&r_leftNodes, (ctx->N - 1)*sizeof(double), cudaHostAllocMapped);

    cudaMalloc(&d_nodes, 2*(ctx->N - 1 + ctx->M - 1)*sizeof(double));

}

void GPUSolver::freeDeviceMemory()
{

    cudaFree(d_w);
    cudaFree(d_B);
    cudaFree(d_Aw);
    cudaFree(d_r);
    cudaFree(d_Ar);
    cudaFree(d_buf);
    cudaFree(d_dotBuf1);
    cudaFree(d_dotBuf2);
    cudaFree(d_topNodes);
    cudaFree(d_bottomNodes);
    cudaFree(d_rightNodes);
    cudaFree(d_leftNodes);
    cudaFree(d_nodes);
    cudaFree(r_bottomNodes);
    cudaFree(r_rightNodes);
    cudaFree(r_leftNodes);

}
