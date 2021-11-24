#include "SerialSolver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstring>

#include "macros.hpp"

double SerialAlgebra::dot(double *u, double *v, const IndexRange &range) 
{

    double sum = 0.0;
    for (int i = range.x1; i <= range.x2; ++i) {
        double rowSum = 0.0; 
        for (int j = range.y1; j <= range.y2; ++j) {
            rowSum += h2*u[i*(N + 1) + j]*v[i*(N + 1) + j];
        }
        sum += h1*rowSum;
    }

    return sum;

}

void SerialAlgebra::mult(double *r, double alpha, double *u, const IndexRange &range)
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i*(N + 1) + j] = alpha*u[i*(N + 1) + j];
        }
    }

}

void SerialAlgebra::subs(double *r, double *u, double *v, const IndexRange &range)
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i*(N + 1) + j] = u[i*(N + 1) + j] - v[i*(N + 1) + j];
        }
    }

}

void SerialAlgebra::A(double *r, double *u, const IndexRange &range,
    double (*k)(double, double), double (*q)(double, double))
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i*(N + 1) + j] = -dx(u, k) - dy(u, k) + q(x(i), y(j))*u[i*(N + 1) + j];
        }        
    }

}

void SerialSolver::solve(double error)
{

    IndexRange range(1, ctx->M - 1, 1, ctx->N - 1);
    do {
        std::memcpy(ctx->buf, ctx->w, sizeof(double)*(ctx->M + 1)*(ctx->N + 1));
        step(range);
        engine->subs(ctx->buf, ctx->w, ctx->buf, range);
    } while(engine->norm(ctx->buf, range) > error);

}

void SerialSolver::step(const IndexRange &range)
{

    //Aw = A@w
    engine->A(ctx->Aw, ctx->w, range, ctx->k, ctx->q);

    //r = Aw - B
    engine->subs(ctx->r, ctx->Aw, ctx->B, range);

    //Ar = A@r
    engine->A(ctx->Ar, ctx->r, range, ctx->k, ctx->q);

    //tau = <Ar, r> / ||Ar||**2
    double norm = engine->norm(ctx->Ar, range);
    double tau = engine->dot(ctx->Ar, ctx->r, range) / (norm*norm);

    //w = w - tau*r
    engine->mult(ctx->r, tau, ctx->r, range);
    engine->subs(ctx->w, ctx->w, ctx->r, range);

}

double SerialSolver::getError(double (*u)(double, double))
{

    double *U = new double[(ctx->M + 1)*(ctx->N + 1)];

    for (int i = 0; i <= ctx->M; ++i)
        for (int j = 0; j <= ctx->N; ++j)
            U[i*(ctx->N + 1) + j] = u(ctx->domain.x1 + i*ctx->h1, ctx->domain.y1 + j*ctx->h2);

    IndexRange range(1, ctx->M - 1, 1, ctx->N - 1);
    engine->subs(U, U, ctx->w, range);
    double norm = engine->norm(U, range);

    delete[] U;

    return norm;

}

double* SerialSolver::getSolution()
{

    ctx->finalize();

    return ctx->w;

}

std::ostream& operator<<(std::ostream& str, const SerialSolver &solver)
{

    str << "B = " << std::endl;
    for (int j = solver.ctx->N; j >= 0; --j) {
        for (int i = 0; i < solver.ctx->M + 1; ++i) {
            str << solver.ctx->B[i*(solver.ctx->N + 1) + j] << " ";
        }
        str << std::endl;
    }

    str << "w = " << std::endl;
    for (int j = solver.ctx->N; j >= 0; --j) {
        for (int i = 0; i < solver.ctx->M + 1; ++i) {
            str << solver.ctx->w[i*(solver.ctx->N + 1) + j] << " ";
        }
        str << std::endl;
    }

    str << "|r| = " << std::endl;
    for (int j = solver.ctx->N; j >= 0; --j) {
        for (int i = 0; i < solver.ctx->M + 1; ++i) {
            str << fabs(solver.ctx->Aw[i*(solver.ctx->N + 1) + j] - solver.ctx->B[i*(solver.ctx->N + 1) + j]) << " ";
        }
        str << std::endl;
    }

    return str;

}
