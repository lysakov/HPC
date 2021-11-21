#include "SerialSolver.hpp"

#include <cstddef>
#include <cmath>
#include <iostream>

#include "macros.hpp"

double SerialAlgebra::dot(double **u, double **v, const IndexRange &range) 
{

    double sum = 0.0;
    for (int i = range.x1; i <= range.x2; ++i) {
        double rowSum = 0.0; 
        for (int j = range.y1; j <= range.y2; ++j) {
            rowSum += h2*u[i][j]*v[i][j];
        }
        sum += h1*rowSum;
    }

    return sum;

}

void SerialAlgebra::mult(double **r, double alpha, double **u, const IndexRange &range)
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i][j] = alpha*u[i][j];
        }
    }

}

void SerialAlgebra::subs(double **r, double **u, double **v, const IndexRange &range)
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i][j] = u[i][j] - v[i][j];
        }
    }

}

void SerialAlgebra::A(double **r, double **u, const IndexRange &range,
    double (*k)(double, double), double (*q)(double, double))
{

    for (int i = range.x1; i <= range.x2; ++i) {
        for (int j = range.y1; j <= range.y2; ++j) {
            r[i][j] = -dx(u, k) - dy(u, k) + q(x(i), y(j))*u[i][j];
        
            /*if (i == 1) {
                r[i][j] += k(x(i) - 0.5*h1, y(j))*u[i - 1][j]/(h1*h1);
            }
            if (i == M - 1) {
                r[i][j] += k(x(i) + 0.5*h1, y(j))*u[i + 1][j]/(h1*h1);
            }
            if (j == 1) {
                r[i][j] += k(x(i), y(j) - 0.5*h2)*u[i][j - 1]/(h2*h2);
            }
            if (j == N - 1) {
                r[i][j] += k(x(i), y(j) + 0.5*h2)*u[i][j + 1]/(h2*h2);
            }*/
        }        
    }

}

void SerialSolver::solve(double error)
{

    IndexRange range(1, ctx->M - 1, 1, ctx->N - 1);
    do {
        copyW();
        step(range);
        engine->subs(ctx->buf, ctx->w, ctx->buf, range);
    } while(engine->norm(ctx->buf, range) > error);

}

void SerialSolver::step(const IndexRange &range)
{

    //curF = A@w
    engine->A(ctx->curF, ctx->w, range, ctx->k, ctx->q);

    //r = curF - B
    engine->subs(ctx->r, ctx->curF, ctx->B, range);

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

    double **U = new double*[ctx->M + 1];
    for (int i = 0; i < ctx->M + 1; ++i) {
        U[i] = new double[ctx->N + 1];
    }

    for (int i = 0; i <= ctx->M; ++i)
        for (int j = 0; j <= ctx->N; ++j)
            U[i][j] = u(ctx->domain.x1 + i*ctx->h1, ctx->domain.y1 + j*ctx->h2);

    /*for (int j = N; j >= 0; --j) {
        for (int i = 0; i < M + 1; ++i) {
            std::cout << fabs(U[i][j] - w[i][j]) << " ";
        }
        std::cout << std::endl;
    }*/

    IndexRange range(1, ctx->M - 1, 1, ctx->N - 1);
    engine->subs(U, U, ctx->w, range);
    double norm = engine->norm(U, range);

    for (int i = 0; i < ctx->M + 1; ++i) {
        delete[] U[i];
    }
    delete[] U;

    return norm;

}

void SerialSolver::copyW()
{

    for (int i = 1; i < ctx->M; ++i) {
        for (int j = 1; j < ctx->N; ++j) {
            ctx->buf[i][j] = ctx->w[i][j];
        }
    }

}

double** SerialSolver::getSolution()
{

    ctx->finalize();

    return ctx->w;

}

std::ostream& operator<<(std::ostream& str, const SerialSolver &solver)
{

    str << "B = " << std::endl;
    for (int j = solver.ctx->N; j >= 0; --j) {
        for (int i = 0; i < solver.ctx->M + 1; ++i) {
            str << solver.ctx->B[i][j] << " ";
        }
        str << std::endl;
    }

    str << "w = " << std::endl;
    for (int j = solver.ctx->N; j >= 0; --j) {
        for (int i = 0; i < solver.ctx->M + 1; ++i) {
            str << solver.ctx->w[i][j] << " ";
        }
        str << std::endl;
    }

    str << "|r| = " << std::endl;
    for (int j = solver.ctx->N; j >= 0; --j) {
        for (int i = 0; i < solver.ctx->M + 1; ++i) {
            str << fabs(solver.ctx->curF[i][j] - solver.ctx->B[i][j]) << " ";
        }
        str << std::endl;
    }

    return str;

}
