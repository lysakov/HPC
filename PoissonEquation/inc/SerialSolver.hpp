#ifndef SOLVER
#define SOLVER

#include "ISolver.hpp"

#include <cstddef>
#include <ostream>

class SerialAlgebra : public AbstractLinearAlgebra
{

public:
    SerialAlgebra(int M, int N, Domain domain) : M(M), N(N), domain(domain)
    {
        h1 = (domain.x2 - domain.x1) / M;
        h2 = (domain.y2 - domain.y1) / N;
    }
    virtual double dot(double **u, double **v, const IndexRange &range);
    virtual void mult(double **r, double alpha, double **u, const IndexRange &range);
    virtual void subs(double **r, double **u, double **v, const IndexRange &range);
    virtual void A(double **r, double **u, const IndexRange &range,
        double (*k)(double, double), double (*q)(double, double));

private:
    int M, N;
    double h1, h2;
    Domain domain;

};

class SerialSolver: public ISolver
{

public:
    SerialSolver(Context *ctx, AbstractLinearAlgebra *engine) : ctx(ctx), engine(engine) {}
    virtual void solve(double error);
    virtual double** getSolution();
    void step(const IndexRange &range);
    virtual double getError(double (*u)(double, double));
    virtual ~SerialSolver() {}

    friend std::ostream& operator<<(std::ostream& str, const SerialSolver &solver);

private:
    void copyW();

    Context *ctx;
    AbstractLinearAlgebra *engine;

};

#endif