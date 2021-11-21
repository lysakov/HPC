#ifndef ISOLVER
#define ISOLVER

#include <cmath>

struct Domain
{
    double x1, x2;
    double y1, y2;
    Domain(double x1, double x2, double y1, double y2): x1(x1), x2(x2), y1(y1), y2(y2) {}
};

struct IndexRange
{
    int x1, x2;
    int y1, y2;
    IndexRange(int x1 = 0, int x2 = 0, int y1 = 0, int y2 = 0): x1(x1), x2(x2), y1(y1), y2(y2) {}
};

class AbstractLinearAlgebra
{

public:
    virtual double dot(double **u, double **v, const IndexRange &range) = 0;
    virtual double norm(double **u, const IndexRange &range) {
        return sqrt(dot(u, u, range));
    }
    virtual void mult(double **r, double alpha, double **u, const IndexRange &range) = 0;
    virtual void subs(double **r, double **u, double **v, const IndexRange &range) = 0;
    virtual void A(double **r, double **u, const IndexRange &range,
        double (*k)(double, double), double (*q)(double, double)) = 0;
    virtual ~AbstractLinearAlgebra() {}

};

class ISolver
{

public:
    virtual void solve(double error) = 0;
    virtual double** getSolution() = 0;
    virtual double getError(double (*u)(double, double)) = 0;
    virtual ~ISolver() {}

};

class Context
{

public:
    Context(double (*k)(double, double), 
        double (*q)(double, double), 
        double (*F)(double, double), 
        double (*phi_1)(double), double (*phi_2)(double), 
        double (*phi_3)(double), double (*phi_4)(double),
        int M, int N, Domain domain) :
        k(k), q(q), F(F),
        phi_1(phi_1), phi_2(phi_2),
        phi_3(phi_3), phi_4(phi_4), 
        M(M), N(N), domain(domain)
    {
        #define x(i) (domain.x1 + i*h1)
        #define y(i) (domain.y1 + i*h2)

        this->h1 = (domain.x2 - domain.x1) / M;
        this->h2 = (domain.y2 - domain.y1) / N;
        this->N = N;
        this->M = M;
        this->domain = domain;

        w = new double*[M + 1];
        curF = new double*[M + 1];
        this->B = new double*[M + 1];
        r = new double*[M + 1];
        Ar = new double*[M + 1];
        buf = new double*[M + 1];
        for (int i = 0; i < M + 1; ++i) {
            w[i] = new double[N + 1];
            curF[i] = new double[N + 1];
            B[i] = new double[N + 1];
            r[i] = new double[N + 1];
            Ar[i] = new double[N + 1];
            buf[i] = new double[N + 1];
        }

        for (int i = 0; i < M + 1; ++i) {
            for (int j = 0; j < N + 1; ++j) {
                r[i][j] = 0.0;
                Ar[i][j] = 0.0;
                buf[i][j] = 0.0;
                w[i][j] = 0.0;
                curF[i][j] = 0.0;
                if (i == 0) {
                    B[i][j] = phi_1(y(j));
                    continue;
                }
                if (i == M) {
                    B[i][j] = phi_2(y(j));
                    continue;
                }
                if (j == 0) {
                    B[i][j] = phi_3(x(i));
                    continue;
                }
                if (j == N) {
                    B[i][j] = phi_4(x(i));
                    continue;
                }

                B[i][j] = F(x(i), y(j));

                if (i == 1) {
                    this->B[i][j] += k(x(i) - 0.5*h1, y(j))*phi_1(y(j))/(h1*h1);
                }
                if (i == M - 1) {
                    this->B[i][j] += k(x(i) + 0.5*h1, y(j))*phi_2(y(j))/(h1*h1);
                }
                if (j == 1) {
                    this->B[i][j] += k(x(i), y(j) - 0.5*h2)*phi_3(x(i))/(h2*h2);
                }
                if (j == N - 1) {
                    this->B[i][j] += k(x(i), y(j) + 0.5*h2)*phi_4(x(i))/(h2*h2);
                }
            }
        }
        #undef x
        #undef y
    }

    void finalize()
    {

        for (int i = 0; i < M + 1; ++i) {
            for (int j = 0; j < N + 1; ++j) {
                if (i == 0) {
                    w[i][j] = phi_1(domain.y1 + j*h2);
                    continue;
                }
                if (i == M) {
                    w[i][j] = phi_2(domain.y1 + j*h2);
                    continue;
                }
                if (j == 0) {
                    w[i][j] = phi_3(domain.x1 + i*h1);
                    continue;
                }
                if (j == N) {
                    w[i][j] = phi_4(domain.x1 + i*h1);
                    continue;
                }
            }
        }

    }

    ~Context()
    {
        for (int i = 0; i < M + 1; ++i) {
            delete[] B[i];
            delete[] w[i];
            delete[] curF[i];
            delete[] r[i];
            delete[] Ar[i];
            delete[] buf[i];
        }

        delete[] B;
        delete[] w;
        delete[] curF;
        delete[] r;
        delete[] Ar;
        delete[] buf;
    }

    double (*k)(double, double);
    double (*q)(double, double);
    double (*F)(double, double);
    double (*phi_1)(double);
    double (*phi_2)(double);
    double (*phi_3)(double);
    double (*phi_4)(double);

    double **w;
    double **B;
    double **buf;
    double **curF;
    double **r;
    double **Ar;
    double h1, h2;
    int M, N;
    Domain domain;

};

#endif