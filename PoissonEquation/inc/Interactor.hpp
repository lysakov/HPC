#ifndef INTERACTOR
#define INTERACTOR

#include <string>
#include <map>

#include "ISolver.hpp"

enum SolversType {
    SERIAL_SOLVER,
    MPI_SOLVER,
    MPI_OMP_SOLVER,
    MPI_CUDA_SOLVER
};

struct Configuration {

    SolversType type;
    int M;
    int N;
    double error;
    bool printSolution;
    Configuration(SolversType type = SERIAL_SOLVER, int M = 100, int N = 100, 
        double error = 0.000001, bool printSolution = false) :
        type(type), M(M), N(N), error(error), printSolution(printSolution) {}

};

class CommandLineController
{

public:
    CommandLineController(int argc, char **argv);
    Configuration parse();

private:
    std::map<std::string, std::string> args;

};

class Interactor
{

public:
    Interactor(Configuration conf, Context *ctx, int rank, int size, 
        double (*u)(double, double) = NULL) : 
        conf(conf), ctx(ctx), rank(rank), size(size), u(u) {}
    double** solve();

private:
    void printSolution(const Context *ctx);

    Configuration conf;
    Context *ctx;
    int rank, size;
    double (*u)(double, double);

};

#endif