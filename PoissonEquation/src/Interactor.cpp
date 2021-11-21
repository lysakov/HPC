#include "Interactor.hpp"

#include <stdexcept>
#include <cstdlib>
#include <iostream>

#include "SerialSolver.hpp"
#include "ParallelSolver.hpp"
#include "MultiThreadSolver.hpp"

CommandLineController::CommandLineController(int argc, char **argv)
{

    for (int i = 1; i < argc - 1; ++i) {
        args.insert(std::make_pair<std::string, std::string>(argv[i], argv[i+1]));
    }

}

Configuration CommandLineController::parse()
{

    Configuration conf;

    if (args.count("-m")) {
        if (args["-m"] == "S") {
            conf.type = SERIAL_SOLVER;
        }
        else if (args["-m"] == "MPI") {
            conf.type = MPI_SOLVER;
        }
    }

    if (args.count("-b")) {
        if (args["-b"] == "OMP") {
            conf.type = MPI_OMP_SOLVER;
        }
        else if (args["-b"] == "CUDA") {
            conf.type = MPI_CUDA_SOLVER;
        }
    }

    if (args.count("-M")) {
        conf.M = static_cast<int>(std::strtol(args["-M"].c_str(), NULL, 10));
    }

    if (args.count("-N")) {
        conf.N = static_cast<int>(std::strtol(args["-N"].c_str(), NULL, 10));
    }

    if (args.count("-e")) {
        conf.error = std::strtod(args["-e"].c_str(), NULL);
    }

    if (args.count("-p")) {
        conf.printSolution = args["-p"] == "ON" ? true : false;
    }

    return conf;

}

double** Interactor::solve()
{

    AbstractLinearAlgebra *algebra = NULL;
    ISolver *solver = NULL;

    switch (conf.type) {
        case SERIAL_SOLVER:
            algebra = new SerialAlgebra(ctx->M, ctx->N, ctx->domain);
            solver = new SerialSolver(ctx, algebra);
            break;
        case MPI_SOLVER:
            algebra = new SerialAlgebra(ctx->M, ctx->N, ctx->domain);
            solver = new ParallelSolver(ctx, algebra, rank, size);
            break;
        case MPI_OMP_SOLVER:
            solver = new MultiThreadSolver(ctx, rank, size);
            break;
        default:
            break;
    }

    double start = MPI_Wtime();
    solver->solve(conf.error);
    double finish = MPI_Wtime();
    solver->getSolution();
    double solutionError = 0.0;
    if (u) {
        solutionError = solver->getError(u);
    }

    double **solution = solver->getSolution();

    if (rank == 0) {
        std::cout << "Process number: " << size << std::endl;
        std::cout << "M: " << ctx->M << std::endl;
        std::cout << "N: " << ctx->N << std::endl;
        std::cout << "Running time: " << finish - start << std::endl;
        std::cout << "Iterration error: " << conf.error << std::endl;
        if (u) {
            std::cout << "Error: " << solutionError << std::endl;
        }
    }

    if (conf.printSolution) {
        printSolution(ctx);
    }

    delete solver;
    if (algebra) {
        delete algebra;
    }

    return solution;

}

void Interactor::printSolution(const Context *ctx) {
    
    std::cout << "X = np.linspace(" << ctx->domain.x1 << ", " 
        << ctx->domain.x2 << ", " << "num=" << ctx->M + 1 << ")\n"; 
    std::cout << "Y = np.linspace(" << ctx->domain.y1 << ", " 
        << ctx->domain.y2 << ", " << "num=" << ctx->N + 1 << ")\n"; 
    std::cout << "X, Y = np.meshgrid(X, Y)" << std::endl;

    std::cout << "Z = np.array([";
    for (int j = 0; j <= ctx->N; ++j) {
        std::cout << "[";
        for (int i = 0; i <= ctx->M; ++i) {
            std::cout << ctx->w[i][j];
            if (i != ctx->M) std::cout << ", ";
            else std::cout << "]";
        }
        if (j != ctx->N) std::cout << ",\n";
        else std::cout << "])\n";
    }

    std::cout << "targetZ = np.exp(1 - (X + Y)**2)" << std::endl;

}