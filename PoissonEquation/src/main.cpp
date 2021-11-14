#include <iostream>
#include <string>
#include <mpi.h>
#include <omp.h>

#include "functions.hpp"
#include "SerialSolver.hpp"
#include "ParallelSolver.hpp"

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Domain domain(-1.0, 2.0, -2.0, 2.0);
    int M = (int)strtol(argv[1], NULL, 10);
    int N = (int)strtol(argv[2], NULL, 10);
    double error = 0.0000001;
    (void)error;
    if (argc > 3) {
        error = strtod(argv[3], NULL);
    }
    Context serialCtx = Context(k, q, F, phi_1, phi_2, phi_3, phi_4, M, N, domain);
    //Context parallelCtx = Context(k, q, F, phi_1, phi_2, phi_3, phi_4, M, N, domain);
    
    if (rank == 0) {
        AbstractLinearAlgebra *algebra = new SerialAlgebra(M, N, domain);
        ISolver *solver = new SerialSolver(&serialCtx, algebra);
        IndexRange range(1, M-1, 1, N-1);
        double start = MPI_Wtime();
        solver->solve(error);
        double finish = MPI_Wtime();
        std::cout << "*********Serial solver**********" << std::endl;
        std::cout << "Process number: " << 1 << std::endl;
        std::cout << "M: " << M << std::endl;
        std::cout << "N: " << N << std::endl;
        std::cout << "Running time: " << finish - start << std::endl;
        std::cout << "Iterration error: " << error << std::endl;
        std::cout << "Error: " << ((SerialSolver*)solver)->getError(u) << std::endl;
        delete solver;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    /*AbstractLinearAlgebra *algebra = new SerialAlgebra(M, N, domain);
    try {
        ParallelSolver pSolver(&parallelCtx, algebra, rank, size);
        IndexRange range;
        pSolver.splitGrid(rank, size, range);
        //std::cout << "{" << range.x1 << " " << range.x2 << "}"<< "{" << range.y1 << " " << range.y2 << "}\n";
        double start = MPI_Wtime();
        pSolver.solve(error);
        double finish = MPI_Wtime();
        double trueError = pSolver.getError(domain, u);
        if (rank == 0) {
            std::cout << "*********Parallel solver**********" << std::endl;
            std::cout << "Process number: " << size << std::endl;
            std::cout << "M: " << M << std::endl;
            std::cout << "N: " << N << std::endl;
            std::cout << "Running time: " << finish - start << std::endl;
            std::cout << "Iterration error: " << error << std::endl;
            std::cout << "Error: " << trueError << std::endl;
        }
        delete algebra;
    }
    catch(std::exception &e) {
        std::cout << e.what() << std::endl;
    }*/
    MPI_Finalize();

    return 0;

}