#include <iostream>
#include <string>
#include <mpi.h>
#include <omp.h>

#include "functions.hpp"
#include "Interactor.hpp"

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Domain domain(0, 0, 0, 0);
    initDomain(domain.x1, domain.x2, domain.y1, domain.y2);

    CommandLineController controller(argc, argv);
    Configuration conf = controller.parse();

    Context ctx(k, q, F, phi_1, phi_2, phi_3, phi_4, conf.M, conf.N, domain);
    Interactor interactor(conf, &ctx, rank, size, u);

    interactor.solve();

    MPI_Finalize();

    return 0;

}