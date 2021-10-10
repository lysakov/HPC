#include <iostream>
#include <mpi.h>

int main(int argc, char **argv)
{

    MPI::Init(argc, argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 3) {
        std::cout << "Hello world!" << std::endl;
    }
    
    MPI::Finalize();

    return 0;

}