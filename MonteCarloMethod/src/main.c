#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "Master.h"
#include "Worker.h"

double F(Point M)
{

    return exp(M.x*M.x + M.y*M.y) * M.z*M.z;

}

int isInDomain(Point M)
{

    return M.z >= 0 && M.x*M.x + M.y*M.y + M.z*M.z <= 1;

}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        Domain domain = {.x1 = -1, .x2 = 1, .y1 = -1, .y2 = 1, .z1 = -1, .z2 = 1};
        struct Master *master = createMaster(domain);

        int N = 1000;
        Point *points = generateDots(master, N);
        for (int i = 0; i < N; ++i) {
            printf("x = %f, y = %f, z = %f\n", points[i].x, points[i].y, points[i].z);
        }

        printf("Sum = %f\n", compute(F, points, N));

        free(points);
        deleteMaster(master);

    }
    
    MPI_Finalize();

    return 0;

}