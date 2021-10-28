#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "Master.h"
#include "Worker.h"

enum {
    POINTS_BROADCAST,
    RESULT_BROADCAST,
    FINAL_BROADCAST
};

int isInDomain(Point M)
{

    return M.z >= 0 && M.x*M.x + M.y*M.y + M.z*M.z <= 1;

}

double F(Point M)
{

    return isInDomain(M) ? exp(M.x*M.x + M.y*M.y) * M.z : 0.0;

}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    int N = 100;

    if (argc > 2) {
        N = (int)strtol(argv[2], NULL, 10);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        Domain domain = {.x1 = -1, .x2 = 1, .y1 = -1, .y2 = 1, .z1 = 0, .z2 = 1};
        struct Master *master = createMaster(domain);
        double targetVal = 2.0 * M_PI * (M_E / 4 - 0.5);
        double error = strtod(argv[1], NULL);
        double startTime = MPI_Wtime();
        long dotsGenerated = 0;
        Point **pointBuf = (Point**)malloc((size)*sizeof(Point*));

        for (int i = 1; i < size; ++i) {
            MPI_Request request;
            pointBuf[i] = generateDots(master, N);
            MPI_Isend((void*)pointBuf[i], 3*N, MPI_DOUBLE, i, POINTS_BROADCAST, MPI_COMM_WORLD, &request);
            dotsGenerated += N;
        }

        while (fabs(computeResult(master) - targetVal) > error) {
            double res = 0.0;
            MPI_Status status;
            MPI_Request request;
            Point *points = generateDots(master, N);
            dotsGenerated += N;

            MPI_Recv(&res, 1, MPI_DOUBLE, MPI_ANY_SOURCE, RESULT_BROADCAST, MPI_COMM_WORLD, &status);

            free(pointBuf[status.MPI_SOURCE]);
            pointBuf[status.MPI_SOURCE] = points;

            MPI_Isend((void*)points, 3*N, MPI_DOUBLE, status.MPI_SOURCE, POINTS_BROADCAST, MPI_COMM_WORLD, &request);
            saveResult(master, res);
        }

        for (int i = 1; i < size; ++i) {
            int data = 0;
            MPI_Send(&data, 1, MPI_INT, i, FINAL_BROADCAST, MPI_COMM_WORLD);
        }
        
        double endTime = MPI_Wtime();

        printf("I = %f\n", computeResult(master));
        printf("Process number: %d\n", size);
        printf("Running time: %f\n", endTime - startTime);
        printf("Target error: %f\n", error);
        printf("Error: %f\n", fabs(targetVal - computeResult(master)));
        printf("Points generated: %ld\n", dotsGenerated);

        for (int i = 1; i < size; ++i) {
            free(pointBuf[i]);
        }
        free(pointBuf);
        deleteMaster(master);
    }
    else {
        while(1) {
            MPI_Status status;
            Point *points = (Point*)malloc(N * sizeof(Point));
            MPI_Recv((void*)points, 3 * N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == FINAL_BROADCAST) {
                free(points);
                break;
            }

            double res = compute(F, points, N);
            MPI_Send(&res, 1, MPI_DOUBLE, 0, RESULT_BROADCAST, MPI_COMM_WORLD);
            free(points);
        }
    }
    
    MPI_Finalize();

    return 0;

}