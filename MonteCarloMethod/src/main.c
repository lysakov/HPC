#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "Master.h"
#include "Worker.h"

enum {
    POINTS_BROADCAST,
    REQUEST_POINTS_BROADCAST,
    FINAL_BROADCAST
};

enum {
    SUM,
    POINTS_GENERATED
};

int isInDomain(Point M)
{

    return M.z >= 0 && M.x*M.x + M.y*M.y + M.z*M.z <= 1;

}

double F(Point M)
{

    return isInDomain(M) ? exp(M.x*M.x + M.y*M.y) * M.z : 0.0;

}

void generatePointBuf(Point **buf, struct Master *master, int size, int dotNum) 
{
    
    for (int i = 1; i < size; ++i) {
        buf[i] = generateDots(master, dotNum);
    }

}

void sendPoints(Point **buf, int size, int dotNum)
{

    MPI_Request *requests = (MPI_Request*)malloc(size*sizeof(MPI_Request)); 
    for (int i = 1; i < size; ++i) {
        MPI_Isend((void*)buf[i], 3*dotNum, MPI_DOUBLE, i, POINTS_BROADCAST, MPI_COMM_WORLD, &requests[i]);
    }
    free(requests);

}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    int N = 100;
    double result_buf[2] = {0};

    if (argc > 2) {
        N = (int)strtol(argv[2], NULL, 10);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        Domain domain = {.x1 = -1, .x2 = 1, .y1 = -1, .y2 = 1, .z1 = 0, .z2 = 1};
        double V = 4.0;
        struct Master *master = createMaster(domain);
        double targetVal = 2.0 * M_PI * (M_E / 4 - 0.5);
        double error = strtod(argv[1], NULL);
        double startTime = MPI_Wtime();
        Point **point_buf1 = (Point**)malloc(size*sizeof(Point*));
        Point **point_buf2 = (Point**)malloc(size*sizeof(Point*));
        generatePointBuf(point_buf1, master, size, N);
        sendPoints(point_buf1, size, N);

        double I = 0.0;
        do {

            sendPoints(point_buf1, size, N);
            generatePointBuf(point_buf2, master, size, N);
            MPI_Reduce(MPI_IN_PLACE, result_buf, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (result_buf[POINTS_GENERATED] != 0) {
                I = V*result_buf[SUM]/result_buf[POINTS_GENERATED];
            }

            for (int i = 1; i < size; ++i) {
                free(point_buf1[i]);
            }

            Point **tmp;
            tmp = point_buf1;
            point_buf1 = point_buf2;
            point_buf2 = tmp;

            /*int pointRequest;
            MPI_Status status;
            Point *points = generateDots(master, N);
            MPI_Recv(&pointRequest, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_POINTS_BROADCAST, MPI_COMM_WORLD, &status);
            MPI_Send((void*)points, 3*N, MPI_DOUBLE, status.MPI_SOURCE, POINTS_BROADCAST, MPI_COMM_WORLD);
            MPI_Ireduce(MPI_IN_PLACE, result_buf, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &pointRequest);
            free(points);
            //printf("%f, %f, %f\n", result_buf[0], result_buf[1], I);
            if (result_buf[POINTS_GENERATED] != 0) {
                I = V*result_buf[SUM]/result_buf[POINTS_GENERATED];
            }*/
        } while (fabs(I - targetVal) > error);

        for (int i = 1; i < size; ++i) {
            int data = 0;
            MPI_Send(&data, 1, MPI_INT, i, FINAL_BROADCAST, MPI_COMM_WORLD);
        }
        
        double endTime = MPI_Wtime();

        printf("I = %f\n", I);
        printf("Process number: %d\n", size);
        printf("Running time: %f\n", endTime - startTime);
        printf("Target error: %f\n", error);
        printf("Error: %f\n", fabs(I - targetVal));
        printf("Points generated: %ld\n", (long int)result_buf[POINTS_GENERATED]);

    }
    else {
        /*MPI_Request request = 0;
        double sum = 0.0;
        double dots = 0.0;*/

        while(1) {
            MPI_Status status;
            Point *points = (Point*)malloc(N * sizeof(Point));
            MPI_Recv((void*)points, 3*N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == FINAL_BROADCAST) {
                free(points);
                break;
            }

            /*int flag = 1;
            sum += compute(F, points, N);
            dots += N;

            if (request) {
                MPI_Test(&request, &flag, &status);
            }

            if (flag) {
                result_buf[SUM] = sum;
                result_buf[POINTS_GENERATED] = dots;
                sum = 0.0;
                dots = 0.0;
                //printf("%d, %f, %f\n", rank, result_buf[0], result_buf[1]);

                MPI_Ireduce(result_buf, NULL, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &request);
            }*/
            //printf("%d %d %f %f\n", rank, request, sum, dots);
            //MPI_Send(&request, 1, MPI_INT, 0, REQUEST_POINTS_BROADCAST, MPI_COMM_WORLD);

            result_buf[SUM] = compute(F, points, N);
            result_buf[POINTS_GENERATED] = N;
            //printf("%d, %f, %f\n", rank, result_buf[0], result_buf[1]);

            MPI_Reduce(result_buf, NULL, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            free(points);
        }
    }
    
    MPI_Finalize();

    return 0;

}