#include "Worker.h"

double compute(double (*F)(Point M), Point *points, int N)
{

    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        sum += F(points[i]);
    }

    return sum;

}