#ifndef WORKER
#define WORKER

#include "geometry.h"

double compute(double (*F)(Point M), Point *points, int N);

#endif