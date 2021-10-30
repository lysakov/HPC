#ifndef MASTER
#define MASTER

#include "geometry.h"

struct Master;

struct Master* createMaster(const Domain domain);
Point* generateDots(struct Master *this, Point *buffer, int n);
void saveResult(struct Master *this, double res);
double computeResult(struct Master *this);
void deleteMaster(struct Master *this);

#endif