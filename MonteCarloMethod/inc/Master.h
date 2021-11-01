#ifndef MASTER
#define MASTER

#include "geometry.h"

struct Master;

struct Master* createMaster(const Domain domain);
void generateDots(struct Master *this, Point *buffer, int n);
int getPoinsNumber(struct Master *this);
void deleteMaster(struct Master *this);

#endif