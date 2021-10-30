#include "Master.h"

#include <stdlib.h>
#include <time.h>

struct Master
{

    double result;
    int dotsGenerated;
    Domain domain;

};

static double getRandomDouble(double lowerBound, double upperBound)
{

    double number = (double)rand() / RAND_MAX;

    return lowerBound + number * (upperBound - lowerBound);

}

struct Master* createMaster(const Domain domain)
{

    srand(time(NULL));
    struct Master *master = (struct Master*)malloc(sizeof(struct Master));
    master->result = 0;
    master->domain = domain;
    master->dotsGenerated = 0;

    return master;

}

Point* generateDots(struct Master *this, Point *buffer, int n)
{

    for (int i = 0; i < n; ++i) {
        Point dot;
        dot.x = getRandomDouble(this->domain.x1, this->domain.x2);
        dot.y = getRandomDouble(this->domain.y1, this->domain.y2);
        dot.z = getRandomDouble(this->domain.z1, this->domain.z2);
        buffer[i] = dot;
    }
    this->dotsGenerated += n;

    return buffer;

}

void saveResult(struct Master *this, double res)
{

    this->result += res;

}

double computeResult(struct Master *this)
{

    double res = (this->domain.x2 - this->domain.x1) * 
        (this->domain.y2 - this->domain.y1) *
        (this->domain.z2 - this->domain.z1) *
        this->result / this->dotsGenerated;

    return res;

}

void deleteMaster(struct Master *this)
{

    free(this);

}