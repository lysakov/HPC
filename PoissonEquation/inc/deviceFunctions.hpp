#ifndef DEVICE_FUNCTIONS
#define DEVICE_FUNCTIONS

#include <cmath>
#include <cuda_runtime.h>

/** Defined on П = [-1, 2] x [-2, 2] */
__device__ inline double d_u(double x, double y)
{
    return exp(1 - (x + y)*(x + y));
}

__device__  inline double d_k(double x, double y)
{
    (void)y;
    return 4 + x;
}

__device__ inline double d_q(double x, double y)
{
    return (x + y)*(x + y);
}

/** F = -delta(u) + q*u */
__device__ inline double d_F(double x, double y)
{
    return (6*x + 2*y - 8*(x + 4)*(x + y)*(x + y) + (x + y)*(x + y) + 16)*exp(1 - (x + y)*(x + y));
}

/** phi_1(y) = u(-1, y) */
__device__ inline double d_phi_1(double y)
{
    return d_u(-1, y);
}

/** phi_2(y) = u(2, y) */
__device__ inline double d_phi_2(double y)
{
    return d_u(2, y);
}

/** phi_3(x) = u(x, -2) */
__device__ inline double d_phi_3(double x)
{
    return d_u(x, -2);
}

/** phi_1(x) = u(x, 2) */
__device__ inline double d_phi_4(double x)
{
    return d_u(x, 2);
}

#endif
