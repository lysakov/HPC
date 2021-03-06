#ifndef FUNCTIONS
#define FUNCTIONS

#include <cmath>

/** Defined on П = [-1, 2] x [-2, 2] */
inline double u(double x, double y)
{
    return exp(1 - (x + y)*(x + y));
}

inline double k(double x, double y)
{
    (void)y;
    return 4 + x;
}

inline double q(double x, double y)
{
    return (x + y)*(x + y);
}

/** F = -delta(u) + q*u */
inline double F(double x, double y)
{
    return (6*x + 2*y - 8*(x + 4)*(x + y)*(x + y) + (x + y)*(x + y) + 16)*exp(1 - (x + y)*(x + y));
}

/** phi_1(y) = u(-1, y) */
inline double phi_1(double y)
{
    return u(-1, y);
}

/** phi_2(y) = u(2, y) */
inline double phi_2(double y)
{
    return u(2, y);
}

/** phi_3(x) = u(x, -2) */
inline double phi_3(double x)
{
    return u(x, -2);
}

/** phi_1(x) = u(x, 2) */
inline double phi_4(double x)
{
    return u(x, 2);
}

inline void initDomain(double &x1, double &x2, double &y1, double &y2)
{
    x1 = -1.0;
    x2 = 2.0;
    y1 = -2.0;
    y2 = 2.0;
}

#endif