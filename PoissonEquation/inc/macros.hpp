#ifndef MACROS
#define MACROS

#define x(i) (domain.x1 + i*h1)
#define y(i) (domain.y1 + i*h2)
#define d_rx(w) (w[(i + 1)*(N + 1) + j] - w[i*(N + 1) + j])/h1
#define d_ry(w) (w[i*(N + 1) + j + 1] - w[i*(N + 1) + j])/h2
#define d_lx(w) (w[i*(N + 1) + j] - w[(i - 1)*(N + 1) + j])/h1
#define d_ly(w) (w[i*(N + 1) + j] - w[i*(N + 1) + j - 1])/h2
#define dx(w, k) (k(x(i) + 0.5*h1, y(j))*d_rx(w) - k(x(i) - 0.5*h1, y(j))*d_lx(w))/h1
#define dy(w, k) (k(x(i), y(j) + 0.5*h2)*d_ry(w) - k(x(i), y(j) - 0.5*h2)*d_ly(w))/h2

#endif