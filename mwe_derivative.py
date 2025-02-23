import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

def myder(x, y):
    yp = np.gradient(y, x)
    ypp = np.gradient(yp, x)
    return yp, ypp

def myder2(x, y):
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    yp = dy / dx
    dyp = np.diff(yp, prepend=yp[0])
    ypp = dyp / dx
    return yp, ypp  

def myder3(x, y):
    dx = x[1] - x[0]
    y1 = np.concatenate(([y[0]], y))
    y2 = np.concatenate(([y[0], y[0]], y))
    yp = (y1[1:] - y1[:-1]) / dx
    ypp = (y2[2:] - 2*y2[1:-1] + y2[:-2]) / dx**2
    return yp, ypp

x, y = sym.symbols('x y')
y = sym.sin(x) #-x**3 + x**2 + 2*x + 1
yp = sym.diff(y, x)
ypp = sym.diff(yp, x)

f = sym.lambdify(x, y)
fp = sym.lambdify(x, yp)
fpp = sym.lambdify(x, ypp)
xs = np.linspace(1, 2, 101, endpoint=True)
# xs = np.logspace(0, np.log10(2), 101, endpoint=True)

yp, ypp = myder(xs, f(xs))
yp2, ypp2 = myder2(xs, f(xs))
yp3, ypp3 = myder3(xs, f(xs))

ax1 = plt.subplot(311)
plt.plot(xs, f(xs))
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(xs, fp(xs), xs, yp, xs, yp2, xs, yp3)
ax3 = plt.subplot(313, sharex=ax1)
plt.plot(xs, fpp(xs), xs, ypp, xs, ypp2, xs, ypp3)
plt.grid()
plt.legend(["sympy", "np.gradient", "np.diff", "manual"])
plt.show()