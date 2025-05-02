# Miscellaneous function in help to NTC simulation
import numpy as np

def curvature(x, y):
    # Compute the curvature of a curve defined by x and y
    yp = np.gradient(y, x)
    ypp = np.gradient(yp, x)
    return np.abs(ypp) / (1 + yp**2)**(3/2)

def curvature2(x, y):
    # Compute the first derivatives
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    yp = dy / dx
    # Compute the second derivatives
    dyp = np.diff(yp, prepend=yp[0])
    ypp = dyp / dx
    # Compute the curvature
    curvature = np.abs(ypp) / (1 + yp**2)**(3/2)
    return curvature

def fake_curvature(x, y):
    # Not a real curvature, but a measure of the slope of the curve
    yp = np.diff(y, axis=0) / np.diff(x, axis=0) #np.gradient(y, x)
    return yp

def multi_curvature(x, y):
    xb, yb = np.broadcast_arrays(x, y[:,np.newaxis])
    res = np.array([curvature2(xb[:,i], yb[:,i]) for i in range(yb.shape[1])])
    return res.T
