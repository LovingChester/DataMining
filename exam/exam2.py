import numpy as np
import math
from numpy.lib.function_base import gradient
from scipy.special import expit

def Logistic_regression(Dx, Dy, maxiter, rate):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    Dx = np.insert(Dx, 0, row*[1], axis=1)
    t = 0
    w = np.zeros((col+1, 1))
    while t < maxiter:
        w_copy = np.copy(w)
        for i in range(row):
            w_gradient = (Dy[i, 0] - expit(np.matmul(Dx[i, :], w_copy))) * np.transpose(Dx[i, :].reshape(1, -1))
            w_copy = w_copy + rate * w_gradient
        
        w = w_copy
        print(w)
        t = t + 1
    
    return w

Dx = np.array([[1,2],[3,4],[5,6]])
Dy = np.array([[1],[-1],[1]])
w = Logistic_regression(Dx, Dy, 3, 0.5)
print(w)
