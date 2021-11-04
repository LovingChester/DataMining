import numpy as np
import math
from numpy.lib.function_base import gradient
from scipy.special import expit
from scipy.spatial import distance_matrix
import random as rd

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


def SVM_DUAL(Dx, Dy, KERNEL, MAXITER, C, EPS, KERNEL_PARAM):
    # compute linear kernel matrix
    row, col = np.size(Dx, 0), np.size(Dx, 1)

    K = []
    if KERNEL == "linear":
        K = np.matmul(Dx, np.transpose(Dx))
    else:
        K = np.full((row, row), math.e)
        dist_matrix = distance_matrix(Dx, Dx)
        gaussian = -(dist_matrix ** 2) / (2 * KERNEL_PARAM ** 2)
        K = K ** gaussian

    # augment K
    K_aug = np.copy(K)
    K_aug = K_aug + np.ones((row, row))

    # store the step size
    step_size = []
    step_size = np.reciprocal(np.diag(K_aug))

    t = 0
    alpha = np.zeros((row, 1))
    while t < MAXITER:
        alpha_next = np.copy(alpha)
        l = list(range(row))
        rd.shuffle(l)
        for k in l:
            alpha_next[k, 0] = alpha_next[k, 0] + step_size[k] * (1 -
                                                                  Dy[k, 0] * np.sum(alpha_next * Dy * K_aug[:, [k]]))

            if alpha_next[k, 0] < 0:
                alpha_next[k, 0] = 0
            if alpha_next[k, 0] > C:
                alpha_next[k, 0] = C

        diff = np.linalg.norm(alpha_next - alpha)
        if diff <= EPS:
            break
        alpha = alpha_next
        #print(t, diff)
        t = t + 1
    print("iteration time: {}".format(t))
    return alpha

Dx = np.array([[1,2],[3,4],[5,6]])
Dy = np.array([[1],[-1],[1]])
w = Logistic_regression(Dx, Dy, 3, 0.5)
print(w)
