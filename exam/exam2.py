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
    w = np.array([[1],[0.5],[0.5]])
    #while t < maxiter:
    w_copy = np.copy(w)
        #for i in range(row):
    w_gradient = (Dy[0, 0] - expit(np.matmul(Dx[0, :], w_copy))) * np.transpose(Dx[0, :].reshape(1, -1))
    print(w_gradient)
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

def ridge(Dx, Dy, alpha):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    Dx = np.insert(Dx, 0, row*[1], axis=1)
    inv = np.linalg.pinv(np.matmul(np.transpose(Dx), Dx) + alpha * np.identity(col+1))
    Y = np.matmul(np.transpose(Dx), Dy)
    w = np.matmul(inv, Y)
    return w

Dx = np.array([[2,1],[1,2],[3,1],[2,2]])
Dy = np.array([[1],[0],[1],[0]])
w = Logistic_regression(Dx, Dy, 0, 1)
print(w)
# print(ridge(Dx, Dy, 0.5))


# print(" ")
# print(np.dot(np.array([-0.22,1.39,-0.83,0.78]), np.array([-0.22,1.39,-0.83,0.78])))
