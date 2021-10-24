import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(precision=3, suppress=False, threshold=5)

FILENAME = "energydata_complete.csv"
C = 0
EPS = 0.0001
MAXITER = 5000
KERNEL = "linear"
KERNEL_PARAM = 10

# SVM dual under linear kernel
def SVM_DUAL_LINEAR(Dx, Dy):
    # compute linear kernel matrix
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    K = []
    for i in range(row):
        K_row = []
        for j in range(row):
            K_row.append(np.dot(Dx[i, :], Dx[j, :]) ** 2)
        K.append(K_row)
    
    K = np.array(K)

    # augment K
    K = np.insert(K, 0, row*[1], axis=1)

    # store the step size
    step_size = []
    for k in range(row):
        step_size.append(1 / K[k, k])
    
    t = 0
    alpha = np.zeros((row, 1))
    while t < MAXITER:
        alpha_prev = np.copy(alpha)
        for k in range(row):
            
            return

    return

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')
#print(D)

# define the training set, validation set and test_set
D = D.to_numpy()
D_train = D[range(0, 5000), :]
D_valid = D[range(5000, 7000), :]
D_test = D[range(7000, 12000), :]

# selecting the response and independent variables
Dx_train = D_train[:, range(1, 27)]
Dy_train = D_train[:, [0]]
Dx_valid = D_valid[:, range(1, 27)]
Dy_valid = D_valid[:, [0]]
Dx_test = D_test[:, range(1, 27)]
Dy_test = D_test[:, [0]]

#SVM_DUAL_LINEAR(Dx_train, Dy_train)
