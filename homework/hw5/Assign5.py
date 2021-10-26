import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import time
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=3, suppress=False, threshold=5)

FILENAME = "energydata_complete.csv"
C = 0.05
EPS = 0.001
MAXITER = 5000
KERNEL = "linear"
KERNEL_PARAM = 10

# SVM dual
def SVM_DUAL(Dx, Dy):
    # compute linear kernel matrix
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    K = []
    for i in range(row):
        K_row = []
        for j in range(row):
            if KERNEL == "linear":
                K_row.append(np.dot(Dx[i, :], Dx[j, :]))
            else:
                diff = Dx[i, :] - Dx[j, :]
                kernel = math.e ** (-np.dot(diff, diff) / (2 * KERNEL_PARAM))
                K_row.append(kernel)
        K.append(K_row)
    
    K = np.array(K)

    # augment K
    #K = np.insert(K, 0, row*[1], axis=1)
    K_aug = np.copy(K)
    for i in range(row):
        for j in range(row):
            K_aug[i, j] += 1

    # store the step size
    step_size = []
    for k in range(row):
        step_size.append(1 / K_aug[k, k])
    
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
    print(t)
    return alpha

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')
#print(D)

# define the training set, validation set and test_set
D = D.to_numpy()
scaler = StandardScaler()
Dx = D[:, range(1, 27)]
Dx = scaler.fit_transform(Dx)
Dy = D[:, [0]]

# Dx_train = Dx[range(0, 5000), :]
# Dx_valid = Dx[range(5000, 7000), :]
# Dx_test = Dx[range(7000, 12000), :]

# selecting the response and independent variables
Dx_train = Dx[range(0, 5000), :]
Dy_train = Dy[range(0, 5000), :]
for i in range(5000):
    if Dy_train[i, 0] <= 50:
        Dy_train[i, 0] = 1
    else:
        Dy_train[i, 0] = -1

Dx_valid = Dx[range(5000, 7000), :]
Dy_valid = Dy[range(5000, 7000), :]
for i in range(2000):
    if Dy_valid[i, 0] <= 50:
        Dy_valid[i, 0] = 1
    else:
        Dy_valid[i, 0] = -1

Dx_test = Dx[range(7000, 12000), :]
Dy_test = Dy[range(7000, 12000), :]
for i in range(5000):
    if Dy_test[i, 0] <= 50:
        Dy_test[i, 0] = 1
    else:
        Dy_test[i, 0] = -1

start = time.time()

alpha = SVM_DUAL(Dx_train, Dy_train)
# print(np.count_nonzero(alpha))
# start validation
#print(np.transpose(Dx_valid[[0],:]))
y_pred = []
for i in range(2000):
    point = Dx_valid[i, :]
    value = 0
    for j in range(5000):
        train_x = Dx_train[j, :]
        if KERNEL == 'linear':
            value += alpha[j, 0] * Dy_train[j, 0] * (np.dot(train_x, point) + 1)
        else:
            diff = train_x - point
            kernel = math.e ** (-np.dot(diff, diff) / (2 * KERNEL_PARAM)) + 1
            value += alpha[j, 0] * Dy_train[j, 0] * kernel
    y_pred.append(value)

count = 0
y_pred = np.sign(y_pred)
# for i in range(2000):
#     if y_pred[i] == 1:
#         count += 1
# print(count)
y_pred = y_pred.reshape(1, -1)
y_pred = np.transpose(y_pred)
print(1 - np.count_nonzero(y_pred-Dy_valid) / 2000)

end = time.time()
print(end - start)
