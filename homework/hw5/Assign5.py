import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

np.set_printoptions(precision=3, suppress=False, threshold=5)
rd.seed(10)

FILENAME = sys.argv[1]
# 0.05
C = float(sys.argv[2])
# 0.01
EPS = float(sys.argv[3])
# 5000
MAXITER = int(sys.argv[4])
# linear gaussian
KERNEL = sys.argv[5]
# 65
KERNEL_PARAM = float(sys.argv[6])

# SVM dual
def SVM_DUAL(Dx, Dy):
    # compute linear kernel matrix
    row, col = np.size(Dx, 0), np.size(Dx, 1)

    K = []
    if KERNEL == "linear":
        K = np.matmul(Dx, np.transpose(Dx))
    else:
        for i in range(row):
            K_row = []
            for j in range(row):
                diff = Dx[i, :] - Dx[j, :]
                kernel = math.e ** (-np.dot(diff, diff) / (2 * KERNEL_PARAM ** 2))
                K_row.append(kernel)
            K.append(K_row)
        K = np.array(K)

    # augment K
    K_aug = np.copy(K)
    K_aug = K_aug + np.ones((row, row))

    # store the step size
    step_size = []
    step_size = np.reciprocal(np.diag(K_aug))
    # for k in range(row):
    #     step_size.append(1 / K_aug[k, k])
    
    t = 0
    alpha = np.zeros((row, 1))
    while t < MAXITER:
        alpha_next = np.copy(alpha)
        l = list(range(row))
        rd.shuffle(l)
        for k in l:
            alpha_next[k, 0] = alpha_next[k, 0] + step_size[k] * (1 - \
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

print("Running {} kernel: ".format(KERNEL))
alpha = SVM_DUAL(Dx_train, Dy_train)
#sup_vec_num = np.count_nonzero(alpha)
sup_vec_num = 0
for i in range(5000):
    if alpha[i, 0] > 0 and alpha[i, 0] < C:
        sup_vec_num += 1
print("Number of support vectors for {} kernel: {}".format(KERNEL, sup_vec_num))

# start validation
y_pred = []
if KERNEL == 'linear':
    K = np.matmul(Dx_valid, np.transpose(Dx_train))
    K = K + np.ones((2000, 5000))
    alpha_y = alpha * Dy_train
    y_pred = alpha_y * np.transpose(K)
    y_pred = np.sum(y_pred, axis=0)
else:
    for i in range(2000):
        point = Dx_valid[i, :]
        value = 0
        for j in range(5000):
            train_x = Dx_train[j, :]
            diff = train_x - point
            kernel = math.e ** (-np.dot(diff, diff) / (2 * KERNEL_PARAM ** 2)) + 1
            value += alpha[j, 0] * Dy_train[j, 0] * kernel
        y_pred.append(value)

y_pred = np.sign(y_pred)
y_pred = y_pred.reshape(1, -1)
y_pred = np.transpose(y_pred)
valid_accuracy = 1 - np.count_nonzero(y_pred-Dy_valid) / 2000
print("Validation accuracy: {:.3f}".format(valid_accuracy))

clf = None
if KERNEL == 'linear':
    clf = make_pipeline(StandardScaler(), SVC(C=float(sys.argv[2]), kernel="linear"))
    clf.fit(Dx_train, np.reshape(Dy_train, (5000,)))
    print("sklearn validation accuracy:", 1 - np.count_nonzero(clf.predict(Dx_valid) - np.reshape(Dy_valid, (2000,))) / 2000)
else:
    clf = make_pipeline(StandardScaler(), SVC(C=float(sys.argv[2]), kernel="rbf", gamma=1/(2*KERNEL_PARAM**2)))
    clf.fit(Dx_train, np.reshape(Dy_train, (5000,)))
    print("sklearn validation accuracy:", 1 - np.count_nonzero(clf.predict(Dx_valid) - np.reshape(Dy_valid, (2000,))) / 2000)

# start test
y_pred = []
if KERNEL == 'linear':
    K = np.matmul(Dx_test, np.transpose(Dx_train))
    K = K + np.ones((5000, 5000))
    alpha_y = alpha * Dy_train
    y_pred = alpha_y * np.transpose(K)
    y_pred = np.sum(y_pred, axis=0)
else:
    for i in range(5000):
        point = Dx_test[i, :]
        value = 0
        for j in range(5000):
            train_x = Dx_train[j, :]
            diff = train_x - point
            kernel = math.e ** (-np.dot(diff, diff) / (2 * KERNEL_PARAM)) + 1
            value += alpha[j, 0] * Dy_train[j, 0] * kernel
        y_pred.append(value)

y_pred = np.sign(y_pred)
y_pred = y_pred.reshape(1, -1)
y_pred = np.transpose(y_pred)
test_accuracy = 1 - np.count_nonzero(y_pred-Dy_test) / 5000
print("Test accuracy: {:.3f}".format(test_accuracy))

print("sklearn test accuracy:", 1 - np.count_nonzero(clf.predict(Dx_test) - np.reshape(Dy_test, (5000,))) / 5000)

# get w and b for linear kernel
# if KERNEL == 'linear':
#     w = np.zeros((1, 26))
#     for i in range(5000):
#         w += alpha[i, 0] * Dy_train[i, 0] * Dx_train[[i], :]

#     b = np.average(Dy_train - np.matmul(Dx_train, np.transpose(w)))
#     print("w is {}".format(w))
#     print("b is {:.3f}".format(b))

end = time.time()
print("running time: {}".format(end - start))
