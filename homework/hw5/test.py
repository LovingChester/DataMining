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

clf = None
if KERNEL == 'linear':
    clf = make_pipeline(StandardScaler(), SVC(C=float(sys.argv[2]), kernel="linear"))
    clf.fit(Dx_train, np.reshape(Dy_train, (5000,)))
    print("sklearn validation accuracy:", 1 - np.count_nonzero(clf.predict(Dx_valid) - np.reshape(Dy_valid, (2000,))) / 2000)
else:
    clf = make_pipeline(StandardScaler(), SVC(C=float(sys.argv[2]), kernel="rbf", gamma=1/(2*KERNEL_PARAM**2)))
    clf.fit(Dx_train, np.reshape(Dy_train, (5000,)))
    print("sklearn validation accuracy:", 1 - np.count_nonzero(clf.predict(Dx_valid) - np.reshape(Dy_valid, (2000,))) / 2000)

print("sklearn test accuracy:", 1 - np.count_nonzero(clf.predict(Dx_test) - np.reshape(Dy_test, (5000,))) / 5000)
