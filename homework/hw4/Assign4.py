import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=3, suppress=False, threshold=5)

ALPHA = 400
ETA = 0.00001
EPS = 0.0001
MAXITER = 5000

D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')

D = D.to_numpy()
# scale the data
scaler = StandardScaler()
D = scaler.fit_transform(D)
Dy = D[:, [0]]

# define the training set, validation set and test_set
D_train = D[range(0, 13735), :]
D_valid = D[range(13735, 15735), :]
D_test = D[range(15735, 19735), :]

# selecting the response and independent variables
Dx_train = D_train[:, range(1, 27)]
Dy_train = D_train[:, [0]]
Dx_valid = D_valid[:, range(1, 27)]
Dy_valid = D_valid[:, [0]]
Dx_test = D_test[:, range(1, 27)]
Dy_test = D_test[:, [0]]

# start linear regression
Dx_train = np.insert(Dx_train, 0, 13735*[1], axis=1)
t = 0
w = np.ones((27, 1))
while(t < MAXITER):
    gradient = -np.matmul(np.transpose(Dx_train), Dy_train) + \
        np.matmul(np.matmul(np.transpose(Dx_train), Dx_train), w) + ALPHA * w
    w_prev = np.copy(w)
    w = w - ETA * gradient
    if np.linalg.norm(w - w_prev) <= EPS:
        break
    t += 1

print("The w is:\n{}".format(w))
#print(t)

# start validation
Dx_valid = np.insert(Dx_valid, 0, 2000*[1], axis=1)
SSE_valid = np.linalg.norm(np.matmul(Dx_valid, w) - Dy_valid) ** 2
print("The SSE for valid is: {}".format(int(SSE_valid)))

# start test
Dx_test = np.insert(Dx_test, 0, 4000*[1], axis=1)
SSE_test = np.linalg.norm(np.matmul(Dx_test, w) - Dy_test) ** 2
#print(SSE_test)
TSS = Dy - np.average(Dy) * np.ones((19735, 1))
TSS = np.sum(TSS * TSS)
#print(TSS_test)
R_square = (TSS - SSE_test) / TSS
print("The R square is: {}".format(R_square))
