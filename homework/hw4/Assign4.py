import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=3, suppress=False, threshold=5)

D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')
print(D)

D = D.to_numpy()
# scale the data
scaler = StandardScaler()
D = scaler.fit_transform(D)

# define the training set, validation set and test_set
D_train = D[range(0, 13735), :]
D_valid = D[range(13735, 15735), :]
D_test = D[range(15735, 19735), :]

# selecting the response and independent variables
Dx_train = D_train[:, range(1, 28)]
Dy_train = D_train[:, [0]]
Dx_valid = D_valid[:, range(1, 28)]
Dy_valid = D_valid[:, [0]]
Dx_test = D_test[:, range(1, 28)]
Dy_test = D_test[:, [0]]


