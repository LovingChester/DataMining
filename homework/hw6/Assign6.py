import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

np.set_printoptions(precision=3, suppress=False, threshold=5)

FILENAME = sys.argv[1]

def BAYESCLASSIFIER(Dx, Dy):

    return

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')

D = D.to_numpy()

Dx = D[:, range(1, 27)]
Dy = D[:, [0]]

Dx_train = Dx[range(0, 14735), :]
Dy_train = Dy[range(0, 14735), :]

Dx_test = Dx[range(14735, 19735), :]
Dy_test = Dy[range(14735, 19735), :]

c0 = []
c1 = []
c2 = []
c3 = []

for i in range(14735):
    if Dy_train[i, 0] <= 40:
        c0.append(i)
    elif Dy_train[i, 0] <= 60:
        c1.append(i)
    elif Dy_train[i, 0] <= 100:
        c2.append(i)
    else:
        c3.append(i)


