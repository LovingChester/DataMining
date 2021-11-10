from numpy.core.fromnumeric import mean
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

np.set_printoptions(precision=3, suppress=False, threshold=5)

FILENAME = sys.argv[1]


def BAYESCLASSIFIER(Dx, classes):
    prior_prob = []
    means = []
    covs = []
    for item in classes:
        D = Dx[item, :]
        n = np.size(D, 0)  # cardinality
        P = n / 14735  # prior probability
        prior_prob.append(P)
        mean = np.mean(D, axis=0).reshape((-1, 1))
        means.append(mean)
        D_center = D - np.matmul(np.ones((n, 1)), np.transpose(mean))
        cov = np.matmul(np.transpose(D_center), D_center) / n
        covs.append(cov)
        
    return prior_prob, means, covs

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

classes = [c0, c1, c2, c3]
prior_prob, means, covs = BAYESCLASSIFIER(Dx, classes)

y_pred = []
for i in range(5000):
    y_hat = []
    for j in range(4):
        y = multivariate_normal.pdf(
            Dx_test[i], np.transpose(means[j]).reshape(26,), covs[j])
        y_hat.append(y)
    y_pred.append(y_hat.index(min(y_hat)))


