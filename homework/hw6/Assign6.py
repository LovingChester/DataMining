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
        #cov = np.matmul(np.transpose(D_center), D_center) / n
        cov = np.cov(D, rowvar=False, bias=True)
        covs.append(cov)
        
    return prior_prob, means, covs


def NAIVEBAYES(Dx, classes):
    prior_prob = []
    means = []
    v = []
    for item in classes:
        D = Dx[item, :]
        n = np.size(D, 0)  # cardinality
        P = n / 14735  # prior probability
        prior_prob.append(P)
        mean = np.mean(D, axis=0).reshape((-1, 1))
        means.append(mean)
        D_center = D - np.matmul(np.ones((n, 1)), np.transpose(mean))
        vars = []
        for j in range(26):
            var = np.matmul(np.transpose(D_center[:, [j]]), D_center[:, [j]]) / n
            vars.append(var)
        vars = np.array(vars).reshape((-1, 1))
        v.append(vars)

    return prior_prob, means, v

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
prior_prob, means, covs = BAYESCLASSIFIER(Dx_train, classes)

y_pred = []
for i in range(5000):
    y_hat = []
    for j in range(4):
        y = multivariate_normal.pdf(
            Dx_test[i], np.transpose(means[j]).reshape(26,), covs[j]) * prior_prob[j]
        y_hat.append(y)
    y_pred.append(y_hat.index(max(y_hat)))

y_pred = np.array(y_pred).reshape((-1, 1))

c0_test = []
c1_test = []
c2_test = []
c3_test = []
Dy_test_class = []
for i in range(5000):
    if Dy_test[i, 0] <= 40:
        Dy_test_class.append(0)
        c0_test.append(i)
    elif Dy_test[i, 0] <= 60:
        Dy_test_class.append(1)
        c1_test.append(i)
    elif Dy_test[i, 0] <= 100:
        Dy_test_class.append(2)
        c2_test.append(i)
    else:
        Dy_test_class.append(3)
        c3_test.append(i)

Dy_test_class = np.array(Dy_test_class).reshape((-1, 1))

c0_pred = []
c1_pred = []
c2_pred = []
c3_pred = []
for i in range(5000):
    if y_pred[i, 0] == 0:
        c0_pred.append(i)
    elif y_pred[i, 0] == 1:
        c1_pred.append(i)
    elif y_pred[i, 0] == 2:
        c2_pred.append(i)
    else:
        c3_pred.append(i)

print("Total accuaracy for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred-Dy_test_class) / 5000))
print("class 0 recall for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c0_test, :]-Dy_test_class[c0_test, :]) / len(c0_test)))
print("class 1 recall for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c1_test, :]-Dy_test_class[c1_test, :]) / len(c1_test)))
print("class 2 recall for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c2_test, :]-Dy_test_class[c2_test, :]) / len(c2_test)))
print("class 3 recall for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c3_test, :]-Dy_test_class[c3_test, :]) / len(c3_test)))
print("class 0 specific accuracy for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c0_pred, :]-Dy_test_class[c0_pred, :]) / len(c0_pred)))
print("class 1 specific accuracy for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c1_pred, :]-Dy_test_class[c1_pred, :]) / len(c1_pred)))
print("class 2 specific accuracy for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c2_pred, :]-Dy_test_class[c2_pred, :]) / len(c2_pred)))
print("class 3 specific accuracy for full Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred[c3_pred, :]-Dy_test_class[c3_pred, :]) / len(c3_pred)))

# prior_prob, means, v = NAIVEBAYES(Dx_train, classes)

# y_pred = []
# for i in range(5000):
#     y_hat = []
#     for j in range(4):
#         y = 1
#         for k in range(26):
#             y *= multivariate_normal.pdf(Dx_test[i, k], means[j][k, 0], v[j][k, 0])
#         y *= prior_prob[j]
#         y_hat.append(y)
#     y_pred.append(y_hat.index(max(y_hat)))

# y_pred = np.array(y_pred).reshape((-1, 1))

# print("Total accuaracy for naive Bayes: {:.3f}".format(1 - np.count_nonzero(y_pred-Dy_test_class) / 5000))
