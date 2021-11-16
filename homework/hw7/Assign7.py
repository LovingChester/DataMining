import pandas as pd
import sys
import numpy as np
from scipy.stats import multivariate_normal

np.set_printoptions(precision=3, suppress=False, threshold=5)

FILENAME = sys.argv[1]
k = sys.argv[2]
EPS = sys.argv[3]
RIDGE = sys.argv[4]
MAXITER = sys.argv[5]

def EXPECTATION_MAXIMIZATION(D):
    row = np.size(D, 0)
    t = 0
    # initialize centers
    index = np.random.choice(row, 1)
    centers = D[index, :]
    # compute rest of the centers
    for i in range(3):
        size = np.size(centers, 0)
        dists = []
        for j in range(row):
            distances = []
            for k in range(size):
                dist = np.linalg.norm(centers[k] - D[j])
                distances.append(dist)
            dists.append(min(distances))
        index = dists.index(max(dists))
        centers = np.append(centers, D[[index], :], axis=0)
    
    # initialize cov matrix
    covs = []
    for i in range(k):
        covs.append(np.identity(row))
    covs = np.array(covs)

    # initialize prior probability
    prob_Cs = []
    for i in range(k):
        prob_Cs.append(1/k)
    
    # initial w where each entry is 0
    w = np.full((k, row), 0)

    # compute prob sum for on each point
    prob_sum = []
    for i in range(row):
        total = 0
        for j in range(k):
            total += multivariate_normal.logpdf(D[i], centers[j], covs[j], allow_singular=True)
        prob_sum.append(total)

    while(True):
        t += 1
        # Expection Step
        for i in range(k):
            for j in range(row):
                w[i, j] = multivariate_normal.logpdf(D[j], centers[i], covs[i], allow_singular=True) / prob_sum[j]
        
        # Maximization Step
        for i in range(k):
            break
        break

    return

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')
print(D)

D = D.to_numpy()

row = np.size(D, 0)

Dx = D[:, range(1, 27)]
Dy = D[:, [0]]

EXPECTATION_MAXIMIZATION(Dx)

c0 = []
c1 = []
c2 = []
c3 = []

for i in range(row):
    if Dy[i, 0] <= 40:
        c0.append(i)
    elif Dy[i, 0] <= 60:
        c1.append(i)
    elif Dy[i, 0] <= 100:
        c2.append(i)
    else:
        c3.append(i)


