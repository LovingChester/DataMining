import pandas as pd
import sys
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

np.set_printoptions(precision=3, suppress=False, threshold=5)

FILENAME = sys.argv[1]
K = int(sys.argv[2])
EPS = float(sys.argv[3])
RIDGE = float(sys.argv[4])
MAXITER = int(sys.argv[5])

def EXPECTATION_MAXIMIZATION(D):
    row, col = np.size(D, 0), np.size(D, 1)
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
    for i in range(K):
        covs.append(np.identity(col))
    covs = np.array(covs)

    # initialize prior probability
    prob_Cs = []
    for i in range(K):
        prob_Cs.append(1/K)
    
    # initial w where each entry is 0
    w = np.full((K, row), 0)
    # compute prob sum for on each point
    # prob_sum = []
    # for i in range(row):
    #     total = 0
    #     for j in range(K):
    #         total += multivariate_normal.logpdf(D[i], centers[j], covs[j], allow_singular=True)
    #         total += multivariate_normal.logpdf(prob_Cs[j], centers[j], covs[j], allow_singular=True)
    #     prob_sum.append(total)

    # compute the logsumexp
    log_sum = []
    for i in range(row):
        data = []
        for j in range(K):
            pdf = multivariate_normal.logpdf(D[i], centers[j], covs[j], allow_singular=True) * prob_Cs[j]
            pdf += np.log(prob_Cs[j])
            #pdf += multivariate_normal.logpdf(prob_Cs[j], centers[j], covs[j], allow_singular=True)
            data.append(pdf)
        log_sum.append(logsumexp(data))

    while(t < MAXITER):
        t += 1

        prev_centers = np.copy(centers)

        # Expection Step
        for i in range(K):
            for j in range(row):
                log_w = multivariate_normal.logpdf(D[j], centers[i], covs[i], allow_singular=True) * prob_Cs[i]
                log_w += np.log(prob_Cs[i])
                #log_w += multivariate_normal.logpdf(prob_Cs[i], centers[i], covs[i], allow_singular=True)
                # if t == 2:
                #     print(log_w)
                #     print(log_sum[j])
                #     print(np.exp(log_w - log_sum[j]))
                w[i, j] = np.exp(log_w - log_sum[j])
        #print(w)
        # Maximization Step
        for i in range(K):
            center = np.matmul(np.transpose(D), np.transpose(w[[i], :])) / np.matmul(w[[i], :], np.ones((row, 1)))
            centers[i] = center.reshape((col,))
            
            center_D = np.copy(D)
            print(centers[i,:])
            center_D = center_D - np.matmul(np.ones((row, 1)), centers[[i], :])

            cov_sum = np.zeros((col, col))
            for j in range(row):
                cov_sum += w[i, j] * np.outer(np.transpose(center_D[[j], :]), center_D[[j], :])
            
            covs[i] = cov_sum / np.matmul(w[[i], :], np.ones((row, 1)))

            prob = np.matmul(w[i, :], np.ones((row, 1))) / row
            prob_Cs[i] = prob[0]

        diff = 0
        for i in range(K):
            diff += np.linalg.norm(centers[i] - prev_centers[i]) ** 2

        if diff <= EPS:
            break

    return t

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')
print(D)

D = D.to_numpy()

row = np.size(D, 0)

Dx = D[:, range(1, 27)]
Dy = D[:, [0]]

t = EXPECTATION_MAXIMIZATION(Dx)
print(t)

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


