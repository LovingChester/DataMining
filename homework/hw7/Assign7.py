from numpy.ma import log
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
    for i in range(K-1):
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
    
    #print(centers)

    center_cluster = dict()
    for i in range(K):
        center_cluster[i] = []
    
    for i in range(row):
        distances = []
        for j in range(K):
            dist = np.linalg.norm(centers[j] - D[i])
            distances.append(dist)
        index = distances.index(min(distances))
        center_cluster[index].append(D[i])
    
    for i in range(K):
        center_cluster[i] = np.array(center_cluster[i])

    # initialize cov matrix
    covs = []
    for i in range(K):
        covs.append(np.cov(center_cluster[i], rowvar=False, bias=True))
    covs = np.array(covs)

    # initialize prior probability
    prob_Cs = []
    for i in range(K):
        size = np.size(center_cluster[i], 0)
        prob_Cs.append(size/row)

    # initial w where each entry is 0
    w = np.full((K, row), 0.0)

    while(t < MAXITER):
        t += 1

        prev_centers = np.copy(centers)

        # for j in range(row):
        #     log_ws = []
        #     for i in range(K):
        #         log_w = multivariate_normal.logpdf(D[j], centers[i], covs[i], allow_singular=True)
        #         log_w += np.log(prob_Cs[i])
        #         log_ws.append(log_w)
        #     log_sum_exp = logsumexp(log_ws)
        #     for i in range(K):
        #         w[i, j] = np.exp(log_ws[i] - log_sum_exp)
        
        w = []
        for i in range(K):
            w_row = multivariate_normal.logpdf(D, centers[i], covs[i], allow_singular=True)
            w.append(w_row)
        w = np.array(w)

        w = w - np.log(np.array(prob_Cs).reshape(-1, 1))

        log_sum_exp = logsumexp(w, axis=0)

        w = np.exp(w - log_sum_exp)

        # Maximization Step
        for i in range(K):
            center = np.matmul(np.transpose(D), np.transpose(w[[i], :])) / np.sum(w[i])
            centers[i] = np.transpose(center)

            D_center = np.copy(D)
            D_center = D_center - np.matmul(np.ones((row, 1)), np.transpose(center))

            cov = np.full((col, col), 0.0)
            for j in range(row):
                cov += w[i, j] * np.outer(np.transpose(D_center[[j], :]), D_center[[j], :])
            cov /= np.sum(w[i])

            covs[i] = cov

            prob_Cs[i] = np.sum(w[i]) / row

        diff = 0
        for i in range(K):
            diff += np.linalg.norm(centers[i] - prev_centers[i]) ** 2

        print("iter {}: {}".format(t, diff))
        if diff <= EPS:
            break

    return w, centers, covs

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')

D = D.to_numpy()

row = np.size(D, 0)

Dx = D[:, range(1, 27)]
Dy = D[:, [0]]

w, centers, covs = EXPECTATION_MAXIMIZATION(Dx)
#print(w)

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

classes = [c0, c1, c2, c3]

max_index = np.argmax(w, axis=0)

for i in range(K):
    print("final mean for cluster {}: {}".format(i, centers[i]))
    print("final covariance matrix for cluster {}: {}".format(i, covs[i]))
    print("size for cluster {}: {}".format(i, np.count_nonzero(max_index == i)))

# compute purity score
purity_score = 0
for i in range(K):
    sub_score = []
    for j in range(K):
        pred_index = np.where(max_index == i)
        pred_index = set(pred_index[0].tolist())
        intersect = pred_index & set(classes[j])
        sub_score.append(len(intersect))
    purity_score += max(sub_score)

print("The purity score is {:.3f}".format(purity_score / row))
