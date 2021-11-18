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

# compute pdf manually
def compute_pdf(x, center, cov):
    col = np.size(cov, 1)
    first = 1 / (np.sqrt(2*np.pi) ** col) * np.sqrt(np.linalg.det(cov + RIDGE*np.identity(col)))
    upper = np.matmul((x-center).reshape(1, -1), np.linalg.inv(cov + RIDGE*np.identity(col)))
    upper = np.matmul(upper, (x-center).reshape(-1, 1))
    upper = upper[0][0]
    second = -upper / 2
    return first, second

def compute_log_sum(D, centers, covs, prob_Cs):
    log_sum = []
    for i in range(row):
        data = []
        for j in range(K):
            log_w = multivariate_normal.logpdf(D[i], centers[j], covs[j], allow_singular=True)
            # first, second = compute_pdf(D[i], centers[j], covs[j])
            # log_w = np.log(first) + second
            log_w += np.log(prob_Cs[j])
            data.append(log_w)
        max_num = max(data)
        print(data)
        data = np.array(data) - max_num
        log_sum.append(logsumexp(data)+max_num)

    return log_sum

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

    while(t < MAXITER):
        t += 1

        prev_centers = np.copy(centers)

        # compute the logsumexp
        log_sum = compute_log_sum(D, centers, covs, prob_Cs)

        # Expection Step
        for i in range(K):
            for j in range(row):
                log_w = multivariate_normal.logpdf(D[j], centers[i], covs[i], allow_singular=True)
                # first, second = compute_pdf(D[j], centers[i], covs[i])
                # log_w = np.log(first) + second
                log_w += np.log(prob_Cs[i])

                w[i, j] = np.exp(log_w - log_sum[j])
        print(w)
        # Maximization Step
        for i in range(K):
            center = np.matmul(np.transpose(D), np.transpose(w[[i], :])) / np.matmul(w[[i], :], np.ones((row, 1)))
            centers[i] = center.reshape((col,))
            
            center_D = np.copy(D)
            #print(centers[i,:])
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

