from os import replace
import pandas as pd
import sys
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.model_selection import StratifiedShuffleSplit

np.set_printoptions(precision=5, suppress=False, threshold=5)
np.random.seed(10)

FILENAME = sys.argv[1]
K = int(sys.argv[2])
n = int(sys.argv[3])
spread = float(sys.argv[4])
obj = sys.argv[5]

def K_MEANS(D):
    t = 0
    indexs = np.random.choice(n, 4, replace=False)
    centers = D[indexs, :]
    center_cluster = None
    center_indexs = None
    while True:
        prev_centers = np.copy(centers)

        center_cluster = dict()
        center_indexs = dict()
        for i in range(K):
            center_cluster[i] = []
            center_indexs[i] = []
        
        point_dist = distance_matrix(D, centers)
        min_dist = np.argmin(point_dist, axis=1)

        for i in range(n):
            center_cluster[min_dist[i]].append(D[i])
            center_indexs[min_dist[i]].append(i)
        
        # update centers
        for i in range(K):
            centers[i] = sum(center_cluster[i]) / len(center_cluster[i])

        diff = np.sum(np.linalg.norm(prev_centers - centers, axis=0) ** 2)

        if diff <= 0.001:
            break
        #print(t)
        t += 1

    return center_cluster, center_indexs

def SPECTRAL_CLUSTERING(D):
    # compute similarity matrix
    A = distance_matrix(D, D)
    A = np.exp(-(A ** 2) / (2 * spread ** 2))

    degree_matrix = np.diag(np.sum(A, axis=0))
    L = degree_matrix - A

    L_s_tmp = 1 / np.sqrt(np.sum(A, axis=0))
    L_s = np.matmul(np.matmul(np.diag(L_s_tmp), L), np.diag(L_s_tmp))
    
    L_a_tmp = 1 / np.sum(A, axis=0)
    L_a = np.matmul(np.diag(L_a_tmp), L)

    B = None
    if obj == 'ratio':
        B = L
    elif obj == 'symmetric':
        B = L_s
    elif obj == 'asymmetric':
        B = L_a
    
    e_values, e_vectors = None, None
    if obj == 'ratio' or obj == 'symmetric':
        e_values, e_vectors = np.linalg.eigh(B)
    else:
        e_values, e_vectors = np.linalg.eig(B)
        e_v_sort = np.argsort(e_values)
        e_values = np.sort(e_values)
        e_vectors = e_vectors[:, e_v_sort]

    e_values = e_values[: K]
    e_vectors = e_vectors[:, range(K)]

    row_square_sum = np.sqrt(np.sum(e_vectors ** 2, axis=1).reshape(-1, 1))
    Y = e_vectors / row_square_sum

    # apply K-means
    center_cluster = K_MEANS(Y)

    return center_cluster

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')

D = D.to_numpy()

row = np.size(D, 0)

Dx = D[:, range(1, 27)]
Dy = D[:, [0]]
Dy_cluster = np.copy(Dy)

c0 = []
c1 = []
c2 = []
c3 = []

for i in range(row):
    if Dy[i, 0] <= 40:
        c0.append(i)
        Dy_cluster[i, 0] = 0
    elif Dy[i, 0] <= 60:
        c1.append(i)
        Dy_cluster[i, 0] = 1
    elif Dy[i, 0] <= 100:
        c2.append(i)
        Dy_cluster[i, 0] = 2
    else:
        c3.append(i)
        Dy_cluster[i, 0] = 3

classes = [c0, c1, c2, c3]

center_cluster = None
center_indexs = None
sss = StratifiedShuffleSplit(train_size=n)
for train_index, test_index in sss.split(Dx, Dy_cluster):
    X_train, X_test = Dx[train_index], Dx[test_index]
    center_cluster, center_indexs = SPECTRAL_CLUSTERING(X_train)
    #print(center_cluster)
    break

precs = []
for i in range(K):
    sub_score = []
    for j in range(K):
        score = len(set(center_indexs[i]) & set(classes[j]))
        sub_score.append(score)
    prec = max(sub_score) / len(center_cluster[i])
    precs.append(prec)

recalls = []
for i in range(K):
    sub_score = []
    for j in range(K):
        score = len(set(center_indexs[i]) & set(classes[j]))
        sub_score.append(score)
    recall = max(sub_score)
    frac = len(set(center_indexs[i]) & set(classes[j])) / len(center_cluster[i])
    recall /= frac
    recalls.append(recall)

F = 0
for i in range(K):
    F_i = (2 * precs[i] * recalls[i]) / (precs[i] + recalls[i])
    F += F_i

print("The F score is {}".format(F / K))
