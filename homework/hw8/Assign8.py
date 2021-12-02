import pandas as pd
import sys
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

#np.set_printoptions(precision=5, suppress=False, threshold=5)
np.random.seed(1314)

FILENAME = sys.argv[1]
K = int(sys.argv[2])
n = int(sys.argv[3])
spread = float(sys.argv[4])
obj = sys.argv[5]

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

    #print(e_vectors)

    row_square_sum = np.sqrt(np.sum(e_vectors ** 2, axis=1).reshape(-1, 1))
    Y = e_vectors / row_square_sum

    # apply K-means
    #center_cluster, center_indexs = K_MEANS(Y)
    center_cluster = dict()
    center_indexs = dict()
    for i in range(K):
        center_cluster[i] = []
        center_indexs[i] = []
    
    kmeans = KMeans(n_clusters=K).fit(Y)
    #print(kmeans.labels_)

    return kmeans.labels_

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')

D = D.to_numpy()

row = np.size(D, 0)

Dx = D[:, range(1, 27)]
Dy = D[:, [0]]
y_true = np.copy(Dy)

c0 = []
c1 = []
c2 = []
c3 = []
for i in range(row):
    if Dy[i, 0] <= 40:
        c0.append(i)
        y_true[i, 0] = 0
    elif Dy[i, 0] <= 60:
        c1.append(i)
        y_true[i, 0] = 1
    elif Dy[i, 0] <= 100:
        c2.append(i)
        y_true[i, 0] = 2
    else:
        c3.append(i)
        y_true[i, 0] = 3

classes = [c0, c1, c2, c3]

y_pred = None
index = None
sss = StratifiedShuffleSplit(train_size=n)
for train_index, test_index in sss.split(Dx, y_true):
    index = train_index
    X_train, X_test = Dx[train_index], Dx[test_index]
    y_pred = SPECTRAL_CLUSTERING(X_train)
    #print(center_cluster)
    break

y_true = np.transpose(y_true).reshape((row,)).astype(int)
y_true = y_true[index]

print("sk-learn F score: {}".format(sum(f1_score(y_true, y_pred, average=None))/4))

precs = []
for i in range(K):
    pred_num = np.count_nonzero(y_pred == i)
    print("cluster {}, size: {}".format(i, pred_num))
    corr_num = 0
    for j in range(n):
        if y_pred[j] == y_true[j] and y_true[j] == i:
            corr_num += 1
    precs.append(corr_num / pred_num)

recalls = []
for i in range(K):
    true_num = np.count_nonzero(y_true == i)
    corr_num = 0
    for j in range(n):
        if y_pred[j] == y_true[j] and y_true[j] == i:
            corr_num += 1
    recalls.append(corr_num / true_num)

F = 0
for i in range(K):
    F_i = (2 * precs[i] * recalls[i]) / (precs[i] + recalls[i])
    F += F_i

print("The F score is {}".format(F / K))
