import pandas as pd
import sys
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.model_selection import StratifiedShuffleSplit

np.set_printoptions(precision=3, suppress=False, threshold=5)
np.random.seed(10)

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

    row_square_sum = np.sqrt(np.sum(e_vectors ** 2, axis=1).reshape(-1, 1))

    Y = e_vectors / row_square_sum

    return

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

sss = StratifiedShuffleSplit(train_size=1000)
for train_index, test_index in sss.split(Dx, Dy_cluster):
    X_train, X_test = Dx[train_index], Dx[test_index]
    SPECTRAL_CLUSTERING(X_train)
    break

