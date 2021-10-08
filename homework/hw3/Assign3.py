from numpy.core.numeric import ones
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

filename = sys.argv[1]

sigma = float(sys.argv[2])

D = pd.read_csv(filename)
D.pop('date')
D.pop('rv2')
#print(D)

np.set_printoptions(precision=3, suppress=False, threshold=5)

# Convert to numpy datatype
D = D.to_numpy()
row = np.size(D, 0)
col = np.size(D, 1)

def KERNEL_DISCRIMINANT(rm_first, sigma):
    rm_row = np.size(rm_first, 0)
    rm_col = np.size(rm_first, 1)
    # kernel matrix
    K = []
    for i in range(rm_row):
        K_row = []
        for j in range(rm_row):
            diff = rm_first[i,:] - rm_first[j,:]
            kernel = math.e ** (-np.dot(diff, diff) / (2 * sigma))
            K_row.append(kernel)
        K.append(K_row)

    K = np.array(K)
    appliances_col = D[range(0,1000),:][:,0]
    #print(appliances_col[0])
    c1 = []
    c2 = []
    for i in range(np.size(appliances_col, 0)):
        if appliances_col[i] <= 50:
            c1.append(i)
        else:
            c2.append(i)
    
    # compute class kernel matrix
    K_1 = K[:,c1]
    K_1_row, K_1_col = np.size(K_1, 0), np.size(K_1, 1)
    K_2 = K[:,c2]
    K_2_row, K_2_col = np.size(K_2, 0), np.size(K_2, 1)

    # compute class means
    m_1 = np.matmul(K_1, np.ones((K_1_col, 1))) / K_1_col
    m_2 = np.matmul(K_2, np.ones((K_2_col, 1))) / K_2_col

    # between-class scatter matrix
    M = np.outer((m_1 - m_2), np.transpose(m_1 - m_2))

    #print(np.size(M,0), np.size(M,1))

    # class scatter matrices
    N_1 = np.matmul(np.matmul(K_1, np.identity(K_1_col)-np.ones((K_1_col, K_1_col))/K_1_col), np.transpose(K_1))
    N_2 = np.matmul(np.matmul(K_2, np.identity(K_2_col)-np.ones((K_2_col, K_2_col))/K_2_col), np.transpose(K_2))

    # within-class scatter matrix
    N = N_1 + N_2
    #print(np.size(N,0), np.size(N,1))

    term = np.matmul(np.linalg.pinv(N), M)
    e_values, e_vectors = np.linalg.eig(term)

    a = e_vectors[:, [0]]
    lamb = e_values[0]
    a = np.real(a)
    term = np.matmul(np.matmul(np.transpose(a), K), a)
    a = a / math.sqrt(term)

    return K, K_1, K_2, lamb, a, appliances_col

'''
he matrix used to compute kernel
remove the first column since it
will determine the class
'''
rm_first = D[:, range(1,27)]
rm_first = rm_first[range(0,1000),:]
#print(rm_first)

K, K_1, K_2, lamb, a, appliances_col = KERNEL_DISCRIMINANT(rm_first, sigma)
# print(K)
print("a:", a)
# print(np.size(a, 0))
proj = np.matmul(K, a)
tmp = np.reshape(proj, (1000))
c1 = []
c2 = []
for i in range(1000):
    if appliances_col[i] <= 50:
        c1.append(proj[i])
    else:
        c2.append(proj[i])

plt.plot(c1, np.array(len(c1)*[0]), 'bo')
plt.plot(c2, np.array(len(c2)*[0]), 'rx')
plt.show()
# sigma = 0.001
# while(sigma <= 100):
    
#     sigma *= 10
