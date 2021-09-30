import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.linalg import norm

K = np.array([[196,256,225,625],[256,576,676,1156],[225,676,900,1225],[625,1156,1225,2500]])

K_center = K - np.matmul(1/4*np.ones((4,4)), K) - np.matmul(K, 1/4*np.ones((4,4))) + np.matmul(np.matmul(1/16*np.ones((4,4)), K), np.ones((4,4)))

e_values, e_vectors = np.linalg.eigh(K_center)

print(e_values)

print(np.matmul(np.transpose(e_vectors[:,[3]]/math.sqrt(764.32)), K_center))

x = 191.08 / np.sum(e_values/4)
print(x)


y = np.array([[0.6,1.6,0.1],[-0.4,-1.4,0.1],[1.1,3.1,0.1]])

inv = np.linalg.inv(y)

print(inv)
z = np.matmul(inv, np.array([[-1],[0],[2]]))

print(z / np.linalg.norm(z))