import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

alpha = 0.975

D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')
print(D)

# Convert to numpy datatype
D = D.to_numpy()
row = np.size(D, 0)
col = np.size(D, 1)

np.set_printoptions(precision=3, suppress=False, threshold=5)

# Compute mean for each attribute
mean = np.matmul(np.transpose(D), np.ones((row, 1))) / row
print("This is the mean vector μ:\n{}\n".format(mean))

# Compute total variance
var_D = sum(sum(D*D)) / row - np.linalg.norm(mean) ** 2
print("This is the total variance:\n{:.3f}\n".format(var_D))

# Compute the center data matrix
D_center = D - np.matmul(np.ones((row, 1)), np.transpose(mean))

# # Compute the sample covariance inner product form
# cov = np.matmul(np.transpose(D_center), D_center) / row

# Compute the sample covariance by np.cov
cov = np.cov(D, rowvar=False, bias=True)

e_values, e_vectors = np.linalg.eigh(cov)

print(e_values[col:])

print(e_vectors)

# Part I: Principal Components Analysis
r = 1
while(True):
    frac = sum(e_values[col-r:]) / sum(e_values)
    if r == 3:
        MSE = sum(e_values) - sum(e_values[col-r:])
        print("MSE is {:.3f}".format(MSE))
    if frac >= alpha:
        break
    r += 1

print("Dimensions required: {}".format(r))

# First PC
proj_u1 = np.matmul(D_center, e_vectors[:, [col-1]])
plt.plot(proj_u1, np.array(row*[0]), 'bx')
plt.show()

# Second PC
proj_u2 = np.matmul(D_center, e_vectors[:, [col-1, col-2]])
plt.plot(proj_u2[:, [0,0]], proj_u2[:, [1,1]], 'bx')
plt.show()

