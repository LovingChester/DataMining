import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

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

# Compute the sample covariance inner product form
D_var_inner = np.matmul(np.transpose(D_center), D_center) / row

e_values, e_vectors = np.linalg.eigh(D_var_inner)

print(e_vectors)


