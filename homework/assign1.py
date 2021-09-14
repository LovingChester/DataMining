import pandas as pd
import sys
import numpy as np

D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')
print(D)

# Convert to numpy datatype
D = D.to_numpy()
row = np.size(D, 0)

# Compute mean for each attribute
mean = np.matmul(np.transpose(D), np.ones((row, 1))) / row
print("This is the mean vector μ:\n{}".format(mean))

# Compute

# Compute the center data matrix
D_center = D - np.matmul(np.ones((row, 1)), np.transpose(mean))

# Compute the sample covariance inner product form
D_var = np.matmul(np.transpose(D_center), D_center) / row
print(D_var)
