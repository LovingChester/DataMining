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

# Compute mean for each attribute
mean = np.matmul(np.transpose(D), np.ones((row, 1))) / row
print("This is the mean vector μ:\n{}".format(mean))

# Compute total variance
var_D = sum(sum(D*D)) / row - np.linalg.norm(mean) ** 2
print("This is the total variance: {}".format(var_D))

# Compute the center data matrix
D_center = D - np.matmul(np.ones((row, 1)), np.transpose(mean))

# Compute the sample covariance inner product form
D_var_inner = np.matmul(np.transpose(D_center), D_center) / row
print("This is the sample convariance in inner product form{}".format(D_var_inner))

# Compute the sample covariance outer product form
#print(np.outer(np.array([[1],[2],[3],[4]]), np.array([1,2,3])))
# D_var_outer = np.zeros((row, row))
# D_center_T = np.transpose(D_center)
# for i in range(row):
#     D_var_outer = D_var_outer + np.outer(D_center[ :,i], D_center_T[i, :])
# print(D_var_outer)

# Compute the correlation matrix
print(np.size(D_var_inner,1))
D_var_inner_row = np.size(D_var_inner,0)
correlate_matrix = np.zeros((D_var_inner_row, D_var_inner_row))
#print(correlate_matrix)
for i in range(D_var_inner_row):
    for j in range(D_var_inner_row):
        correlate_matrix[i,j] = D_var_inner[i,j] / math.sqrt(D_var_inner[i,i]*D_var_inner[j,j])

print('This is the correlation_matrix: {}'.format(correlate_matrix))

# Convert the correlation matrix to the degree
correlate_matrix_degree = np.degrees(np.arccos(correlate_matrix))
print(np.degrees(np.arccos(correlate_matrix)))

most_correlated = 99999
most_correlated_coord = None
least_correlated = -99999
least_correlated_coord = None
anti_correlated = -99999
anti_correlated_coord = None
for i in range(D_var_inner_row):
    for j in range(D_var_inner_row):
        if correlate_matrix_degree[i, j] < most_correlated and correlate_matrix_degree[i, j] != 0:
            most_correlated = correlate_matrix_degree[i, j]
            most_correlated_coord = (i, j)
        if correlate_matrix_degree[i, j] > least_correlated and correlate_matrix_degree[i, j] <= 90:
            least_correlated = correlate_matrix_degree[i, j]
            least_correlated_coord = (i, j)
        if correlate_matrix_degree[i, j] > anti_correlated:
            anti_correlated = correlate_matrix_degree[i, j]
            anti_correlated_coord = (i, j)

print(most_correlated_coord, least_correlated_coord, anti_correlated_coord)

max_chart_size = max(np.max(D[:, most_correlated_coord[0]]),
                 np.max(D[:, most_correlated_coord[1]]))
print(max_chart_size)
min_chart_size = min(np.min(D[:, most_correlated_coord[0]]),
                     np.min(D[:, most_correlated_coord[1]]))

# plt.axis([int(min_chart_size), math.ceil(max_chart_size),
#          int(min_chart_size), math.ceil(max_chart_size)])
plt.plot(D[:, most_correlated_coord[0]], D[:, most_correlated_coord[1]], 'bx')
plt.show()

plt.plot(D[:, least_correlated_coord[0]], D[:, least_correlated_coord[1]], 'bx')
plt.show()

plt.plot(D[:, anti_correlated_coord[0]],
         D[:, anti_correlated_coord[1]], 'bx')
plt.show()


print(anti_correlated)
