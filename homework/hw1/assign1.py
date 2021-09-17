import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

filename = sys.argv[1]

EPS = float(sys.argv[2])

D = pd.read_csv(filename)
D.pop('date')
D.pop('rv2')
print(D)

# Convert to numpy datatype
D = D.to_numpy()
row = np.size(D, 0)
col = np.size(D, 1)
#print(np.sum(D[:,1])/row)
np.set_printoptions(precision=3, suppress=False, threshold=5)
#np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Compute mean for each attribute
mean = np.matmul(np.transpose(D), np.ones((row, 1))) / row
print("This is the mean vector μ:\n{}\n".format(mean))

# Compute total variance
var_D = sum(sum(D*D)) / row - np.linalg.norm(mean) ** 2
print("This is the total variance:\n{:.3f}\n".format(var_D))

# Compute the center data matrix
D_center = D - np.matmul(np.ones((row, 1)), np.transpose(mean))
#print(np.matmul(np.ones((row, 1)), np.transpose(mean)))
# Compute the sample covariance inner product form
D_var_inner = np.matmul(np.transpose(D_center), D_center) / row
print("This is the sample convariance in inner product form:\n{}\n".format(D_var_inner))

D_var_outer = np.zeros((col, col))
#D_center_T = np.transpose(D_center)
for i in range(row):
    D_var_outer = D_var_outer + np.outer(D_center[i, :].reshape(col,1), D_center[i, :])
D_var_outer = D_var_outer / row
print("This is the sample convariance in outer product form:\n{}\n".format(D_var_outer))

# Compute the correlation matrix
#print(np.size(D_var_inner,1))
#D_var_inner_row = np.size(D_var_inner,0)
correlate_matrix = np.zeros((col, col))
#print(correlate_matrix)
for i in range(col):
    for j in range(col):
        correlate_matrix[i,j] = D_var_inner[i,j] / math.sqrt(D_var_inner[i,i]*D_var_inner[j,j])

print('This is the correlation_matrix:\n{}\n'.format(correlate_matrix))

# Convert the correlation matrix to the degree
correlate_matrix_degree = np.degrees(np.arccos(correlate_matrix))
#print(np.degrees(np.arccos(correlate_matrix)))

most_correlated = 99999
most_correlated_coord = None
least_correlated = -99999
least_correlated_coord = None
anti_correlated = -99999
anti_correlated_coord = None
for i in range(col):
    for j in range(col):
        if correlate_matrix_degree[i, j] < most_correlated and correlate_matrix_degree[i, j] != 0:
            most_correlated = correlate_matrix_degree[i, j]
            most_correlated_coord = (i, j)
        if correlate_matrix_degree[i, j] > least_correlated and correlate_matrix_degree[i, j] <= 90:
            least_correlated = correlate_matrix_degree[i, j]
            least_correlated_coord = (i, j)
        if correlate_matrix_degree[i, j] > anti_correlated:
            anti_correlated = correlate_matrix_degree[i, j]
            anti_correlated_coord = (i, j)

#print(most_correlated_coord, least_correlated_coord, anti_correlated_coord)

max_chart_size = max(np.max(D[:, most_correlated_coord[0]]),
                 np.max(D[:, most_correlated_coord[1]]))
# print(max_chart_size)
min_chart_size = min(np.min(D[:, most_correlated_coord[0]]),
                     np.min(D[:, most_correlated_coord[1]]))

# plt.axis([int(min_chart_size), math.ceil(max_chart_size),
#          int(min_chart_size), math.ceil(max_chart_size)])
plt.title("most correlated")
plt.xlabel('A ' + str(most_correlated_coord[0]+1))
plt.ylabel('A ' + str(most_correlated_coord[1]+1))
plt.plot(D[:, most_correlated_coord[0]], D[:, most_correlated_coord[1]], 'bx')
plt.show()

plt.title("least correlated")
plt.xlabel('A ' + str(least_correlated_coord[0]+1))
plt.ylabel('A ' + str(least_correlated_coord[1]+1))
plt.plot(D[:, least_correlated_coord[0]], D[:, least_correlated_coord[1]], 'bx')
plt.show()

plt.title("anti correlated")
plt.xlabel('A ' + str(anti_correlated_coord[0]+1))
plt.ylabel('A ' + str(anti_correlated_coord[1]+1))
plt.plot(D[:, anti_correlated_coord[0]],
         D[:, anti_correlated_coord[1]], 'bx')
plt.show()

# print(anti_correlated)
#np.random.seed(13)
x_init = np.random.normal(size=(col, 1))
x_new = None
while(True):
    x_new = np.matmul(D_var_inner,x_init)
    if np.max(np.abs(x_new)) == np.max(x_new):
        x_scaled = x_new / np.max(x_new)
    else:
        x_scaled = x_new / (-np.max(np.abs(x_new)))
    if np.linalg.norm(x_scaled - x_init) < EPS:
        break
    x_init = x_scaled

#print(x_new)
x_normalize = x_new / np.linalg.norm(x_new)
#print(x_normalize)

e_value = np.max(np.abs(x_new))
e_vector = x_normalize
print('This is eigenvalue:\n{:.3f}\n'.format(e_value))
print('This is domiant eigenvector:\n{}\n'.format(e_vector))
#domiant_e_vector = D_var_inner - np.outer(x_normalize, np.transpose(x_normalize))
#print(np.linalg.eig(D_var_inner))

proj_u1 = np.matmul(D_center, e_vector)
#print(proj_u1)
plt.plot(proj_u1, np.array(row*[0]), 'bx')
plt.show()
