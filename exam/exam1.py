import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

EPS = 0.001

np.set_printoptions(precision=3, suppress=False, threshold=5)

# Compute everything except and power iteration
def compute(D):
    row = np.size(D, 0)
    col = np.size(D, 1)

    # Compute mean for each attribute
    mean = np.matmul(np.transpose(D), np.ones((row, 1))) / row
    print("This is the mean vector μ:\n{}\n".format(mean))

    # Compute total variance
    var_D = sum(sum(D*D)) / row - np.linalg.norm(mean) ** 2
    print("This is the total variance:\n{:.3f}\n".format(var_D))

    # Compute the center data matrix
    D_center = D - np.matmul(np.ones((row, 1)), np.transpose(mean))
    print("This is the centered D:\n{}\n".format(D_center))

    # # Compute the sample covariance inner product form
    # D_var_inner = np.matmul(np.transpose(D_center), D_center) / row
    # print("This is the sample convariance in inner product form:\n{}\n".format(D_var_inner))
    # #print(np.cov(D, rowvar=False, bias=True))
    # D_var_outer = np.zeros((col, col))
    # #D_center_T = np.transpose(D_center)
    # for i in range(row):
    #     D_var_outer = D_var_outer + \
    #         np.outer(D_center[i, :].reshape(col, 1), D_center[i, :])
    # D_var_outer = D_var_outer / row
    # print("This is the sample convariance in outer product form:\n{}\n".format(D_var_outer))

    # Compute the sample covariance by np.cov
    cov = np.cov(D, rowvar=False, bias=True)
    print("This is the sample convariance :\n{}\n".format(cov))

    # Compute the correlation matrix
    correlate_matrix = np.zeros((col, col))
    for i in range(col):
        for j in range(col):
            correlate_matrix[i, j] = cov[i, j] / \
                math.sqrt(cov[i, i]*cov[j, j])

    print('This is the correlation_matrix:\n{}\n'.format(correlate_matrix))

    correlate_matrix_degree = np.degrees(np.arccos(correlate_matrix))
    print('degree correlation:\n{}\n'.format(correlate_matrix_degree))

#compute power iteration
def power_iteration(D):
    row = np.size(D, 0)
    col = np.size(D, 1)
    cov = np.cov(D, rowvar=False, bias=True)
    x_init = np.random.normal(size=(col, 1))
    x_new = None
    iteration = 1
    while(True):
        print("iteration {}".format(iteration))
        x_new = np.matmul(cov, x_init)
        print("unscaled: {}".format(x_new))
        if np.max(np.abs(x_new)) == np.max(x_new):
            x_scaled = x_new / np.max(x_new)
        else:
            x_scaled = x_new / (-np.max(np.abs(x_new)))
        print("scaled: {}".format(x_new))
        if np.linalg.norm(x_scaled - x_init) < EPS:
            break
        x_init = x_scaled
        iteration += 1
    
    x_normalize = x_new / np.linalg.norm(x_new)
    e_value = np.max(np.abs(x_new))
    e_vector = x_normalize
    print('This is eigenvalue:\n{:.3f}\n'.format(e_value))
    print('This is domiant eigenvector:\n{}\n'.format(e_vector))


D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')
D = D.to_numpy()

compute(D)

power_iteration(D)
