import numpy as np
from scipy.stats import multivariate_normal

#multivariate_normal.pdf()

D1 = np.array([[3],[5],[7]])
D2 = np.array([[4],[8],[7]])

# cov = np.cov(D, rowvar=False, bias=True)
# print(cov)

y = multivariate_normal.pdf(D1, 5, 2.67) * multivariate_normal.pdf(D2, 19/3, 2.89) * (1/2)
print(y)

D = np.array([[3,4],[5,4],[5,8],[7,5],[7,7]])
w = np.array([[0.9],[0.5],[0],[0],[0.9]])
mu = np.matmul(np.transpose(D), w) / np.sum(w)
#print(mu)
#print(D - np.transpose(mu))

D_center = D - np.transpose(mu)
#print(D_center)
sum_ = 0.9 * np.outer(np.array([[-2],[-1.17]]), np.array([-2, -1.17])) + 0.5 * np.outer(np.array([[0],[-1.17]]), np.array([0, -1.17])) + \
    0.9 * np.outer(np.array([[2],[1.83]]), np.array([2, 1.83]))
#print(sum_/2.3)


e_vectors = np.array([[0.5, 0.66],[0.5,0.29],[0.5,-0.34],[0.5,-0.6]])
row_square_sum = np.sqrt(np.sum(e_vectors ** 2, axis=1).reshape(-1, 1))
Y = e_vectors / row_square_sum
#print(Y)
