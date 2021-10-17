import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=3, suppress=False, threshold=5)

D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')
print(D)

D = D.to_numpy()
# scale the data
scaler = StandardScaler()
D = scaler.fit_transform(D)

# define the training set, validation set and test_set
D_train = D[range(0, 13735), :]
D_valid = D[range(13735, 15735), :]
D_test = D[range(15735, 19735), :]


