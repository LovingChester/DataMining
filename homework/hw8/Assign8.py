import pandas as pd
import sys
import numpy as np
from scipy.spatial import distance_matrix

FILENAME = sys.argv[1]
K = int(sys.argv[2])
n = int(sys.argv[3])
spread = float(sys.argv[4])
obj = sys.argv[5]

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')

D = D.to_numpy()

row = np.size(D, 0)

Dx = D[:, range(1, 27)]
Dy = D[:, [0]]

c0 = []
c1 = []
c2 = []
c3 = []

for i in range(row):
    if Dy[i, 0] <= 40:
        c0.append(i)
    elif Dy[i, 0] <= 60:
        c1.append(i)
    elif Dy[i, 0] <= 100:
        c2.append(i)
    else:
        c3.append(i)

classes = [c0, c1, c2, c3]
