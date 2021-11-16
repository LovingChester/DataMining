import pandas as pd
import sys
import numpy as np

FILENAME = sys.argv[1]
k = sys.argv[2]
EPS = sys.argv[3]
RIDGE = sys.argv[4]
MAXITER = sys.argv[5]

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')
print(D)


