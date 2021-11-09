import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

np.set_printoptions(precision=3, suppress=False, threshold=5)

FILENAME = sys.argv[1]

D = pd.read_csv(FILENAME)
D.pop('date')
D.pop('rv2')

D = D.to_numpy()
