import pandas as pd
import sys
import numpy as np

D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')
print(D)

# Convert to numpy datatype
D = D.to_numpy()

mean = np.matmul(np.transpose(D), np.ones())
