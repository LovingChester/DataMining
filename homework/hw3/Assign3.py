import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

D = pd.read_csv('energydata_complete.csv')
D.pop('date')
D.pop('rv2')


