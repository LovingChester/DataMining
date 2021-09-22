import numpy as np
import matplotlib.pyplot as plt
import math

n = 100000

np.set_printoptions(threshold=5)

#generate pairs of 10-dimensional vectors
def PMF(d):
    sample = np.random.choice([1,-1], size=(n,d,2))

    angles = []

    for i in range(n):
        pair = sample[i, :]
        angle = np.dot(pair[:, 0], pair[:, 1]) / \
            (np.linalg.norm(pair[:, 0]) * np.linalg.norm(pair[:, 1]))
        angle = math.degrees(math.acos(angle))
        angles.append(angle)

    unique_angle = sorted(list(set(angles)))
    angle_prob = []
    for i in range(len(unique_angle)):
        angle_prob.append(angles.count(unique_angle[i]) / n)

    plt.plot(unique_angle, angle_prob, 'b')
    plt.plot(unique_angle, angle_prob, 'rx')
    plt.show()


PMF(10)
PMF(100)
PMF(1000)
