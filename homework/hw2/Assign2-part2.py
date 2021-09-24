import numpy as np
import matplotlib.pyplot as plt
import math

n = 100000

np.set_printoptions(threshold=5)

np.random.seed(10)
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

    plt.title("PMF for d = {}".format(d))
    plt.xlabel("angle")
    plt.ylabel("probability")
    plt.plot(unique_angle, angle_prob, 'b')
    plt.plot(unique_angle, angle_prob, 'rx')
    plt.show()

    print("min for d = {} is {}".format(d, min(unique_angle)))
    print("max for d = {} is {}".format(d, max(unique_angle)))
    print("range for d = {} is {}".format(d, max(unique_angle)-min(unique_angle)))
    print("mean for d = {} is {}".format(d, np.mean(np.array(angles))))
    print("variance for d = {} is {}".format(d, np.var(np.array(angles))))

PMF(10)
PMF(100)
PMF(1000)
