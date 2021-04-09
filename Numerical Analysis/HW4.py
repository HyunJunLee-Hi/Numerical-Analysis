import random
import numpy as np
'''
X = [[0, 0, 1] for i in range(6)]
Y = [[0] for i in range(6)]
data = [[-2.9, 35.4], [-2.1, 19.7], [-0.9, 5.7], [1.1, 2.1], [0.1, 1.2], [1.9, 8,7], [3.1, 25.7], [4.0, 41.5]]

for i in range(2):
    print("----------Test " + str(i) + "----------")
    sample_data = random.sample(data, 6)
    for j in range(6):
        X[j][0] = sample_data[j][0]**2
        X[j][1] = sample_data[j][0]
        Y[j][0] = sample_data[j][1]
    print(Y)
'''

X = np.array([[0, 0, 1],
             [0, 0, 1],
             [0, 0, 1],
             [0, 0, 1],
             [0, 0, 1],
             [0, 0, 1]])
Y = np.array([[0],
              [0],
              [0],
              [0],
              [0],
              [0]])
data = np.array([[-2.9, 35.4],
                 [-2.1, 19.7],
                 [-0.9, 5.7],
                 [1.1, 2.1],
                 [0.1, 1.2],
                 [1.9, 8,7],
                 [3.1, 25.7],
                 [4.0, 41.5]])

for i in range(2):
    print("----------Test " + str(i) + "----------")
    sample_data = np.random.choice(data, 6)
    for j in range(6):
        X[j , 0] = sample_data[j , 0]*sample_data[j , 0]
        X[j , 1] = sample_data[j , 0]
        Y[j , 0] = sample_data[j , 1]
    print(Y)
