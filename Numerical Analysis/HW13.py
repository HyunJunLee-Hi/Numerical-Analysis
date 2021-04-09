import numpy as np
import random

min_err = 9999999999
while 1:
    #Generate 12 samples
    x = np.arange(-5,7)

    #Adding Gaussian noise for each y = 2x-1
    y = 2*x-1 + np.random.normal(0,np.sqrt(2),12)

    #Randomly select 6 samples from 12 points
    idx = random.sample(list(np.arange(0,12)), 6)


    #Compare the solution of RANSAC with the solution of while 12 samples
    x_whole = [[0,1] for i in range(12)]
    y_whole = [[0] for i in range(12)]
    for i in range(12):
        x_whole[i][0] = x[i]
        y_whole[i][0] = y[i]
    x_whole = np.array(x_whole, dtype='float32')
    y_whole = np.array(y_whole, dtype='float32')

    x_whole_inv = np.linalg.pinv(x_whole)
    coef_whole = x_whole_inv@y_whole


    #Find the fitted line using 6 samples and least square
    x_ = [[0,1] for i in range(6)]
    y_ = [[0] for i in range(6)]
    j = 0
    for i in idx:
        x_[j][0] = x[i]
        y_[j][0] = y[i]
        j += 1
    x_ = np.array(x_, dtype='float32')
    y_ = np.array(y_, dtype='float32')

    x_inv = np.linalg.pinv(x_)
    coef = x_inv@y_

    #Calculate the error
    error = 0
    for i in range(6):
        error += abs(y_[i][0] - (coef[0][0]*x_[i][0] + coef[1][0]))

    #Compare the current error with previous error
    if error < min_err:
        min_err = error

    #Repeat 1~4 until your criterion
    if min_err < 0.3:
        print("Whole samples")
        print("y = %f * x %f" % (coef_whole[0][0], coef_whole[1][0]))
        print("Select 6 samples")
        print("y = %f * x %f" % (coef[0][0], coef[1][0]))
        break
    
    
    
    
