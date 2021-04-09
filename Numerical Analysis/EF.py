import os
import cv2
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
'''
def show(r, c, a, b):
    fig = plt.figure()
    rows = r
    cols = c
    case = 1
    for j in range(a, b):
           img = np.reshape((re2[j]), (32, 32))
           ax = fig.add_subplot(rows, cols, case)
           ax.imshow(img, cmap = 'gray')
           ax.set_xlabel(j+1)
           ax.set_xticks([]), ax.set_yticks([])
           case += 1

    plt.show()
'''
#train data
data = None
cnt = 0
for img in os.listdir('Test_img'):
    cnt += 1
    imgraw = cv2.imread(os.path.join('Test_img', img), cv2.IMREAD_GRAYSCALE)
    imgvector = imgraw.reshape(32*32)
    try:
        data = np.vstack((data, imgvector))
    except:
        data = imgvector

#mean, eigenvectors
cov, mean = cv2.calcCovarMatrix(data, mean=None, flags = cv2.COVAR_NORMAL+cv2.COVAR_ROWS)

A = data - mean


#Apply SVD
U, Sigma, VT = svd(A)

#pick eigenvectors
num_EV = 100
reducedEV = np.zeros((num_EV, 1024), dtype='float64')
for i in range(num_EV):
    reducedEV[i] = VT[i]
'''
#Show Eigenface
fig = plt.figure()
rows = 6
cols = 6
case = 1
for j in range(0, 36):
       img = np.reshape((VT[j]), (32, 32))
       ax = fig.add_subplot(rows, cols, case)
       ax.imshow(img, cmap = 'gray')
       ax.set_xlabel(j+1)
       ax.set_xticks([]), ax.set_yticks([])
       case += 1

plt.show()
'''

#test data
test_data = None
test_cnt = 0
for img2 in os.listdir('Test_img'):
    test_cnt += 1
    test_imgraw = cv2.imread(os.path.join('Test_img', img2), cv2.IMREAD_GRAYSCALE)
    test_imgvector = test_imgraw.reshape(32*32)
    try:
        test_data = np.vstack((test_data, test_imgvector))
    except:
        test_data = test_imgvector
'''
fig = plt.figure()
rows = 5
cols = 5
case = 1
for j in range(0, 25):
       img = np.reshape((test_data[j]), (32, 32))
       ax = fig.add_subplot(rows, cols, case)
       ax.imshow(img, cmap = 'gray')
       ax.set_xlabel(j+1)
       ax.set_xticks([]), ax.set_yticks([])
       case += 1

plt.show()

fig = plt.figure()
rows = 5
cols = 5
case = 1
for j in range(25, 50):
       img = np.reshape((test_data[j]), (32, 32))
       ax = fig.add_subplot(rows, cols, case)
       ax.imshow(img, cmap = 'gray')
       ax.set_xlabel(j+1)
       ax.set_xticks([]), ax.set_yticks([])
       case += 1

plt.show()
'''
#find test data coefficient
coe = (test_data - mean)@VT.T
coe2 = (test_data - mean)@reducedEV.T

#Reconstruction
re = np.zeros((test_cnt, 1024), dtype = 'float64')
for j in range(test_cnt):
    for i in range(1024):
        re[j] += coe[j][i]*VT[i].T

re2 = np.zeros((test_cnt, 1024), dtype = 'float64')
for j in range(test_cnt):
    for i in range(num_EV):
        re2[j] += coe2[j][i]*reducedEV[i].T

#scale up
re = re * (255/(np.max(re)-np.min(re)))
re = re + mean

re2 = re2 *(255/(np.max(re2)-np.min(re2)))
re2 = re2 + mean

'''
fig = plt.figure()
rows = 5
cols = 5
case = 1
for j in range(0, 25):
       img = np.reshape((re[j]), (32, 32))
       ax = fig.add_subplot(rows, cols, case)
       ax.imshow(img, cmap = 'gray')
       ax.set_xlabel(j+1)
       ax.set_xticks([]), ax.set_yticks([])
       case += 1

plt.show()
fig = plt.figure()
rows = 5
cols = 5
case = 1
for j in range(25, 50):
       img = np.reshape((re[j]), (32, 32))
       ax = fig.add_subplot(rows, cols, case)
       ax.imshow(img, cmap = 'gray')
       ax.set_xlabel(j+1)
       ax.set_xticks([]), ax.set_yticks([])
       case += 1
'''
plt.show()
fig = plt.figure()
rows = 5
cols = 5
case = 1
for j in range(0, 25):
       img = np.reshape((re2[j]), (32, 32))
       ax = fig.add_subplot(rows, cols, case)
       ax.imshow(img, cmap = 'gray')
       ax.set_xlabel(j+1)
       ax.set_xticks([]), ax.set_yticks([])
       case += 1

plt.show()
fig = plt.figure()
rows = 5
cols = 5
case = 1
for j in range(25, 50):
       img = np.reshape((re2[j]), (32, 32))
       ax = fig.add_subplot(rows, cols, case)
       ax.imshow(img, cmap = 'gray')
       ax.set_xlabel(j+1)
       ax.set_xticks([]), ax.set_yticks([])
       case += 1

plt.show()

'''
#recognition 1 : Distance for all coefficents

for j in range(0, 45, 5):
    res1=0
    res2=0
    res3=0
    res4=0
    res5=0
    print("--------------------")
    for i in range(num_EV):
        res1 += np.sqrt((coe2[j][i] - coe2[j+1][i])**2)
        res2 += np.sqrt((coe2[j+2][i] - coe2[j+3][i])**2)
        res3 += np.sqrt((coe2[j+4][i] - coe2[j][i])**2)
        res4 += np.sqrt((coe2[j][i] - coe2[j+7][i])**2)
        res5 += np.sqrt((coe2[j+4][i] - coe2[j+7][i])**2)
    print("Same")
    print(res1)
    print(res2)
    print(res3)
    print("Different")
    print(res4)
    print(res5)


print("----------Recognition----------")
print(" Input index ")
a = int(input("First picture : "))
b = int(input("Second picture : "))
testing = 0
for i in range(num_EV):
    testing += np.sqrt((coe2[a][i] - coe2[b][i])**2)
if testing > 4250:
    print("Different")
else:
    print("Same")
'''

#Recognition 2 : main coefficients
test_case = 0
for j in range(0, 50):
    res1=0
    for i in range(num_EV):
        if j%5 != 4:
            if np.sqrt((coe[j][i] - coe[j+1][i])**2) < 7:
                print(i)
    print("===========")
    '''
    for i in range(num_EV):
        if np.sqrt((coe[j][i] - coe[j+6][i])**2) < 5:
            print(i)
    '''


