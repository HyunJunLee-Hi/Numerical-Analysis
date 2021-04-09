import os
import cv2
from numpy import dot
from numpy.linalg import norm
import numpy as np
from matplotlib import pyplot as plt
import random

def cos_sim(a, b):
       return dot(a, b)/(norm(a)*norm(b))

def dft_(a):

    img = imgs[a]
    
    #Randomly selected blocks in the fabric images
    row = random.randint(0, img.shape[0] - 65)
    column = random.randint(0, img.shape[1] - 65)

    test_img = img[row:row+64, column:column+64]

    '''
    plt.figure()
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.show()
    '''
    
    dft = cv2.dft(np.float32(test_img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    '''
    plt.subplot(121),plt.imshow(test_img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    '''
    
    select_mag = magnitude_spectrum[33:64, 33:64]
    total_mag.append(select_mag)

    '''
    pick = []
    sort_mag = []
    tmp = []
    for k in range(0, 31):
        for t in range(0, 31):
            tmp.append(abs(select_mag[k, t]))
    sort_mag = sorted(tmp, reverse=True)
    pick.append(sort_mag[10])


    cnt = 0
    for k in range(0, 31):
        for t in range(0, 31):
            if abs(select_mag[k, t]) < pick[cnt]:
                select_mag[k, t] = 0
    cnt += 1

    
    total_mag.append(select_mag)
    '''

    
#fig = plt.figure()
rows = 4
cols = 5
i = 1

imgs = []
for img in os.listdir('output'):
    imgs.append(cv2.imread(os.path.join('output', img), cv2.IMREAD_GRAYSCALE))
    '''
    check = cv2.imread(os.path.join('output', img), cv2.IMREAD_COLOR)
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(cv2.cvtColor(check, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(i)
    ax.set_xticks([]), ax.set_yticks([])
    i += 1
    '''
#plt.show()


total_mag = []

for i in range(20):
    dft_(i)

'''
for k in range(1, 20):
    res = 0
    for i in range(31):
        for j in range(31):
            res += (np.sqrt((total_mag[0][i][j]-total_mag[k][i][j])**2))
    print(res)

recog_distance = 2700
avg_res = 0

for k in range(5):
    for s in range(20):
        portion = 0
        testing = 50
        for t in range(testing):
            recog_idx = random.randint(0, 19)
            distance = 0
            for i in range(31):
                for j in range(31):
                    distance += (np.sqrt((total_mag[s][i][j]-total_mag[recog_idx][i][j])**2))
            #오류 날 확률
            if distance < recog_distance:
                if s != recog_idx:
                    portion += 1
            else:
                if s == recog_idx:
                    portion += 1

        avg_res += portion/testing*100
        #print(portion/testing*100)
print("Average Recognition = ", end = " ")
print(100-(avg_res/100), end = " ")
print("%")


#Threshold
total_mag = np.array(total_mag)
threshold = 0.02
mag_threshold = total_mag * (abs(total_mag) > (threshold*np.max(total_mag)))


for k in range(1, 20):
    res = 0
    for i in range(31):
        for j in range(31):
            res += (np.sqrt((mag_threshold[0][i][j]-mag_threshold[k][i][j])**2))
    print(res)

recog_distance = 16000
avg_res = 0
for k in range(5):
    for s in range(20):
        portion = 0
        testing = 50
        for t in range(testing):
            recog_idx = random.randint(0, 19)
            distance = 0
            for i in range(31):
                for j in range(31):
                    distance += (np.sqrt((mag_threshold[s][i][j]-mag_threshold[recog_idx][i][j])**2))
            #오류 날 확률
            if distance < recog_distance:
                if s != recog_idx:
                    portion += 1
            else:
                if s == recog_idx:
                    portion += 1

        avg_res += portion/testing*100
        print(portion/testing*100)
print("Average Recognition = ", end = " ")
print(100-(avg_res/100), end = " ")
print("%")
'''


#Cosine Similarity
for k in range(1, 20):
    res += cos_sim(total_mag[0], total_mag[k])
    print(res)

total_mag = np.array(total_mag)
recog_distance = 2700
avg_res = 0

for k in range(5):
    for s in range(20):
        portion = 0
        testing = 50
        for t in range(testing):
            recog_idx = random.randint(0, 19)
            distance = 0
            for i in range(31):
                for j in range(31):
                    distance += cos_sim(total_mag[s][i][j], total_mag[recog_idx][i][j])
            #오류 날 확률
            if distance < recog_distance:
                if s != recog_idx:
                    portion += 1
            else:
                if s == recog_idx:
                    portion += 1

        avg_res += portion/testing*100
        #print(portion/testing*100)
print("Average Recognition = ", end = " ")
print(100-(avg_res/100), end = " ")
print("%")











