import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def corr_coef(a, b):
    imsize_ = a.shape
    imsize = imsize_[0]*imsize_[1]
    a_ = a.reshape(imsize)
    b_ = b.reshape(imsize)

    avg = np.ones(imsize)
    avg_a = avg * np.mean(a_)
    avg_b = avg * np.mean(b_)

    E = (a_-avg_a) * (b_-avg_b)
    Exp = sum(E)
    sigma = np.std(a_) * np.std(b_)

    return Exp/(imsize*sigma)

fig = plt.figure()
rows = 3
cols = 4
i = 1

for img in os.listdir('test'):
    check = cv2.imread(os.path.join('test', img), cv2.IMREAD_COLOR)
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(cv2.cvtColor(check, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(i)
    ax.set_xticks([]), ax.set_yticks([])
    i += 1
plt.show()


cnt = 1
    
for img in os.listdir('test'):
    img_raw = cv2.imread(os.path.join('test', img), cv2.IMREAD_COLOR)
    #RGB image
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    #YUV channel
    img_yuv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2YUV)

    #split
    b,g,r = cv2.split(img_raw)
    y,u,v = cv2.split(img_yuv)

    #Show R, G, B channel
    r_ = img_rgb.copy()
    r_[:,:,1] = 0
    r_[:,:,2] = 0
    g_ = img_rgb.copy()
    g_[:,:,0] = 0
    g_[:,:,2] = 0    
    b_ = img_rgb.copy()
    b_[:,:,0] = 0
    b_[:,:,1] = 0

    fig = plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 4, 2)
    plt.imshow(r_)
    plt.subplot(1, 4, 3)
    plt.imshow(g_)
    plt.subplot(1, 4, 4)
    plt.imshow(b_)
    
    #Show V, Y, U channel
    fig = plt.figure()    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 4, 2)
    plt.imshow(v, cmap = 'gray')
    plt.subplot(1, 4, 3)
    plt.imshow(y, cmap = 'gray')
    plt.subplot(1, 4, 4)
    plt.imshow(u, cmap = 'gray')
    plt.show()


    print("--------------------", end = '')
    print(cnt, end = '')
    cnt += 1
    print(" image--------------------")
    
    #correlation coefficients
    print("G-R in RGB space : ", end = '')
    print(corr_coef(g, r))
    print("G-B in RGB space : ", end = '')
    print(corr_coef(g, b))
    print("R-B in RGB space : ", end = '')
    print(corr_coef(r, b))
    print("Y-U in YUV space : ", end = '')
    print(corr_coef(y, u))
    print("Y-V in YUV space : ", end = '')
    print(corr_coef(y, v))
    print("U-V in YUV space : ", end = '')
    print(corr_coef(u, v))
    print("Y-G : ", end = '')
    print(corr_coef(y, g))
    print()
    print()






    
