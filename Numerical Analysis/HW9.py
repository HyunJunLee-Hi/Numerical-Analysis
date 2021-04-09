import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.image as mpimg
import scipy.fftpack

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc
import matplotlib.pylab as pylab

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def dct_(a):
    imsize = a.shape
    dct = np.zeros(imsize)
    sort_dct = np.zeros(imsize)

    for i in r_[:imsize[0]:16]:
        for j in r_[:imsize[1]:16]:
            dct[i:(i+16),j:(j+16)] = dct2( a[i:(i+16),j:(j+16)] )

    pick = []
    for i in r_[:imsize[0]:16]:
        for j in r_[:imsize[1]:16]:
            sort_dct = []
            tmp = []
            for k in range(0, 16):
                for t in range(0, 16):
                    sort_dct.append(abs(dct[i+k, j+t]))
            tmp = sorted(sort_dct, reverse=True)
            pick.append(tmp[15])


    cnt = 0
    for i in r_[:imsize[0]:16]:
        for j in r_[:imsize[1]:16]:
            for k in range(0, 16):
                for t in range(0, 16):
                    if abs(dct[i+k, j+t]) < pick[cnt]:
                        dct[i+k, j+t] = 0
            cnt += 1

    im_dct = np.zeros(imsize)

    for i in r_[:imsize[0]:16]:
        for j in r_[:imsize[1]:16]:
            im_dct[i:(i+16),j:(j+16)] = idct2( dct[i:(i+16),j:(j+16)] )

    
    for i in range(imsize[0]):
        for j in range(imsize[1]):
            if im_dct[i][j] > 255:
                im_dct[i][j] = 255
            elif im_dct[i][j] < 0:
                im_dct[i][j] = 0

    return im_dct


im = cv2.imread('test.png', cv2.IMREAD_COLOR)
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()
b, g, r = cv2.split(im)



b_dct = dct_(b)
g_dct = dct_(g)
r_dct = dct_(r)

res = cv2.merge((b_dct,g_dct,r_dct))
plt.figure()
plt.imshow(cv2.cvtColor(np.uint8(res), cv2.COLOR_BGR2RGB))
plt.show()
