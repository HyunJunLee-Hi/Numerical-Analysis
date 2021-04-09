from random import *
import time

#주어진 방정식
def func(x):
    return (a*x**4) + (b*x**3) + (c*x**2) + (d*x) + e

a = float(5)
b = -22.4
c = 15.85272
d = 24.161472
e = -23.4824832

alpha = .1
min_y = 999999
min_x = 999999

#f'(x)를 계산하기 위한 함수
def der1(x):
    return (4*a*x**3) + (3*b*x**2) + (2*c*x) + d
#f''(x)를 계산하기 위한 함수
def der2(x):
    return (12*a*x**2) + (6*b*x) + (2*c)

t = time.time()
for i in range(100):
    Xi = randint(-7, 7)
    for j in range(999999):
        Xi_ = Xi - alpha*(der1(Xi)/der2(Xi))
        if der1(Xi_) < 0.0001 and der1(Xi_) > -0.0001:
            if der2(Xi_) > 0:
                if min_y > func(Xi_):
                    min_y = func(Xi_)
                    min_x = Xi_
                    print("min : " + str(min_y) + "  at  x = " + str(min_x))
                    
                break
        else:
            Xi = Xi_
print(str(time.time()-t) + "sec")







