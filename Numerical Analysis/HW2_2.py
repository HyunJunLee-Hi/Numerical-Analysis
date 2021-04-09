from random import *
import time

#주어진 방정
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

h = .001


for i in range(100):
    Xi = randint(-7, 7)
    t = time.time()
    for j in range(999999):
        #Using approximation
        d1 = (func(Xi + h) - func(Xi)) / h
        d2 = (func(Xi + h) - 2*func(Xi) + func(Xi - h)) / (h*h)
        Xi_ = Xi - alpha*(d1/d2)
        if d1 < 0.0001 and d1 > -0.0001:
            if d2 > 0:
                if min_y > func(Xi_):
                    min_y = func(Xi_)
                    min_x = Xi_
                    print("min : " + str(min_y) + "  at  x = " + str(min_x))
                    
                break
        else:
            Xi = Xi_
    print(str(time.time()-t) + "sec")

