
def func(x):
    return (a*x**4) + (b*x**3) + (c*x**2) + (d*x) + e

a = float(5)
b = -22.4
c = 15.85272
d = 24.161472
e = -23.4824832

print("----------Bisection----------\n\n")
p1 = -7
q1 = 0
p2 = 0
q2 = 7

mid1 = p1 + abs(q1-p1)/2
mid2 = p2 + abs(q2-p2)/2

#case1 : -7 ~ 0
while(1):
    if func(mid1)*func(p1) < 0:
        e1 = abs((mid1 - q1)/mid1) * 100
        q1 = mid1
    else:
        e1 = abs((mid1 - p1)/mid1) * 100
        p1 = mid1
    if e1 < .0001:
        print(mid1)
        break
    else:
        mid1 = p1 + (q1-p1)/2
        
#case2 : 0 ~ 7
while(1):
    if func(mid2)*func(p2) < 0:
        e2 = abs((mid2 - q2)/mid2) * 100
        q2 = mid2
    else:
        e2 = abs((mid2 - p2)/mid2) * 100
        p2 = mid2
    if e2 < .0001:
        print(mid2)
        break
    else:
        mid2 = p2 + (q2-p2)/2

print("\n\n----------Newton-Raphson----------\n\n")

def slope(x):
    return (4*a*x**3) + (3*b*x**2) + (2*c*x) + d

Xi = -7
Xj = 7
Xk = 0

#case1 : init = -7
while(1):
    Xi_ = Xi - func(Xi)/slope(Xi)
    if abs((Xi_ - Xi)/Xi_) * 100 < .0001:
        print(Xi_)
        break
    else:
        Xi = Xi_

#case2 : init = 7
while(1):
    Xj_ = Xj - func(Xj)/slope(Xj)
    if abs((Xj_ - Xj)/Xj_) * 100 < .0001:
        print(Xj_)
        break
    else:
        Xj = Xj_

#case3 : init = 0
while(1):
    Xk_ = Xk - func(Xk)/slope(Xk)
    if abs((Xk_ - Xk)/Xk_) * 100 < .0001:
        print(Xk_)
        break
    else:
        Xk = Xk_










