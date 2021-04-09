import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from collections import Counter

#Generate 5 classes(clusters) data using random number generation
def class_maker():
    np.random.seed(seed=100)

    class1_x = np.random.normal(0, 2, size = 300)
    class1_y = np.random.normal(0, 2, size = 300)
    class1_z = np.random.normal(0, 1, size = 300)
    for i in range(300):
        class1[i][0] = class1_x[i]
        class1[i][1] = class1_y[i]
        class1[i][2] = class1_z[i]

    class2_x = np.random.normal(3, 1, size = 300)
    class2_y = np.random.normal(3, 2, size = 300)
    class2_z = np.random.normal(3, 1, size = 300)
    for i in range(300):
        class2[i][0] = class2_x[i]
        class2[i][1] = class2_y[i]
        class2[i][2] = class2_z[i]

    class3_x = np.random.normal(-5, 3, size = 300)
    class3_y = np.random.normal(-10, 2, size = 300)
    class3_z = np.random.normal(-3, 1, size = 300)
    for i in range(300):
        class3[i][0] = class3_x[i]
        class3[i][1] = class3_y[i]
        class3[i][2] = class3_z[i]

    class4_x = np.random.normal(-8, 2, size = 300)
    class4_y = np.random.normal(-4, 1, size = 300)
    class4_z = np.random.normal(2, 3, size = 300)
    for i in range(300):
        class4[i][0] = class4_x[i]
        class4[i][1] = class4_y[i]
        class4[i][2] = class4_z[i]

    class5_x = np.random.normal(3, 3, size = 300)
    class5_y = np.random.normal(5, 4, size = 300)
    class5_z = np.random.normal(-6, 2, size = 300)
    for i in range(300):
        class5[i][0] = class5_x[i]
        class5[i][1] = class5_y[i]
        class5[i][2] = class5_z[i]

#Testing with new vectors
#3D vectors generated with the same distributions of 5 clusters
def test_maker():
    np.random.seed(seed=150)

    tclass1_x = np.random.normal(0, 2, size = 100)
    tclass1_y = np.random.normal(0, 2, size = 100)
    tclass1_z = np.random.normal(0, 1, size = 100)
    for i in range(100):
        tclass1[i][0] = tclass1_x[i]
        tclass1[i][1] = tclass1_y[i]
        tclass1[i][2] = tclass1_z[i]

    tclass2_x = np.random.normal(3, 1, size = 100)
    tclass2_y = np.random.normal(3, 2, size = 100)
    tclass2_z = np.random.normal(3, 1, size = 100)
    for i in range(100):
        tclass2[i][0] = tclass2_x[i]
        tclass2[i][1] = tclass2_y[i]
        tclass2[i][2] = tclass2_z[i]

    tclass3_x = np.random.normal(-5, 3, size = 100)
    tclass3_y = np.random.normal(-10, 2, size = 100)
    tclass3_z = np.random.normal(-3, 1, size = 100)
    for i in range(100):
        tclass3[i][0] = tclass3_x[i]
        tclass3[i][1] = tclass3_y[i]
        tclass3[i][2] = tclass3_z[i]

    tclass4_x = np.random.normal(-8, 2, size = 100)
    tclass4_y = np.random.normal(-4, 1, size = 100)
    tclass4_z = np.random.normal(2, 3, size = 100)
    for i in range(100):
        tclass4[i][0] = tclass4_x[i]
        tclass4[i][1] = tclass4_y[i]
        tclass4[i][2] = tclass4_z[i]

    tclass5_x = np.random.normal(3, 3, size = 100)
    tclass5_y = np.random.normal(5, 4, size = 100)
    tclass5_z = np.random.normal(-6, 2, size = 100)
    for i in range(100):
        tclass5[i][0] = tclass5_x[i]
        tclass5[i][1] = tclass5_y[i]
        tclass5[i][2] = tclass5_z[i]

    #100 samples with different distributions
    tclass6_x = np.random.normal(7, 1, size = 100)
    tclass6_y = np.random.normal(7, 1, size = 100)
    tclass6_z = np.random.normal(7, 1, size = 100)
    for i in range(100):
        tclass6[i][0] = tclass6_x[i]
        tclass6[i][1] = tclass6_y[i]
        tclass6[i][2] = tclass6_z[i]

#Show individual class graph
def individual_graph(classN, color):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(classN[:,0],classN[:,1],classN[:,2],c=color,marker='o', s=5)
    plt.grid(True)
    plt.show()

#Show total class graph
def total_graph(total_class):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(total_class[0:300,0],total_class[0:300,1],total_class[0:300,2],c='red',marker='o', s=5)
    ax.scatter(total_class[300:600,0],total_class[300:600,1],total_class[300:600,2],c='green',marker='o', s=5)
    ax.scatter(total_class[600:900,0],total_class[600:900,1],total_class[600:900,2],c='blue',marker='o', s=5)
    ax.scatter(total_class[900:1200,0],total_class[900:1200,1],total_class[900:1200,2],c='orange',marker='o', s=5)
    ax.scatter(total_class[1200:1500,0],total_class[1200:1500,1],total_class[1200:1500,2],c='purple',marker='o', s=5)
    plt.grid(True)
    plt.show()

#Show total class graph with centers
def total_center_graph(c1,c2,c3,c4,c5,out,centers):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(c1[0:c1.shape[0],0],c1[0:c1.shape[0],1],c1[0:c1.shape[0],2],c='red',marker='o', s=5)
    ax.scatter(c2[0:c2.shape[0],0],c2[0:c2.shape[0],1],c2[0:c2.shape[0],2],c='green',marker='o', s=5)
    ax.scatter(c3[0:c3.shape[0],0],c3[0:c3.shape[0],1],c3[0:c3.shape[0],2],c='blue',marker='o', s=5)
    ax.scatter(c4[0:c4.shape[0],0],c4[0:c4.shape[0],1],c4[0:c4.shape[0],2],c='orange',marker='o', s=5)
    ax.scatter(c5[0:c5.shape[0],0],c5[0:c5.shape[0],1],c5[0:c5.shape[0],2],c='purple',marker='o', s=5)
    #Out data get black marker
    ax.scatter(out[0:out.shape[0],0],out[0:out.shape[0],1],out[0:out.shape[0],2],c='black',marker='o', s=5)
    ax.scatter(centers[:,0],centers[:,1],centers[:,2],c='yellow',marker='*', s=200)
    plt.grid(True)
    plt.show()

#Calculate distance
def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

#Recognition
def recog(test_data, centers, mode):
    tmp = []
    rec = []
    
    for i in range(test_data.shape[0]):
        lst = []
        #Nearest neighbor with 5 mean vectors of clusters
        for j in range(5):
            lst.append(distance(test_data[i], centers[j]))    
        for k in range(5):
            #Modeling the maximum distance for the clusters
            if min(lst) < max_distance:
                #Labeling
                if min(lst) == lst[k]:
                    tmp.append(k)
            else:
                #Out data
                tmp.append(5)
                break
    rec.append(tmp)
    rec = np.array(rec)
    
    c1 = test_data[rec.ravel() == 0]
    c2 = test_data[rec.ravel() == 1]
    c3 = test_data[rec.ravel() == 2]
    c4 = test_data[rec.ravel() == 3]
    c5 = test_data[rec.ravel() == 4]
    out = test_data[rec.ravel() == 5]

    total_center_graph(c1,c2,c3,c4,c5,out,centers)

    #Percent check
    #다르게 라벨링 된 것들 count
    #Recongition accuracy for each cluster
    
    #Case 1 : 미리 어떤식으로 라벨링 되는지 확인 후 recogntion
    #random하게 initial 값을 설정할 경우 변경될 가능성이 존재
    if mode == 1:
        #print(rec) # 3 4 1 2 0 5
        print("--------------Mode 1--------------")
        cnt1 = 0
        for i in range(0,100):
            if tmp[i] == 3:
                cnt1 += 1
        print(int(cnt1), end='')
        print('%')
        cnt2 = 0
        for i in range(100,200):
            if tmp[i] == 4:
                cnt2 += 1
        print(int(cnt2), end='')
        print('%')
        cnt3 = 0
        for i in range(200,300):
            if tmp[i] == 1:
                cnt3 += 1
        print(int(cnt3), end='')
        print('%')
        cnt4 = 0
        for i in range(300,400):
            if tmp[i] == 2:
                cnt4 += 1
        print(int(cnt4), end='')
        print('%')
        cnt5 = 0
        for i in range(400,500):
            if tmp[i] == 0:
                cnt5 += 1
        print(int(cnt5), end='')
        print('%')
        cnt6 = 0
        for i in range(500,600):
            if tmp[i] == 5:
                cnt6 += 1 
        print(int(cnt6), end='')
        print('%')

        print("Total : ", end='')
        print(int((cnt1+cnt2+cnt3+cnt4+cnt5+cnt6)/6), end='')
        print('%')
    
    #Case 2 : 100 단위로 가장 많은 라벨링 값으로 recogntion
    elif mode == 2:
        print("--------------Mode 2--------------")
        #100단위로 가장 많은 라벨링 값을 list에 넣어줌
        #정말 인식률이 낮은 경우 효과 적을 가능성 존재
        label_num = []
        label_num.append(Counter(tmp[0:100]).most_common(1)[0][0])
        label_num.append(Counter(tmp[100:200]).most_common(1)[0][0])
        label_num.append(Counter(tmp[200:300]).most_common(1)[0][0])
        label_num.append(Counter(tmp[300:400]).most_common(1)[0][0])
        label_num.append(Counter(tmp[400:500]).most_common(1)[0][0])
        label_num.append(Counter(tmp[500:600]).most_common(1)[0][0])

        #print(label_num)
        
        cnt1 = 0
        for i in range(0,100):
            if tmp[i] == label_num[0]:
                cnt1 += 1
        print(int(cnt1), end='')
        print('%')
        cnt2 = 0
        for i in range(100,200):
            if tmp[i] == label_num[1]:
                cnt2 += 1
        print(int(cnt2), end='')
        print('%')
        cnt3 = 0
        for i in range(200,300):
            if tmp[i] == label_num[2]:
                cnt3 += 1
        print(int(cnt3), end='')
        print('%')
        cnt4 = 0
        for i in range(300,400):
            if tmp[i] == label_num[3]:
                cnt4 += 1
        print(int(cnt4), end='')
        print('%')
        cnt5 = 0
        for i in range(400,500):
            if tmp[i] == label_num[4]:
                cnt5 += 1
        print(int(cnt5), end='')
        print('%')
        cnt6 = 0
        for i in range(500,600):
            if tmp[i] == label_num[5]:
                cnt6 += 1 
        print(int(cnt6), end='')
        print('%')

        print("Total : ", end='')
        print(int((cnt1+cnt2+cnt3+cnt4+cnt5+cnt6)/6), end='')
        print('%')

def Kmeans(data, clusters):
    pixel_vals = data
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1500, 0.85) 
    k = clusters
    #Find the 5 mean vectors of clusters
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS) 
    #print(centers)
    #print(labels)
    for i in range(pixel_vals.shape[0]):
        #Modeling the maximum distance for the clusters
        if(distance(pixel_vals[i], centers[labels[i][0]]) > max_distance):
            #Out data
            labels[i][0] = 5

    
    c1 = pixel_vals[labels.ravel() == 0]
    c2 = pixel_vals[labels.ravel() == 1]
    c3 = pixel_vals[labels.ravel() == 2]
    c4 = pixel_vals[labels.ravel() == 3]
    c5 = pixel_vals[labels.ravel() == 4]
    out = pixel_vals[labels.ravel() == 5]

    total_center_graph(c1,c2,c3,c4,c5,out,centers)

    return centers



class1 = np.array([[0, 0, 0]for i in range(300)], dtype='float32')
class2 = np.array([[0, 0, 0]for i in range(300)], dtype='float32')
class3 = np.array([[0, 0, 0]for i in range(300)], dtype='float32')
class4 = np.array([[0, 0, 0]for i in range(300)], dtype='float32')
class5 = np.array([[0, 0, 0]for i in range(300)], dtype='float32')

tclass1 = np.array([[0, 0, 0]for i in range(100)], dtype='float32')
tclass2 = np.array([[0, 0, 0]for i in range(100)], dtype='float32')
tclass3 = np.array([[0, 0, 0]for i in range(100)], dtype='float32')
tclass4 = np.array([[0, 0, 0]for i in range(100)], dtype='float32')
tclass5 = np.array([[0, 0, 0]for i in range(100)], dtype='float32')
tclass6 = np.array([[0, 0, 0]for i in range(100)], dtype='float32')

#For recognition mode
recog_mode = 1
max_distance = 10
class_maker()
total_class = np.vstack((class1,class2,class3,class4,class5))
#print(total_class)
total_graph(total_class)

centers = Kmeans(total_class, 5)

test_maker()
total_test = np.vstack((tclass1,tclass2,tclass3,tclass4,tclass5, tclass6))

recog(total_test, centers, recog_mode)

#recog(total_test, centers, 1)
#recog(total_test, centers, 2)






