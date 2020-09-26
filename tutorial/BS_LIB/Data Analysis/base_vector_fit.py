import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

#Time fit for LQ layer:
'''
Parameter: dimension is ignored
(M, nbits, nprobe, instances) -> time
'''

def fitFunc(x, a1, a2, a3):
    return a1 * x[1] * x[0] + a2 * x[1] + a3
def fit(x, y):
    print(x[0])
    plt.figure()
    plt.scatter(x[1], y)
    plt.show()

    popt,pcov=curve_fit(fitFunc,x,y)
    a1, a2, a3 = popt
    print(a1, a2, a3)
    y_fitted = a1 * x[1] * x[0] + a2 * x[1] + a3
    print(np.average(abs(y_fitted - y)/y))

    for M in range(4, 20, 4):
        if M == 12:
            continue
        #for nbits in range(4, 10, 2):
        sub_x = []
        sub_y = []
        sub_y_fitted = []

            
        for i in range(x.shape[1]):
            if x[0][i] == M :
                sub_x.append(x[1][i])
                sub_y.append(y[i])
                sub_y_fitted.append(y_fitted[i])


        plt.figure()
        plt.title("Estimated base time curve " + str(M))
        plt.scatter(sub_x,sub_y, c = 'r', alpha=0.3)
        plt.plot(sub_x,sub_y_fitted,'-b', label ='Fitted curve')
        plt.xlabel("Number of query")
        plt.ylabel("Time / ms")
        plt.legend()
        plt.savefig("/home/yujian/Desktop/Recording_Files/VQ/analysis/base_time "+ str(M) +".png")



filepath = "/home/yujian/Downloads/similarity_search_datasets/data/sift1M/recording/base_vector_time.txt"

file = open(filepath, "r")

f1 = file.readlines()
nprobe = 0
M = 0
nbits = 0
train_x = []
train_y = []

for x in f1:
    if "PQ structure" in x:
        M = float(x.split(" ")[-3])
        nbits = float(x.split(" ")[-1])

    if "The nprobe for search is: " in x:
        nprobe = int(x.split(" ")[-1])


    if "Search base instances: " in x:
        instances = int(x.split(" ")[3])
        train_x.append([M, instances])
        time = float(x.split(" ")[-1])
        train_y.append(time * 1000)

train_x = np.array(train_x)
train_y = np.array(train_y)

print(train_x, train_y)
fit(train_x.T, train_y)

# fast: 0.0008424393938696317 0.01857056273987856 -3.0638269954115085  (error: 0.534)
# no fast: 0.0014367530256419458 -0.005218256399372123 1.8645497735450391 (erroe: 0.085)


