import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

#Time fit for LQ layer:
'''
Parameter: dimension is ignored
(M, nbits, keep_space, nq) time

a1 + keep_space * nq + a2 * nq + a3 * M * nbits * keep_space * nq + a4
'''

def fitFunc(x, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a1 * x[0] * (2 ** x[1]) * x[3] + a2 * x[2] * x[3]+ a3 * x[0] * (2 ** x[1]) * x[2] *x[3] + a4 * x[0] * (2 ** x[1]) +  a6 * x[0] * x[2] * x[3] + a7 * (2 ** x[1]) * x[2] * x[3] +  a9 * (2 ** x[1]) * x[3] + a10


def fit(x, y):
    popt,pcov=curve_fit(fitFunc, x,y)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = popt
    y_fitted = fitFunc(x, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    print(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)

    print((np.average(abs(y_fitted - y)/y)))

    save_file_path = "/home/yujian/Desktop/Recording_Files/PQ/analysis/PQ_time_para_error.txt"
    save_file = open(save_file_path, "w")

    error_visualization = []
    for M in range(2, 6, 2):
        for nbits in range(4, 10, 2):
            for keep_space in range(10, 110, 10):
                sub_x = []
                sub_y = []
                sub_y_fitted = []

                
                for i in range(x.shape[1]):
                    if x[0][i] == M and x[1][i] == nbits and x[2][i] == keep_space:
                        sub_x.append(x[3][i])
                        sub_y.append(y[i])
                        sub_y_fitted.append(y_fitted[i])

                if (len(sub_x) > 0):
                    error_average = round(np.sum((np.array(sub_y) - np.array(sub_y_fitted))**2) / np.sum(np.array(sub_y) ** 2) / len(sub_x), 4)
                    save_file.write(str(M) + " " + str(nbits) + " " + str(keep_space) + " " + str(error_average) + "\n")
                    plt.figure()
                    plt.title("Estimated PQ time curve " + str(M) +" "+ str(nbits) + " " + str(keep_space))
                    plt.scatter(sub_x,sub_y, c = 'r', alpha=0.3)
                    plt.plot(sub_x,sub_y_fitted,'-b', label ='Fitted curve')
                    plt.xlabel("Number of query")
                    plt.ylabel("Time / ms")
                    plt.legend()
                    plt.savefig("/home/yujian/Desktop/Recording_Files/PQ/analysis/PQ_time_"+ str(M) +" "+ str(nbits) + " " + str(keep_space) +".png")
                    plt.close("all")
                    error_visualization.append([M, nbits, keep_space, error_average])


    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0], np.array(error_visualization)[:, -1])
    plt.xlabel("M")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/PQ/analysis/PQ_time_M_error.png")
    
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("nbits")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/PQ/analysis/PQ_time_nbits_error.png")
    
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 2], np.array(error_visualization)[:, -1])
    plt.xlabel("keep space")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/PQ/analysis/PQ_time_keep_space_error.png")


    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0] * np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("M * nbits")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/PQ/analysis/PQ_time_mul_space_error.png")


filepath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/VQ_PQ.txt"

file = open(filepath, "r")

f1 = file.readlines()
nq = 0
keep_space = 0
nbits = 0
M = 0
train_x = []
train_y = []

for x in f1:
    if "PQ structure" in x:
        nbits = float(x.split(" ")[-1])
        M = float(x.split(" ")[-3])

    if "Time for" in x:
        nq = float(x.split(" ")[2])
        #train_x.append([1, nc1, nq])
        #time = float(x.split(" ")[10])
        #train_y.append(time * 1000)

        keep_space = float(x.split(" ")[8])
        train_x.append([M, nbits, keep_space, nq])
        time = float(x.split(" ")[-1])
        train_y.append(time * 1000)

train_x = np.array(train_x)
train_y = np.array(train_y)

fit(train_x.T, train_y)




