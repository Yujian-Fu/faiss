import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

#Time fit for LQ layer:
'''
Parameter: dimension is ignored
(upper_nc, upper_space, nc_per_group, nq) -> time
time = f(upper_sapce, nc_per_group) = a * upper_space * nc_per_group + b (a = alpha * dimension)
'''

def fitFunc(x, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a1 * x[1] * x[2] * x[3] +  a3 * x[0] * x[1] * x[2] * x[3] + a4 * x[3] + a5 * x[0] * x[3]  + a6 * x[1] * x[3] + a8 * x[0] * x[1] * x[3]  + a10


def fit(x, y):
    popt,pcov=curve_fit(fitFunc, x,y)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = popt
    y_fitted = fitFunc(x, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    print(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    print(np.average(abs(y_fitted - y)/y))

    save_file_path = "/home/yujian/Desktop/Recording_Files/VQ_LQ/analysis/LQ_time_para_error.txt"
    save_file = open(save_file_path, "w")

    error_visualization = []
    for upper_nc in range(100, 500, 100):
        for upper_space in range(10, 40, 10):
            for nc in range(20, 80, 20):
                sub_x = []
                sub_y = []
                sub_y_fitted = []

                for i in range(x.shape[1]):
                    if x[0][i] == upper_nc and x[1][i] == upper_space and x[2][i] == nc:
                        sub_x.append(x[3][i])
                        sub_y.append(y[i])
                        sub_y_fitted.append(y_fitted[i])

                if len(sub_x) > 0:
                    error_average = round(np.sum((np.array(sub_y) - np.array(sub_y_fitted))**2) / len(sub_x), 4)
                    save_file.write(str(upper_nc) + " " + str(upper_space) + " " + str(nc) + " " + str(error_average) + "\n")
                    plt.figure()
                    plt.title("Estimated LQ time curve " + str(upper_nc) +" "+ str(upper_space) + " " + str(nc))
                    plt.scatter(sub_x,sub_y, c = 'r', alpha=0.3)
                    plt.plot(sub_x,sub_y_fitted,'-b', label ='Fitted curve')
                    plt.xlabel("Number of query")
                    plt.ylabel("Time / ms")
                    plt.legend()
                    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_LQ/analysis/LQ_time "+ str(upper_nc) +" "+ str(upper_space) + " " + str(nc) +".png")
                    plt.close("all")
                    error_visualization.append([upper_nc, upper_space, nc, error_average])

    
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0], np.array(error_visualization)[:, -1])
    plt.xlabel("upper nc")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_LQ/analysis/LQ_time_origin_upper_nc_error.png")
    
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("upper space")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_LQ/analysis/LQ_time_upper_space_error.png")
    
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0] + np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("nc")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_LQ/analysis/LQ_time_nc_error.png")


    '''
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0] * np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("space multiplication")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_LQ/analysis/K_min_time_mul_space_error.png")
    '''

    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0] / np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("nc upper division")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_LQ/analysis/LQ_time_div_space_error.png")
    


filepath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/VQ_LQ.txt"

file = open(filepath, "r")

f1 = file.readlines()
nq = 0
upper_space = 0
nc1 = 0
nc2 = 0
train_x = []
train_y = []

for x in f1:
    if "LQ structure" in x:
        nc1 = float(x.split(" ")[-2])
        nc2 = float(x.split(" ")[-1])

    if "Time for" in x:
        nq = float(x.split(" ")[2])
        #train_x.append([1, nc1, nq])
        #time = float(x.split(" ")[10])
        #train_y.append(time * 1000)

        upper_space = float(x.split(" ")[8])
        train_x.append([nc1, upper_space, nc2, nq])
        time = float(x.split(" ")[12])
        train_y.append(time * 1000)

train_x = np.array(train_x)
train_y = np.array(train_y)

fit(train_x.T, train_y)

# fast: 0.0008424393938696317 0.01857056273987856 -3.0638269954115085  (error: 0.534)
# no fast: 0.0014367530256419458 -0.005218256399372123 1.8645497735450391 (erroe: 0.085)

