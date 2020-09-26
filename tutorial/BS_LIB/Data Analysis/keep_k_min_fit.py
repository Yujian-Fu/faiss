import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

# Axes3D 包含的是实现3d绘图的各种方法
from mpl_toolkits.mplot3d import Axes3D


#Time fit for VQ layer:
'''
Parameter: dimension is ignored
origin_value, target_value, nq -> time
'''

def fitFunc(x, a1, a2, a3, a4, a5, a6, a7, a8):
    return  a1 * x[0] * x[2] + a2 * x[1] * x[2] + a3 * x[0] * x[1] * x[2] + a4 * x[2] + a5 * x[0] / x[1] * x[2]  + a6

def fit(x, y):
    print((x[1]))
    print(y)
    popt,pcov=curve_fit(fitFunc, x,y)
    a1, a2, a3, a4, a5, a6, a7, a8 = popt
    y_fitted = fitFunc(x, a1, a2, a3, a4, a5, a6, a7, a8)
    print(a1, a2, a3, a4, a5, a6, a7, a8)
    print(np.average(abs(y_fitted - y)/y))
    print(np.average(((y_fitted - y) / y) ** 2))
    
    #plt.figure()
    #plt.scatter(x[0], x[1])
    #plt.show()

    save_file_path = "/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/K_min_time_para_error.txt"
    save_file = open(save_file_path, "w")

    error_visualization = []
    for origin_sapce in [800, 100, 200, 1800, 300, 400, 1200, 600]:
        for second_space in [160, 40, 10, 80, 240, 20, 180, 120, 60, 30]:
            sub_x = []
            sub_y = []
            sub_y_fitted = []

            for i in range(x.shape[1]):
                if x[0][i] == origin_sapce:
                    if x[1][i] == second_space:
                        sub_x.append(x[2][i])
                        sub_y.append(y[i])
                        sub_y_fitted.append(y_fitted[i])

            if len(sub_x) > 0:
                error_average = round(np.sum((np.array(sub_y) - np.array(sub_y_fitted))**2) / len(sub_x), 4)
                save_file.write(str(origin_sapce) + " " + str(second_space) + " " + str(error_average) + "\n")
                plt.figure()
                plt.title("Estimated LQ time curve " + str(origin_sapce) + " " + str(second_space) + " (" + str(error_average) + ")")
                plt.scatter(sub_x,sub_y, c = 'r', alpha=0.3)
                plt.plot(sub_x,sub_y_fitted,'-b', label ='Fitted curve')
                plt.xlabel("Number of query")
                plt.ylabel("Time / ms")
                plt.legend()
                plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/K_min_time_"+ str(origin_sapce) +"_"+ str(second_space) + ".png")
                error_visualization.append([origin_sapce, second_space, error_average])
                plt.close('all') 

    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0], np.array(error_visualization)[:, -1])
    plt.xlabel("origin space")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/K_min_time_origin_space_error.png")
    
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("second_space")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/K_min_time_second_space_error.png")
    
    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0] + np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("space sum")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/K_min_time_sum_space_error.png")

    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0] * np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("space multiplication")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/K_min_time_mul_space_error.png")

    plt.figure()
    plt.scatter(np.array(error_visualization)[:, 0] / np.array(error_visualization)[:, 1], np.array(error_visualization)[:, -1])
    plt.xlabel("space division")
    plt.ylabel("error")
    plt.grid()
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/K_min_time_div_space_error.png")


    '''
    #print(np.average(abs(y_fitted - y)/y))

    # 新建一个画布
    figure = plt.figure()
    # 新建一个3d绘图对象
    ax = Axes3D(figure)
    X, Y = np.meshgrid(x[0][:], x[1][:])
    y_fitted = fitFunc([X, Y], a1, a2, a3, a4, a5, a6, a7, a8)
    #print(x[0][:], X)
    # 定义x,y 轴名称
    plt.xlabel("x")
    plt.ylabel("y")
    # 设置间隔和颜色
    ax.plot_surface(X, Y, y_fitted, rstride=1, cstride=1)
    ax.scatter(x[0][:], x[1][:], y)

    plt.show()
    '''




filepath = "/home/yujian/Downloads/similarity_search_datasets/data/sift1M/recording/VQ_VQ.txt"

file = open(filepath, "r")

f1 = file.readlines()
nq = 0
upper_space = 0
nc1 = 0
nc2 = 0
train_x = []
train_y = []

for x in f1:
    if "VQ structure" in x:
        nc1 = int(x.split(" ")[-2])
        nc2 = int(x.split(" ")[-1])

    if "Time for" in x:
        nq = int(x.split(" ")[2])

        upper_space = int(x.split(" ")[8])
        train_x.append([nc1, upper_space, nq])
        time = float(x.split(" ")[11])
        train_y.append(time * 1000)

        second_origin = upper_space * nc2;
        second_space = upper_space * int(x.split(" ")[9])
        train_x.append([second_origin, second_space, nq])
        time = float(x.split(" ")[13])
        train_y.append(time * 1000)

train_x = np.array(train_x)
train_y = np.array(train_y)
fit(train_x.T, train_y)


