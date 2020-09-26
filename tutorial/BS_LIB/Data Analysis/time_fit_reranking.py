import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

#Time fit for VQ layer:
'''
Parameter: dimension is ignored
(visited vectors, reranking space) -> time
'''

def fitFunc(x, a1, a2, a3, a4, a5, a6, a7, a8):
    return a1 * x[0] * x[1] + a2 * x[0] + a3 * x[1] + a4


def fit(x, y):
    popt,pcov=curve_fit(lambda x, a1, a2, a3, a4: a1 * x[1] * x[0] + a2 * x[0] + a3 * x[1] + a4,x,y)
    a1, a2, a3, a4 = popt
    y_fitted =a1 * x[1] * x[0] + a2 * x[0] + a3 * x[1] + a4
    print(a1, a2, a3, a4)
    print(np.average(abs(y_fitted - y)/y))

    print(list(x[1])[:10])
    print()

    plt.figure()
    plt.plot(list(x[1])[:10] , y_fitted[:10], '-b', label ='Fitted curve')
    plt.scatter(list(x[1])[:200], list(y)[:200], c = 'r', alpha=0.3)
    plt.xlabel("reranking space")
    plt.ylabel("Time / ms")
    plt.legend()
    plt.title("Reranking time - reranking space curve")
    plt.savefig("/home/yujian/Desktop/Recording_Files/VQ/analysis/reranking_time.png")
    

    '''
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


                plt.figure()
                plt.title("Estimated LQ time curve " + str(upper_nc) +" "+ str(upper_space) + " " + str(nc))
                plt.scatter(sub_x,sub_y, c = 'r', alpha=0.3)
                plt.plot(sub_x,sub_y_fitted,'-b', label ='Fitted curve')
                plt.xlabel("Number of query")
                plt.ylabel("Time / ms")
                plt.legend()
                plt.savefig("/home/yujian/Desktop/Recording_Files/VQ_VQ/analysis/VQ_time_"+ str(upper_nc) +" "+ str(upper_space) + " " + str(nc) +".png")
        '''



filepath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/reranking_time.txt"

file = open(filepath, "r")

f1 = file.readlines()

train_x = []
train_y = []

for x in f1:
    if "Visited vectors" in x:
        vv = int(x.split(" ")[2])
        reranking_space = int(x.split(" ")[5])

        train_x.append([vv, reranking_space])
        time = float(x.split(" ")[-1])
        train_y.append(time * 1000)


train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x, train_y)
fit(train_x.T, train_y)


