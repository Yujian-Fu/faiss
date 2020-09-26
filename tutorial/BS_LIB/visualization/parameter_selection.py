import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

def inter(x):
    inter_x = []
    type_x = type(x[0])
    for i in range(len(x)):
        if i == len(x) -1:
            inter_x.append(x[i])
        else:
            inter_x.append(x[i])
            inter_x.append(type_x((int(x[i]) + int(x[i + 1])) / 2))
    return inter_x


def fit(x, y, size):
    #x = np.linspace(-10,10,100)  # 创建时间序列
    #p_value = [-2,5,10] # 原始数据的参数
    #noise = np.random.randn(len(x))  # 创建随机噪声
    #y = Fun(p_value,x)+noise*2 # 加上噪声的序列

    print(x, y)
    popt,pcov=curve_fit(lambda x, a1,a2,a3,a4: a1 * x**3 + a2*x**2+a3*x +a4,x,y)
    a1,a2,a3,a4 = popt
    y_fitted = a1 * x**3 + a2*x**2+a3*x +a4
    funtion_string = str(round(a1, 4))+ "x^3+" + str(round(a2, 3)) + "x^2+" + str(round(a3, 2)) + "x+" + str(round(a4, 2))

    plt.figure
    plt.plot(x,y,'r', label = 'Original curve')
    plt.plot(x,y_fitted,'-b', label ='Fitted curve')

    plt.text(x[0]/2, y[-1], funtion_string, ha='left', wrap=True)
    if (size == 10000):
        distance_ratio = 50
    elif (size == 100000):
        distance_ratio = 10
    elif (size == 1000000):
        distance_ratio = 5
    a3 = a3 + distance_ratio
    y_time = a1 * x**3 + a2*x**2+a3*x +a4
    plt.plot(x,y_time,'-b', label ='Actual Time')
    print(a1, a2, a3, a4)
    if (a1 > 0):
        a11 = 0
    else:
        a11 = a1
    turning_point_x = ((-2 * a2) + math.sqrt(4 * a2 * a2 - 12 * a11 * a3)) / (6 * a1)
    turning_point_y = a1 * turning_point_x**3 + a2*turning_point_x**2+a3*turning_point_x +a4
    print(turning_point_x, turning_point_y)
    plt.text(turning_point_x, turning_point_y,"(" + str(round(turning_point_x, 2)) + " , " + str(round(turning_point_y, 2)) + ")")
    plt.legend()
    #plt.show()


dataset = "Random_700"

recording_proportion = [1]
recall_performance = [1, 10, 100]

filepaths = ["/home/yujian/Desktop/Recording_Files/VQ/analysis/" + dataset + "_reasoning.txt"]

# This is the figure for centroids - visited vectors (recall@K p%)
# This is the figure for centroids - training time
# This is the figure for centroids - avg_train distance

# This is the figure for 


size_list = [1000000]

for i in range(len(recall_performance)):

    
    #legends = ["1"]
    position = 0
    plt.figure()

    
    visited_vectors = []
    each_visited_vectors = []
    #visited_centroids = []
    #each_visited_centroids = []

    for filepath in filepaths:
        file = open(filepath, "r")
        f1 = file.readlines()

        recording_vectors = 0
        recording_position = []
        is_recording = False
        dataset_recording = False

        for size in size_list:
            plt.figure()
            dataset_name = dataset + " " +str(size) + " "
            title = dataset_name + " / Num Centroids - Search Time"
            title1 = title + " / Recall@ " + str(recall_performance[i])
            n_centroids = []

            for x in f1:
                if dataset_name in x:
                    dataset_recording = True
                elif dataset in x:
                    dataset_recording = False

                if "n_centroids:" in x and dataset_recording:
                    n_centroids.append(x.split(" ")[-3])

                if "R@" + str(recall_performance[i]) + " MC:" in x and dataset_recording: 
                    #max_visited_centroids = x.split(" ")[-1] 
                    position = 0 
                    is_recording = True 
                else: 
                    position += 1 
                
                if recall_performance[i] == 1 and is_recording and dataset_recording: 
                    if position == 2: 
                        each_visited_vectors.append(x.split(" ")[-2])
                        #each_visited_centroids.append(len(x.split(" ")) - 1)
                        is_recording = False
                
                else: 
                    if position == 1 and is_recording: 
                        recording_position = [] 
                        visiting_vectors = x.split(" ") 
                        for j in range(len(recording_proportion)): 
                            recording_vectors = recall_performance[i] * recording_proportion[j] 
                            for k in range(len(visiting_vectors)): 
                                if (recording_vectors <= float(visiting_vectors[k])): 
                                    recording_position.append(k) 
                                    #each_visited_centroids.append(k+1)
                                    break 
                    if position == 2 and is_recording:
                        is_recording = False
                        for j in range(len(recording_proportion)):
                            each_visited_vectors.append(x.split(" ")[recording_position[j]])

                if "The average max centroids:" in x and dataset_recording:
                    if recall_performance[i] == 1:
                        visited_vectors.append(np.mean(list(map(float, each_visited_vectors))))
                        #visited_centroids.append(np.mean(list(map(float, each_visited_centroids))))
                    else:
                        each_visited_length = int(len(each_visited_vectors) / len(recording_proportion))
                        for j in range(len(recording_proportion)):
                            sub_visited_vectors = [each_visited_vectors[temp * len(recording_proportion) + j] for temp in range(each_visited_length)]                
                            visited_vectors.append(np.mean(list(map(float, sub_visited_vectors))))
                            
                            #sub_visited_centroids = [each_visited_centroids[temp * len(recording_proportion) + j] for temp in range(each_visited_length)] 
                            #visited_centroids.append(np.mean(list(map(float, sub_visited_centroids)))) 

                    each_visited_vectors = []
                    #each_visited_centroids = []


            if (recall_performance[i] == 1):
                inter_n_centroids = inter(n_centroids)
                inter_visited_vectors = inter(visited_vectors)
                print(inter_n_centroids, inter_visited_vectors)
                #plt.plot(list(map(float, inter_n_centroids)), inter_visited_vectors)

                fit(np.array(np.array(list(map(float, inter_n_centroids)))), np.array(inter_visited_vectors), size)
                #plt.plot(list(map(float, n_centroids)), list(np.array(visited_vectors) + distance_ratio * np.array(visited_centroids)))
                visited_vectors = []
            else:
                for j in range(len(recording_proportion)):
                    
                    sub_visited_vectors = [visited_vectors[temp * len(recording_proportion) + j] for temp in range(len(n_centroids))]
                    inter_n_centroids = inter(n_centroids)
                    inter_sub_visited_vectors = inter(sub_visited_vectors)
                    fit(np.array(np.array(list(map(float, inter_n_centroids)))), np.array(inter_sub_visited_vectors), size)
                    #sub_visited_centroids = [visited_centroids[temp * len(recording_proportion) + j] for temp in range(len(n_centroids))]
                    #plt.plot(list(map(float, n_centroids)), sub_visited_vectors)
                    #plt.plot(list(map(float, n_centroids)), list(np.array(sub_visited_vectors) + distance_ratio * np.array(sub_visited_centroids)))

                visited_vectors = []

            print(n_centroids)
            #plt.xticks(np.arange(0, int(n_centroids[-1]), int(n_centroids[-1]) / 10))
            #plt.legend(legends)
            plt.xlabel("num centroids")
            plt.ylabel("sum search time / T")
            plt.title(title1)
            plt.savefig("/home/yujian/Desktop/Recording_Files/VQ/analysis/" + dataset_name + "_" + str(recall_performance[i]))






            

                


            
            
                
            
            
            














