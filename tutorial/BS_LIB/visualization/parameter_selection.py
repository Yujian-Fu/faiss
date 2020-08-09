import numpy as np 
import matplotlib.pyplot as plt

dataset = "DEEP1M"
title = dataset + " / Num Centroids - Visited Vectors"

recording_proportion = [0.1, 0.5, 0.8, 1]
recall_performance = [1, 10, 100]

filepath = "/home/yujian/Desktop/Recording_Files/VQ/" + dataset + "/reasoning.txt"

# This is the figure for centroids - visited vectors (recall@K p%)
# This is the figure for centroids - training time
# This is the figure for centroids - avg_train distance

# This is the figure for 

for i in range(len(recall_performance)):
    title = title + " / Recall@ " + str(recall_performance[i])
    legends = ["1"] if recall_performance[i] == 1 else ["0.1", "0.5", "0.8", "1"] 
    position = 0
    plt.figure()

    n_centroids = []
    visited_vectors = []
    each_visited_vectors = []

    file = open(filepath, "r")
    f1 = file.readlines()

    recording_vectors = 0
    recording_position = []
    is_recording = False

    for x in f1:
        if "n_centroids:" in x:
            n_centroids.append(x.split(" ")[-3])


        if "R@" + str(recall_performance[i]) + " MC:" in x: 
            max_visited_centroids = x.split(" ")[-1] 
            position = 0 
            is_recording = True 
        else: 
            position += 1 
        
        if recall_performance[i] == 1 and is_recording: 
            if position == 2: 
                each_visited_vectors.append(x.split(" ")[-2])
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
                            break 
            if position == 2 and is_recording:
                is_recording = False
                for j in range(len(recording_proportion)):
                    each_visited_vectors.append(x.split(" ")[recording_position[j]])

        if "The average max centroids:" in x:
            if recall_performance[i] == 1:
                visited_vectors.append(np.mean(list(map(float, each_visited_vectors))))
            else:
                each_visited_length = int(len(each_visited_vectors) / len(recording_proportion))
                for j in range(len(recording_proportion)):
                    sub_visited_vectors = [each_visited_vectors[temp * len(recording_proportion) + j] for temp in range(each_visited_length)]
            
                    
                    visited_vectors.append(np.mean(list(map(float, sub_visited_vectors))))

            each_visited_vectors = []


    if (recall_performance[i] == 1):
        plt.plot(list(map(float, n_centroids)), visited_vectors)
        visited_vectors = []
    else:
        for j in range(len(recording_proportion)):
            sub_visited_vectors = [visited_vectors[temp * len(recording_proportion) + j] for temp in range(len(n_centroids))]
            plt.plot(list(map(float, n_centroids)), sub_visited_vectors)
        visited_vectors = []


    plt.xticks(np.linspace(0, float(n_centroids[-1]), 5))
    plt.legend(legends)
    plt.xlabel("num centroids")
    plt.ylabel("visited vectors")
    plt.title(title)
    plt.show()






            

                


            
            
                
            
            
            














