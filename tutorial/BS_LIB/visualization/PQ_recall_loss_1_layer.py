import numpy as np 
import matplotlib.pyplot as plt

dataset = "SIFT1M"


recall_metrics = [1, 5, 10, 20, 30, 40, 50, 80, 100]
recall_length = len(recall_metrics)

filenames = ["/reasoning_PQ_8.txt", "/reasoning_PQ_12.txt"]
title = dataset + " / Visited Vectors - Recall Proportion"
filepaths = ["/home/yujian/Desktop/Recording_Files/VQ/" + dataset + filenames[i] for i in range(len(filenames))]
legends = ["PQ 8", "PQ 12", "Visited GroundTruth"]


plt.figure()

position = 0
recording = False
record_position = [] 
gt_record = []
PQ_record_metrics = []
visited_vectors = []
avg_gt_record = []
avg_PQ_record_metrics = []
avg_visited_vectors = []

for filepath in filepaths:

    record_position = [] 
    gt_record = []
    PQ_record_metrics = []
    visited_vectors = []
    avg_gt_record = []
    avg_PQ_record_metrics = []
    avg_visited_vectors = []
    file = open(filepath, "r")
    f1 = file.readlines()


    for x in f1:
        position = position + 1

        if "MC: " in x:
            position = 0
            recording = True

        
        if recording:
            if position == 1:
                visited_gt = list(map(float, x.split(" ")[0: -1]))
                record_position  = [-1] * recall_length
                for i in range(len(visited_gt)):
                    for j in range(recall_length):
                        if record_position[j] == -1 and visited_gt[i] >= recall_metrics[j]:
                            record_position[j] = i
                            gt_record.append(visited_gt[i])

            if position == 2:
                PQ_metrics = list(map(float, x.split(" ")[0: -1]))
                
                for i in range(recall_length):
                    assert(record_position[i] >= 0)
                    PQ_record_metrics.append(PQ_metrics[record_position[i]])

            if position == 3:
                query_visited_vectors = list(map(float, x.split(" ")[0: -1]))
                for i in range(recall_length):
                    assert(record_position[i] >= 0)
                    visited_vectors.append(query_visited_vectors[record_position[i]])
                record_position = []
    
    sum_queries =int(len(visited_vectors) / recall_length)
    for i in range(recall_length):
        avg = np.mean([gt_record[temp * recall_length + i] for temp in range(sum_queries)])
        avg_gt_record.append(avg)
        avg = np.mean([PQ_record_metrics[temp * recall_length + i] for temp in range(sum_queries)])
        avg_PQ_record_metrics.append(avg)
        avg = np.mean([visited_vectors[temp * recall_length + i] for temp in range(sum_queries)])
        avg_visited_vectors.append(avg)
    
    plt.plot(avg_visited_vectors, avg_PQ_record_metrics)

plt.plot(avg_visited_vectors, avg_gt_record)
plt.legend(legends)
plt.xlabel("Visited Vectors")
plt.ylabel("Correct Result")
plt.show()





