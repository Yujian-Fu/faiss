import numpy as np 
import networkx as nx
import faiss 
import os
import utils
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew

#Load the dataset
dataset_path = ""
property_file = ""
dist_path = ""
label_path = ""

k_result = 10
dataset = utils.fvecs_read(dataset_path)
dimension = dataset.shape[0]

#Compute the NN graph
if os.path.exists(dist_path):
    distance = np.load(dist_path)
    labels = np.load(label_path)

else:
    index = faiss.IndexFlatL2(dimension)
    index.add(dataset)
    distance, labels = index.search(dataset, k_result + 1)


'''
Draw the distance distribution of the NN graph
'''
x = []
for i in range(dataset.shape[0]):
    for j in range(k_result):
        x.append(distance[i][j + 1])
x = np.array(x)
dist_kurtosis = kurtosis(x)
dist_skew = skew(x)

sns.kdeplot(x, shade=True)
sns.rugplot(x)
plt.show()

'''
Compute the LID and entropy of the dataset
'''
LID = utils.compute_LID(dataset)
entropy = utils.compute_entropy(dataset)


#Analyze the property
DG = nx.DiGraph()
DG.add_nodes_from(range(dataset.shape[0]))

for i in range(dataset.shape[0]):
    for j in range(k_result):
        DG.add_weighted_edges_from([(i, labels[j + 1], distance[j + 1])])

G = nx.Graph()
DG.add_nodes_from(range(dataset.shape[0]))

for i in range(dataset.shape[0]):
    for j in range(k_result):
        G.add_edges_from([(i, labels[j + 1])])

'''
Properties from networkX
Re-check the requirement of those features 
'''

AC = nx.average_clustering(DG)

DSC = nx.degree_assortativity_coefficient(DG)

DC_List = []
DC = np.array(list(nx.degree_centrality(DG)))
DC_List.append(np.mean(DC))
DC_List.append(np.median(DC))
DC_List.append(np.std(DC))

#IDC = nx.in_degree_centrality(DG)

#ODC = nx.out_degree_centrality(DG)

TS = nx.transitivity(DG)

DT = nx.diameter(DG)

RD = nx.radius(DG)

#Not implemented for directed type
#GE = nx.global_efficiency(DG)

#Only support when all nodes are connected
ASPL = nx.average_shortest_path_length(DG)

#Not implemented for directed type
#SIGMA = nx.sigma(DG)

#Not implemented for directed type
#OMEGA = nx.omega(DG)















