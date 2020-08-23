import numpy as np 
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

'''
Read the recording file
'''

record_path = "/home/yujian/Desktop/Recording_Files/VQ/SIFT1M/recording_reranking_paras_8_8_.txt"
record_file = open(record_path, "r")
f1 = record_file.readlines()

dimension = 128
query_vector = []

d_1st_1 = []
d_10th_1 = []
d_1st_d_10th_1 = []
update_search_1 = []

d_1st_2 = []
d_10th_2 = []
d_1st_d_10th_2 = []
update_search_2 = []

target_space = []

next_query = False
next_space = False
space_position = 0
for x in f1:
    if next_query:
        query_vector.append(list(map(float, x.split(" ")[0: -1])))
        next_query = False
    
    if "Query: " in x:
        next_query = True
    
    if "n / 2 d_1st: " in x:
        d_1st_1.append(float(x.split("n / 2 d_1st: ")[1].split("\n")[0]))
    
    if "n / 2 d_10th: " in x:
        d_10th_1.append(float(x.split("n / 2 d_10th: ")[1].split("\n")[0]))
    
    if "n / 2 d_1st / d_10th " in x:
        d_1st_d_10th_1.append(float(x.split("n / 2 d_1st / d_10th ")[1].split("\n")[0]))
    
    if "search times / n / 2 update times " in x:
        update_search_1.append(float(x.split("search times / n / 2 update times ")[1].split("\n")[0]))
    
    if "n d_1st: " in x:
        d_1st_2.append(float(x.split("n d_1st: ")[1].split("\n")[0]))
    
    if "n d_10th: " in x:
        d_10th_2.append(float(x.split("n d_10th: ")[1].split("\n")[0]))
    
    if "n d_1st / d_10th " in x:
        d_1st_d_10th_2.append(float(x.split("n d_1st / d_10th ")[1].split("\n")[0]))
    
    if "search times / n update times " in x:
        update_search_2.append(float(x.split("search times / n update times ")[1].split("\n")[0]))
    
    if next_space and space_position < 13:
        target_space.append(float(x.split(" ")[1]))
        space_position += 1
    
    if space_position == 13:
        next_space = False
    
    if "Visited vectors: " in x:
        next_space = True
        space_position = 0

total_feature_dimension = dimension + 8
total_target_dimesnion = 13
total_size = 1000

set_x = np.zeros((1000, total_feature_dimension))
set_y = np.zeros((1000, total_target_dimesnion))

for i in range(1000):
    for j in range(dimension):
        set_x[i][j] = query_vector[i][j]
    set_x[i][dimension] = d_1st_1[i]
    set_x[i][dimension + 1] = d_10th_1[i]
    set_x[i][dimension + 2] = d_1st_d_10th_1[i]
    set_x[i][dimension + 3] = update_search_1[i]
    set_x[i][dimension + 4] = d_1st_2[i]
    set_x[i][dimension + 5] = d_10th_2[i]
    set_x[i][dimension + 6] = d_1st_d_10th_2[i]
    set_x[i][dimension + 7] = update_search_2[i]
    
    set_y[i][:] = target_space[i * total_target_dimesnion : (i + 1) * total_target_dimesnion]

x_train = set_x[0: 900, :]
y_train = set_y[0: 900, 0]
x_test = set_x[900:, :]
y_test = set_y[900:, 0]


'''
Train GBDT model
'''

lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'}, # l1和l2代表两种误差计算
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
               lgb_train,
               num_boost_round=20,
               valid_sets=lgb_eval,
               early_stopping_rounds=15)

gbm.save_model('Regressionmodel_reranking.txt')
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
print(y_pred)
print(y_test)
print(np.sum(abs((y_pred - y_test) / y_test)))







