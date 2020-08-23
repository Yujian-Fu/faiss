import numpy as np 
import 

'''
Read the recording file
'''

record_path = ""
record_file = open(record_path, "r")
f1 = record_file.readlines()

dimension = 128
query vector = []

d_1st_1 = []
d_10th_1 = []
d_1st_d_10th_1 = []
update_search_1 = []

d_1st_2 = []
d_10th_2 = []
d_1st_d_10th_2 = []
update_search_2 = []

target_space = []

for x in f1:

train_set = ""
