import numpy as np 
import 

'''
Read the recording file
'''

record_path = ""
record_file = open(record_path, "r")
f1 = record_file.readlines()

for x in f1:

train_set = ""
