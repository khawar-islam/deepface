import pandas as pd
import numpy as np
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import json
import multiprocessing
import os
from itertools import combinations
from numba import jit, cuda
import numba

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def myproduct(si, sj):
    #this is equivalent to malloc in c, thath is, prerightallocate the memory
    l = len(si)*len(sj)*[None]# I do not use append because ok
    zi = len(si)
    for i, di in enumerate(si):
        for j, dj in enumerate(sj):
            l[i*zi + j] = (di, dj)
    return l
'''
class DynamicNegative:
    def __init__(self, samples_list, idendities):
        self.samples_list = samples_list
        self.idendities = idendities

    def getSize(self): 
        cnt = 0
        for i in range(0, len(idendities) - 1):
             for j in range(i + 1, len(idendities)):
                 zi = len(samples_list[i])
                 zj = len(samples_list[j])
                 cnt = cnt + zi*zj
 
        return cnt
   

    def generate()

'''

def product(samples_list, idendities):

  cnt = 0
  for i in range(0, len(idendities) - 1):
      for j in range(i + 1, len(idendities)):
          zi = len(samples_list[i])
          zj = len(samples_list[j])
          
 
  return cnt
  
  negatives = []
  for i in range(0, len(idendities) - 1):
    for j in range(i + 1, len(idendities)):

        si = [i for i, v in enumerate(samples_list[i])]
        sj = [i for i, v in enumerate(samples_list[j])]

        #cross_product = itertools.product(samples_list[i], samples_list[j])
        cross_product = itertools.product(si, sj)
        cross_product = list(cross_product)
         
        print("________________")
        print(len(samples_list[i]))
        print(len(samples_list[j]))
        print(len(cross_product))
       
        #continue

        for cross_sample in cross_product:
            # print(cross_sample[0], " vs ", cross_sample[1])
            negative = []
            negative.append(cross_sample[0])
            negative.append(cross_sample[1])
            negatives.append(negative)
            #print(len(negatives))




with open('/home/khawar/deepface/tests/morph.json') as f:
    data = json.load(f)

idendities = data

# Positives

positives = []

for key, values in idendities.items():
    print(key)
    for i in range(0, len(values) - 1):
        for j in range(i + 1, len(values)):
            print(values[i], " and ", values[j])
            positive = [values[i], values[j]]
            positives.append(positive)

positives = pd.DataFrame(positives, columns=["file_x", "file_y"])
positives["decision"] = "Yes"
print(positives.shape)
# --------------------------
# Negatives

samples_list = list(idendities.values())
print(samples_list)
negatives = []

#13673
print(samples_list[0])
print(len(idendities))
print(len(samples_list))
