import csv

import pandas as pd
import numpy as np
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import json
import multiprocessing
import os
import gc
gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

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
print(positives)
# --------------------------
# Negatives

samples_list = list(idendities.values())

negatives = pd.DataFrame()

print(samples_list)
negatives = []

# 13673
print(samples_list[0])
print(len(idendities))
print(len(samples_list))

# wait a moment
# wait a moment
# wait me a moment OK "In this cross_product loop, if you run has been comptelely hanged after 1 minute"
samples_list = list(idendities.values())

negatives = []

for i in range(0, len(idendities) - 1):
    for j in range(i + 1, len(idendities)):
        cross_product = itertools.product(samples_list[i], samples_list[j])
        cross_product = list(cross_product)
        for cross_sample in cross_product:
            negative = [cross_sample[0], cross_sample[1]]
            negatives.append(negative)

negatives = pd.DataFrame(negatives, columns=["file_x", "file_y"])
negatives["decision"] = "No"

negatives = negatives.sample(positives.shape[0])
# --------------------------
# Merge positive and negative ones

df = pd.concat([positives, negatives]).reset_index(drop=True)

print(df.decision.value_counts())
df.file_x = "deepface/tests/dataset/" + df.file_x
df.file_y = "deepface/tests/dataset/" + df.file_y
# --------------------------
# DeepFace

from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID

pretrained_models = {}

pretrained_models["VGG-Face"] = VGGFace.loadModel()
print("VGG-Face loaded")
pretrained_models["Facenet"] = Facenet.loadModel()
print("Facenet loaded")
pretrained_models["OpenFace"] = OpenFace.loadModel()
print("OpenFace loaded")
pretrained_models["DeepFace"] = FbDeepFace.loadModel()
print("FbDeepFace loaded")
pretrained_models["DeepID"] = DeepID.loadModel()
print("DeepID loaded")

instances = df[["file_x", "file_y"]].values.tolist()

models = ['VGG-Face']
metrics = ["cosine"]

if True:
    for model in models:
        for metric in metrics:

            resp_obj = DeepFace.verify(instances
                                       , model_name=model
                                       , model=pretrained_models[model]
                                       , distance_metric=metric)

            distances = []

            for i in range(0, len(instances)):
                distance = round(resp_obj["pair_%s" % (i + 1)]["distance"], 4)
                distances.append(distance)

            df['%s_%s' % (model, metric)] = distances

    df.to_csv("face-recognition-pivot.csv", index=False)
else:
    df = pd.read_csv("face-recognition-pivot.csv")

df_raw = df.copy()

# --------------------------
# Distribution

# fig = plt.figure(figsize=(15, 15))

# figure_idx = 1
# for model in models:
#    for metric in metrics:
#        feature = '%s_%s' % (model, metric)

#        ax1 = fig.add_subplot(4, 2, figure_idx)

#        df[df.decision == "Yes"][feature].plot(kind='kde', title=feature, label='Yes', legend=True)
#        df[df.decision == "No"][feature].plot(kind='kde', title=feature, label='No', legend=True)

#        figure_idx = figure_idx + 1

# plt.show()
# --------------------------
# Pre-processing for modelling

columns = []
for model in models:
    for metric in metrics:
        feature = '%s_%s' % (model, metric)
        columns.append(feature)

columns.append("decision")

df = df[columns]

df.loc[df[df.decision == 'Yes'].index, 'decision'] = 1
df.loc[df[df.decision == 'No'].index, 'decision'] = 0

print(df.head())
# --------------------------
# Train test split

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.30, random_state=17)

target_name = "decision"

y_train = df_train[target_name].values
x_train = df_train.drop(columns=[target_name]).values

y_test = df_test[target_name].values
x_test = df_test.drop(columns=[target_name]).values

# --------------------------
# LightGBM

import lightgbm as lgb

features = df.drop(columns=[target_name]).columns.tolist()
lgb_train = lgb.Dataset(x_train, y_train, feature_name=features)
lgb_test = lgb.Dataset(x_test, y_test, feature_name=features)

params = {
    'task': 'train'
    , 'boosting_type': 'gbdt'
    , 'objective': 'multiclass'
    , 'num_class': 2
    , 'metric': 'multi_logloss'
}

gbm = lgb.train(params, lgb_train, num_boost_round=1000, early_stopping_rounds=15, valid_sets=lgb_test)

gbm.save_model("face-recognition-ensemble-model.txt")

# --------------------------
# Evaluation

predictions = gbm.predict(x_test)

predictions_classes = []
for i in predictions:
    prediction_class = np.argmax(i)
    predictions_classes.append(prediction_class)

cm = confusion_matrix(list(y_test), predictions_classes)

tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn)
precision = tp / (tp + fp)
accuracy = (tp + tn) / (tn + fp + fn + tp)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision: ", 100 * precision, "%")
print("Recall: ", 100 * recall, "%")
print("F1 score ", 100 * f1, "%")
print("Accuracy: ", 100 * accuracy, "%")

# --------------------------
# Interpretability

# ax = lgb.plot_importance(gbm, max_num_features=20)
# plt.show()

# import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# plt.rcParams["figure.figsize"] = [20, 20]

# for i in range(0, gbm.num_trees()):
#    ax = lgb.plot_tree(gbm, tree_index=i)
#    # plt.show()
#
#    if i == 2:
#        break
# --------------------------
# ROC Curve

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

y_pred_proba = predictions[::, 1]
y_test = y_test.astype(int)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('DeepID_Euclidean_l2')
plt.plot(fpr, tpr, label="AUC=" + str(auc))
# plt.figure(figsize=(6, 6))
plt.legend(loc=4)
plt.savefig('DeepID_Euclidean_l2.png')
plt.show()

# --------------------------
