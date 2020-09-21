from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import shutil
from datetime import datetime
import time
import json
import pickle
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns



class Model():
    pass

def libby_signs():
    files = [('2','2'),('22','22'),('24','24'),('32','32'),('39','39'),('50','unknown'),('51','unknown'),('52','unknown'),('53','unknown')]


    model = load_model('models/4_traffic')


    f = open('app/static/final_report/final_eval.txt', 'w')
    for img_path, alt in files:
        image = Image.open(f'app/static/final_report/libby_signs/{img_path}.png')
        image = np.asarray(image.resize((30,30)))
        image = np.resize(image, [1,30,30,3])
        certainty_values = model.predict(image)[0]
        predicted_class = np.argmax(certainty_values, axis=-1)
        certainty =f'{round(certainty_values[predicted_class],4)*100} %'

        f.write(f'{img_path}.png,{predicted_class},{certainty},{alt}.png\n')

    f.close()

def ROC():
    with open('pickles/4_traffic.pk', 'rb') as fb:
        model = pickle.load(fb)
    preds = model.preds

    # Create ROC Curve plot
    fig, axes = plt.subplots()

    l = len(preds)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    best_acc = 0
    best_threshold = 0

    x = list()
    y = list()
    # Creating an ROC curve
    for threshold in range(100):

        # All predictions that are above the confidence threshold (predictions above threshold)
        classified = preds[preds.confidence > threshold/100]
        unclassified = preds[preds.confidence <= (threshold/100)]

        # Calculating TP, FP, TN, FN
        # TP are CORRECTLY classified signs with confidence ABOVE the threshold
        # TN are INCORRECTLY classified signs with confidence BELOW the threshold
        # FP are INCORRECTLY classified signs with confidence ABOVE the threshold
        # FN are CORRECTLY classified signs with confidence BELOW the threshold
        true_positives = len(classified[classified.predicted_class == classified.actual_class])  
        true_negatives = len(unclassified[unclassified.predicted_class != unclassified.actual_class])
        false_positives = len(classified[classified.predicted_class != classified.actual_class])
        false_negatives = len(unclassified[unclassified.predicted_class == unclassified.actual_class])

        acc = (true_positives + true_negatives) / l

        y.append(true_positives/(true_positives+false_negatives))
        x = false_positives/(false_positives+true_positives)

        if acc > best_acc:
            best_acc = acc
            best_thresh = threshold

    area_under = (sum(y)/x)*(x*100) + (1-x)*100


    # axes.axvline(best_conf_thresh,)
    # axes.text(best_conf_thresh, best_tp_rate, f'. {best_tp_rate} True Positive Rate')

    return area_under

def conf_curve():

    with open('pickles/4_traffic.pk', 'rb') as fb:
        model = pickle.load(fb)
    preds = model.preds

    # Create ROC Curve plot
    fig, axes = plt.subplots()

    # Elements to include in final evaluation
    best_acc = 0
    zero_acc = 0
    best_threshold = 0
    best_conf_matrix = []
    zero_conf_matrix = []

    l= len(preds)

    for threshold in range(100):

        # All predictions that are above the confidence threshold (predictions above threshold)
        classified = preds[preds.confidence > threshold/100]
        unclassified = preds[preds.confidence <= (threshold/100)]

        # Calculating TP, FP, TN, FN
        # TP are CORRECTLY classified signs with confidence ABOVE the threshold
        # TN are INCORRECTLY classified signs with confidence BELOW the threshold
        # FP are INCORRECTLY classified signs with confidence ABOVE the threshold
        # FN are CORRECTLY classified signs with confidence BELOW the threshold
        true_positives = len(classified[classified.predicted_class == classified.actual_class])  
        true_negatives = len(unclassified[unclassified.predicted_class != unclassified.actual_class])
        false_positives = len(classified[classified.predicted_class != classified.actual_class])
        false_negatives = len(unclassified[unclassified.predicted_class == unclassified.actual_class])
        acc = (true_positives + true_negatives) / l

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_conf_matrix = ([
                f'{round(true_positives/l*100,2)}%',
                f'{round(false_positives/l*100,2)}%',
                f'{round(true_negatives/l*100,2)}%',
                f'{round(false_negatives/l*100,2)}%'
                ])

        if threshold == 0:
            zero_acc = acc
            zero_conf_matrix = ([
                f'{round(true_positives/l*100,2)}%',
                f'{round(false_positives/l*100,2)}%',
                f'{round(true_negatives/l*100,2)}%',
                f'{round(false_negatives/l*100,2)}%'
                ])


    report = {
        'best_acc':round(best_acc*100,2),
        'zero_acc':round(zero_acc*100,2),
        'threshold':best_threshold,
        'best_matrix':best_conf_matrix,
        'zero_matrix':zero_conf_matrix}

    with open('app/static/final_report/final_report.txt', 'w') as f:
        print(report, file=f)


if __name__ == "__main__":
    conf_curve()