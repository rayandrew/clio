#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys
from subprocess import call
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
from timeit import default_timer
from PIL import Image

import statsmodels.api as sm
import tensorflow as tf
from tensorflow import keras
from keras import layers
from functools import partial
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, classification_report, average_precision_score, accuracy_score
import matplotlib.pyplot as plt

sys.path.append('../../../commonutils')
# import default_ip_finder

INPUT_FEATURES = 31
CUSTOM_LOSS = 5.0

# IPFinder Algorithm only uses Tangent
def tangent_based(arr_value: list[int]) -> list[int]:
    ip = [0, 0]
    lat_array = np.array(arr_value)

    # remove the inf
    lat_array = lat_array[~np.isinf(lat_array)]

    lat_97 = np.percentile(lat_array, 97)
    lat_array = lat_array[lat_array <= lat_97]
    max_lat = np.max(lat_array)
    lat_array = np.divide(lat_array, max_lat)

    ecdf = sm.distributions.ECDF(lat_array)
    x = np.linspace(0, 1, num=10000)
    y = ecdf(x)

    t = y - x
    ip_idx = np.argmax(t)
    ip[0] = x[ip_idx]
    ip[1] = y[ip_idx]
    ip[0] = int(ip[0] * max_lat)
    ip[1] = ip[1] * 0.97 * 100
    return ip


def create_output_dir(output_path):
    os.makedirs(output_path, exist_ok=True)
    return output_path

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    print("===== output file : " + filePath)

def print_confusion_matrix(figure_path, y_test, y_pred):
    y_test_class, y_pred_class = y_test, y_pred
    target_names = ["Fast", "Slow"]
    labels_names = [0,1]
    stats = []
    stats.append(classification_report(y_test_class, y_pred_class,labels=labels_names, target_names=target_names, zero_division=0))

    fig, ax = plt.subplots(figsize=(4, 3))

    cm = confusion_matrix(y_test_class, y_pred_class)

    # Calculate ROC-AUC and FPR/FNR
    cm_values = [0 for i in range(4)]
    i = 0
    for row in cm:
        for val in row:
            cm_values[i] = val
            i += 1
    TN, FP, FN, TP = cm_values[0], cm_values[1], cm_values[2], cm_values[3]
    FPR, FNR = round(FP/(FP+TN + 0.1),3), round(FN/(TP+FN  + 0.1),3)
    try:
        ROC_AUC = round(roc_auc_score(y_test, y_pred),3)
    except ValueError:
        ROC_AUC = 0 # if all value are classified into one class, which is BAD dataset
    try:
        PR_AUC = round(average_precision_score(y_test, y_pred),3)
    except ValueError:
        PR_AUC = 0
        
    stats.append("FPR = "+ str(FPR) + "  (" + str(round(FPR*100,1))+ "%)")
    stats.append("FNR = "+ str(FNR) + "  (" + str(round(FNR*100,1))+ "%)")
    stats.append("ROC-AUC = "+ str(ROC_AUC) + "  (" + str(round(ROC_AUC*100,1))+ "%)")
    stats.append("PR-AUC = "+ str(PR_AUC) + "  (" + str(round(PR_AUC*100,1))+ "%)")
    
    disp = ConfusionMatrixDisplay(np.reshape(cm_values, (-1, 2)), display_labels=target_names)
    disp = disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='g')
    ax.set_title("FPR = " + str(round(FPR*100,1))+ "%  and FNR = " + str(round(FNR*100,1))+ "%"); 

    # FN -> bottom left corner
    plt.savefig(figure_path, bbox_inches='tight')
    # print("===== output figure : " + figure_path)
    return stats, ROC_AUC, PR_AUC

def plot_latency_cdf(figure_path, complete_df, title):
    # the df is already readonly IOs
    y_pred = complete_df.loc[complete_df["y_pred"] == 0, "latency"].values
    # Draw CDF
    N=len(y_pred)
    data = y_pred
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)

    y_test = complete_df["latency"].values
    N=len(y_test)
    data = y_test
    # sort the data in ascending order
    x_2 = np.sort(data)
    # get the cdf values of y
    y_2 = np.arange(N) / float(N)
    percent_slow = int( (N-len(y_pred)) / N * 100)

    # plotting
    plt.figure(figsize=(6,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title(title + "; Slow = " + str(percent_slow)+ "%")
    p70_lat = np.percentile(x_2, 70)
    plt.xlim(0, max(p70_lat * 3, 1000)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_2, y_2, label = "Raw Latency", color="red")
    plt.plot(x_1, y_1, label = "FlashNet-powered", linestyle='dashdot', color="green")
    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches='tight')
    # print("===== output figure : " + figure_path)

    arr_accepted_io = map(str, y_pred)
    return arr_accepted_io

def plot_loss(figure_path, history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_path, bbox_inches='tight')

#-------------------------Custom Loss--------------------------
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = tf.keras.backend.zeros_like(y_pred[:, 0])
    y_pred_max = tf.keras.backend.max(y_pred, axis=1)
    y_pred_max = tf.keras.backend.reshape(y_pred_max, (tf.keras.backend.shape(y_pred)[0], 1))
    y_pred_max_mat = tf.keras.backend.cast(tf.keras.backend.equal(y_pred, y_pred_max), tf.keras.backend.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    cross_ent = tf.keras.losses.categorical_hinge(y_true, y_pred)
    return cross_ent * final_mask

#-------------------------Print FP TP FN TN--------------------------
def perf_measure(y_actual, y_pred):
    class_id = set(y_actual).union(set(y_pred))
    TP = []
    FP = []
    TN = []
    FN = []

    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                FP[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                FN[index] += 1
    total_data = len(y_actual)
    print ("total dataset " + str(total_data))
    print ( "  id  TP  FP  TN   FN")
    for x ,_id in enumerate(class_id):
        print ("  " + str(_id) + "\t" + str(TP[x]) + "\t" +  str(FP[x]) + "\t" +  str(TN[x]) + "\t" +  str(FN[x]))
    
    print ("\n_id    %FP        %FN")
    percentFP = []
    percentFN = []
    for x ,_id in enumerate(class_id):
        if ( FN[x]+ TP[x] > 0 and FP[x]+ TN[x] > 0):
            percentFP.append(FP[x]/( FP[x]+ TN[x])*100)
            percentFN.append(FN[x]/( FN[x]+ TP[x])*100)
            print ("  " + str(_id) + "   " + str(float("{:.2f}".format(percentFP[x]))) + " \t\t " + str(float("{:.2f}".format(percentFN[x]))))
  
    # print (  "\nmacro %FP and %FN = " + str(float("{:.2f}".format(np.sum(percentFP)/2))))
    print ("\n")


def get_model():
    w_array = np.ones((2,2))
    w_array[1, 0] = CUSTOM_LOSS   #Custom Loss Multiplier

    ncce = partial(w_categorical_crossentropy, weights=w_array)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=31, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss=ncce, metrics=['accuracy'])
    return model

def get_X_y(file_path):
    train_data = pd.read_csv(file_path, dtype='float32',sep=',', header=None)
    train_data = train_data.values
    train_input = train_data[:,:INPUT_FEATURES]
    train_output = train_data[:,INPUT_FEATURES]
    
    lat_threshold, _ = tangent_based(train_data[:,INPUT_FEATURES])
    train_y = []
    for num in train_output:
        labels = [0] * 2
        if num < lat_threshold:
            labels[0] = 1
        else:
            labels[1] = 1
        train_y.append(labels)
    
    return np.array(train_input), np.array(train_y,dtype=np.float32), lat_threshold

def train_model(model, train_input_path, train_eval_split):
    ratios = train_eval_split.split("_")
    percent_data_for_training = int(ratios[0])
    percent_data_for_eval = int(ratios[1])
    assert( percent_data_for_training + percent_data_for_eval == 100)
    
    input_feature = INPUT_FEATURES
    
    X, y, lat_threshold = get_X_y(train_input_path)
    
    num_train_entries = int(len(y) * (percent_data_for_training / 100))
    train_Xtrn = X[:num_train_entries,:]
    train_Xtst = X[num_train_entries:,:]
    train_ytrn = y[:num_train_entries]
    train_ytst = y[num_train_entries:]

    stop_train = 0
    for i in range(8):
        start_train = default_timer()
        model.fit(train_Xtrn, train_ytrn, epochs=1, batch_size=128, verbose=0)
        stop_train += default_timer() - start_train
        print('Iteration '+str(i)+'\n')
    
    y_pred = np.argmax(model.predict(train_Xtst), axis=1)
    y_test = np.argmax(train_ytst, axis=1)
    
    accuracy = np.mean(y_test == y_pred)
    print('Accuracy: ', accuracy)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", help="Path to the dataset dir", type=str)
    parser.add_argument("-train_eval_split", help="Ratio to split the dataset for training and evaluation",type=str)
    parser.add_argument("-output")
    parser.add_argument("-retrain")
    parser.add_argument("-store_w", help="Add the flag to store the weights and biases", action='store_true', default=False)
    args = parser.parse_args()
    if (not args.dataset_dir or not args.train_eval_split or not args.output):
        print("    ERROR: You must provide these arguments: -dataset <the labeled trace>  -train_eval_split <the split ratio> ")
        exit(-1)

    # print("Dataset " + args.dataset)
    import glob
    # glob dataset_dir for .csv.labeled files
    files = []
    for file in glob.glob(args.dataset_dir + "/*.csv.labeled", recursive=True):
        files.append(file)
        
    if len(files) == 0:
        print("No .csv.labeled files found in the dataset directory")
        exit(-1)
    
    # natsorted file
    from natsort import natsorted
    files = natsorted(files)
    
    model = get_model()
    model = train_model(model, files[0], args.train_eval_split)
    
    train_prop = args.train_eval_split.split("_")[0]
    eval_prop = args.train_eval_split.split("_")[1]
    results_df = pd.DataFrame(columns=['no', 'type', 'accuracy', 'auc', 'lat_threshold', 'path'])

    total = len(files)
    for idx, file in enumerate(files):
        print("Now at idx/total: " + str(idx) + "/" + str(total))
        X, y, lat_threshold = get_X_y(file)
        y_pred = np.argmax(model.predict(X), axis=1)
        y_test = np.argmax(y, axis=1)
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        if args.retrain:
            model = train_model(model, file, args.train_eval_split)
            results_df = results_df._append({'no': idx, 'type': "retrain", 'accuracy': acc, 'auc': auc, 'lat_threshold': lat_threshold, 'path': file}, ignore_index=True)
        else:
            results_df = results_df._append({'no': idx, 'type': "no_retrain", 'accuracy': acc, 'auc': auc, 'lat_threshold': lat_threshold, 'path': file}, ignore_index=True)
        
    # mkdir
    output_path = create_output_dir(args.output)
    results_df.to_csv(args.output + "/results.csv", index=False)
        
        
        
        

        
    
    

# Example how to run:
# python linnOS_model_B.py -dataset /mnt/extra/flashnet/model_collection/1_per_io_admission/dataset/nvme1n1/alibaba.cut.per_10k.least_size.686/profile_feat_linnOS.ionet -train_eval_split 50_50 -store_w
# python model.py -dataset_dir /home/cc/clio/runs/exp/tencent/1063/1m/iops/linnos/feat/sudden/6800_6840 -train_eval_split 70_30 
# python model.py -dataset_dir /home/cc/clio/runs/exp/tencent/1063/1m/iops/linnos/feat/sudden/2000_2050 -train_eval_split 70_30  -output ./linnos/