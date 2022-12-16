from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn import tree, datasets
from scipy.io import arff
from os.path import join
from timeit import default_timer as timer
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import keras.datasets.cifar10 as cifar10
import tensorflow as tf
import numba
import torch
import wandb
import datetime
import time
import os
import math
import json
import inspect
import concurrent.futures as cf  # doesn't work with sklearn
import pandas as pd
import numpy as np
import copy as copy
import statistics as stt
import seaborn as sns
import pickle
import sys
sns.set_theme()
tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(log_device_placement=True))


class CNN():
    def __init__(self, X_train, y_train, X_test, y_test):
        # stats variables
        self.stats_dict = {
            'model_name': None,
            'model_file_path': None,
            'epochs': None,
            'batch_size': None,
            'building_time': None,
            'training_time': None,
            'testing_time': None,
            'loss': None,
            'accuracy': None,
        }
        current_time = str(time.strftime(
            "%Y/%m/%d %H:%M:%S", time.localtime()))
        self.stats_dict['time'] = current_time
        current_time = current_time.replace(
            '/', '').replace(' ', '_').replace(':', '')
        self.current_time = current_time

    def build_model(self, model_num=1):
        start = timer()
        model = Sequential()
        if model_num == 1:
            channel1 = 100
            channel2 = 200
            channel3 = 400
            model.add(Conv2D(channel1, (9, 9),
                             input_shape=X_train.shape[1:], activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(channel2, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(channel3, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(channel3, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(160, activation='relu'))
            model.add(Dense(40, activation='relu'))
            model.add(Dense(10, activation='softmax'))
        elif model_num == 2:
            model.add(Conv2D(32, (3, 3),
                             input_shape=X_train.shape[1:], activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Conv2D(256, (3, 3), activation='relu'))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(160, activation='relu'))
            model.add(Dense(40, activation='relu'))
            model.add(Dense(10, activation='softmax'))
        elif model_num == 3:
            model.add(Conv2D(32, (3, 3),
                             input_shape=X_train.shape[1:], activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(10, activation='softmax'))
        elif model_num == 4:
            model.add(Conv2D(32, (3, 3),
                             input_shape=X_train.shape[1:], activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(10, activation='softmax'))
        elif model_num == 5:
            model.add(Conv2D(32, (5, 5),
                             input_shape=X_train.shape[1:], activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(512, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(160, activation='relu'))
            model.add(Dense(40, activation='relu'))
            model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        self.model = model
        self.stats_dict['building_time'] = timer() - start

    def train_model(self, epochs=5, batch_size=500, verbose=1):
        self.stats_dict['epochs'] = epochs
        self.stats_dict['batch_size'] = batch_size
        start = timer()
        self.history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.stats_dict['training_time'] = timer() - start
        # self.model.summary()

    def store_model(self, model_name, path=''):
        path = os.path.join(path, f'{model_name}_{self.current_time}.h5')
        print(path)
        self.model.save(path)
        self.stats_dict['model_name'] = model_name
        self.stats_dict['model_file_path'] = path

    def store_meta_data(self, model_name, path=''):
        # check empty values
        values = list(self.stats_dict.values())
        if None in values:
            arr = []
            for element in self.stats_dict:
                if self.stats_dict[element] == None:
                    arr.append(element)
            ans = input(
                f'Not all stats are recorded...\nThese stats are not recorded: {arr}\nDo you still want to export? (y/[n]): ')
            if ans != 'y':
                return

        # read meta data, append, and write
        path = os.path.join(path, 'model_meta_data.csv')
        if os.path.exists(path):
            meta_data = pd.read_csv(path, encoding='utf-8', index_col=0)
            meta_data = pd.concat(
                [meta_data, pd.DataFrame(self.stats_dict, index=[0])])
            meta_data.reset_index(drop=True, inplace=True)
        else:
            meta_data = pd.DataFrame(self.stats_dict, index=[0])
        meta_data.to_csv(path, encoding='utf-8')
        print('Meta data exported to', path)

    def load_model(self, path):
        self.model = load_model(path)

    def evaluate(self):
        # record loss and accuracy
        start = timer()
        result = self.model.evaluate(X_test, y_test)
        self.stats_dict['loss'] = result[0]
        self.stats_dict['accuracy'] = result[1]
        self.stats_dict['testing_time'] = timer() - start

    def evaluate1(self):
        # top 1 accuracy
        pass

    def evaluate2(self):
        # top 3 accuracy
        pass

    def evaluate3(self):
        # top 5 accuracy
        pass

    def clear_mem(self):
        # https://www.kaggle.com/getting-started/140636
        torch.cuda.empty_cache()
        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)
        print('Cleared GPU memory')

    def plot_history(self, model_name, path):
        history = self.history
        plt.figure(figsize=(20, 20), dpi=300)
        # plot loss
        plt.subplot(211)
        plt.title(f'{model_name}_{self.current_time} - Train Loss')
        plt.plot(history.history['loss'], label='train_loss')
        plt.xlabel('Epoch')
        # plt.legend(loc='upper right')
        # plot accuracy
        plt.subplot(212)
        plt.title(f'{model_name}_{self.current_time} - Train Accuracy')
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.xlabel('Epoch')
        # plt.legend(loc='lower right')
        # save
        path = os.path.join(path, f'{model_name}_{self.current_time}.png')
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    os.system('cls')
    ##############################################################################
    # 1. Load the CIFAR-10 dataset & Preprocess the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    ##############################################################################
    # 2. Set user parameters
    model_root_path = os.path.join(
        os.getcwd().rstrip('src'), 'data', 'ml_w14_hw')
    epochs = 150
    batch_size = 1024
    loop_times = 30

    ##############################################################################
    # 3. Loop testing models
    wandb.init()
    for i in range(loop_times):
        for j in range(1, 5+1):
            # gpu_usage()
            model_name = f'model{j}'
            cnn = CNN(X_train=X_train, y_train=y_train,
                      X_test=X_test, y_test=y_test)
            cnn.build_model(model_num=j)
            cnn.train_model(epochs=epochs, batch_size=batch_size, verbose=1)
            cnn.plot_history(model_name=model_name, path=model_root_path)
            cnn.store_model(model_name=model_name, path=model_root_path)
            cnn.evaluate()
            cnn.store_meta_data(model_name=model_name, path=model_root_path)
