from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
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
        # data variables
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
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
            'evaluating_proba_time': None,
        }
        # labels
        self.label = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
                      4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
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
        elif model_num == 6:
            # testing deep conv layer
            img_width = X_train.shape[1:][0]
            filter_size = 3
            layer_depth = img_width / (filter_size - 1)
            model = Sequential()
            model.add(Conv2D(32, (3, 3),
                             input_shape=X_train.shape[1:], activation='relu'))
            for i in range(int(layer_depth)-2):
                model.add(Conv2D(32, (3, 3), activation='relu'))
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
            self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
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

    def evaluate(self):
        # record loss and accuracy
        start = timer()
        result = self.model.evaluate(X_test, y_test)
        self.stats_dict['loss'] = result[0]
        self.stats_dict['accuracy'] = result[1]
        self.stats_dict['testing_time'] = timer() - start

    def load_model(self, path):
        self.model = load_model(path)
        self.model_name = os.path.basename(path).split('.')[0]

    def evaluate_proba(self, img_row_num=2, img_col_num=5, display_bar_num=3, random=False, start_img_num=0, save=False, path='', show=True):
        # top 1 accuracy
        '''
        display images and their predicted probabilities
        -----------------------------------------------------------------------
        start_img_num: start image number (necessary if random=False)
        save_img_path: path to save images
        img_row_num: number of rows of images
        img_col_num: number of columns of images
        display_bar_num: number of bars to display (probability of each class)
        random: if True, select images randomly
        save: if True, save images
        show: if True, show images
        -----------------------------------------------------------------------
        src: https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot
        '''
        start = timer()
        model = self.model
        X_test = self.X_test
        y_test = self.y_test
        label = self.label
        model_name = self.model_name
        # load test data (images)
        image_count = img_row_num * img_col_num
        test_image_count = X_test.shape[0]
        if random:
            selected_image_indexes = np.random.choice(
                a=test_image_count, size=image_count, replace=False)
            selected_image = X_test[selected_image_indexes]
        else:
            selected_image_indexes = np.arange(
                start=start_img_num, stop=start_img_num+image_count)
            selected_image = X_test[selected_image_indexes]
        # predict
        probabilities = model.predict(selected_image)
        # calculate prediction accuracy
        correct_count = 0
        for i in range(image_count):
            prediction = np.argmax(probabilities[i])
            answer = np.argmax(y_test[selected_image_indexes[i]])
            if prediction == answer:
                correct_count += 1
        accuracy = correct_count / image_count * 100
        # plot
        fig = plt.figure(figsize=(img_col_num*2, img_row_num*2), dpi=300)
        plt.title(
            f'Top {display_bar_num} Accuracy of {image_count} Images ({img_col_num} x {img_row_num})\nAccuracy of this batch: {accuracy:.2f}%', pad=30)
        plt.axis('off')
        outer = gridspec.GridSpec(
            img_row_num, img_col_num, wspace=0.1, hspace=0.1)
        for i in range(image_count):
            prediction = label[np.argmax(probabilities[i])]
            answer = label[np.argmax(y_test[selected_image_indexes[i]])]
            x = i//img_col_num
            y = i % img_col_num
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            # plot image
            ax = plt.Subplot(fig, inner[0])
            ax.set_title(f'{prediction} ({answer})', fontsize=8)
            ax.imshow(selected_image[i])
            ax.axis('off')
            fig.add_subplot(ax)
            # get top x probabilities
            temp_probabilities = (copy.copy(probabilities[i])).tolist()
            temp_probabilities.sort(reverse=True)
            temp_probabilities = temp_probabilities[:display_bar_num]
            temp_label = []
            for j in temp_probabilities:
                prob_list = probabilities[i].tolist()
                index = prob_list.index(j)
                temp_label.append(label[index])
            # plot bar
            ax = plt.Subplot(fig, inner[1])
            ax.barh(np.arange(display_bar_num),
                    temp_probabilities, align='center', alpha=0.5)
            for element in temp_label:
                text = f'{element} ({temp_probabilities[temp_label.index(element)]*100:.2f}%)'
                ax.text(0.01, temp_label.index(element),
                        text, fontsize=6, va='center')
            ax.invert_yaxis()
            ax.axis('off')
            fig.add_subplot(ax)
        if save:
            path = os.path.join(
                path, '{}_eval_top{}_{}x{}_{}-{}.png'.format(model_name, display_bar_num, img_col_num, img_row_num, start_img_num, start_img_num+image_count))
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()
        print('Image exported to', path)
        self.stats_dict['evaluating_proba_time'] = timer() - start

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
        path = os.path.join(
            path, f'{model_name}_{self.current_time}_train_hist.png')
        plt.savefig(path)
        plt.close()


class LOOP_EXECUTOR():
    def loop_build_model(model_st_num, model_end_num, loop_times, X_train, y_train, X_test, y_test, epochs, batch_size, model_root_path, eval_root_path):
        '''
        Create and store all models multiple times
        '''
        for i in range(loop_times):
            for j in range(model_st_num, model_end_num+1):
                model_name = f'model{j}'
                cnn = CNN(X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test)
                cnn.build_model(model_num=j)
                cnn.train_model(
                    epochs=epochs, batch_size=batch_size, verbose=1)
                cnn.plot_history(model_name=model_name, path=eval_root_path)
                cnn.store_model(model_name=model_name, path=model_root_path)
                cnn.evaluate()
                cnn.store_meta_data(model_name=model_name,
                                    path=model_root_path)
            print('progress: {}/{}'.format(i+1, loop_times))
        print('Done building all models')

    def loop_eval_model(X_train, y_train, X_test, y_test, model_root_path, eval_root_path):
        '''
        Loop through all models in model_root_path and evaluate them
        '''
        files = []
        for r, d, f in os.walk(model_root_path):
            for file in f:
                if file.endswith(".h5"):
                    files.append(os.path.join(r, file))
        for i in range(len(files)):
            cnn = CNN(X_train=X_train, y_train=y_train,
                      X_test=X_test, y_test=y_test)
            cnn.load_model(path=files[i])
            cnn.evaluate_proba(img_row_num=4, img_col_num=6,
                               display_bar_num=3, random=False, start_img_num=10,
                               save=True, path=eval_root_path, show=False)
            print('progress: {}/{}'.format(i+1, len(files)))
        print('Done evaluating all models')


if __name__ == '__main__':
    os.system('cls')
    wandb.init()
    ##############################################################################
    # 1. Load the CIFAR-10 dataset & Preprocess the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    ##############################################################################
    # 2. Set user parameters
    model_root_path = os.path.join(
        os.getcwd().rstrip('src'), 'data', 'ml_w14_hw')
    eval_root_path = os.path.join(
        os.getcwd().rstrip('src'), 'pic', 'ML_w14_hw')
    model_root_path = os.path.join(model_root_path, 'finished_models')
    epochs = 150
    batch_size = 1024
    loop_times = 30
    model_st_num = 6
    model_end_num = 6
    # parameter for testing loop_build_model
    epochs = 10
    batch_size = 10
    loop_times = 1

    ##############################################################################
    # 3. Loop testing models
    LOOP_EXECUTOR.loop_build_model(model_st_num=model_st_num, model_end_num=model_end_num, loop_times=loop_times, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   epochs=epochs, batch_size=batch_size, model_root_path=model_root_path, eval_root_path=eval_root_path)
    # LOOP_EXECUTOR.loop_eval_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    #                               model_root_path=model_root_path, eval_root_path=eval_root_path)
