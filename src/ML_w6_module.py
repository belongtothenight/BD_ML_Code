import math
import pandas as pd
import numpy as np
import copy as copy
import statistics as stt
import seaborn as sns
from os import system, getcwd, startfile
from os.path import join
from time import time
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

'''
Link to images: https://github.com/belongtothenight/BD_ML_Code/tree/main/pic/ML_w6_module
'''


class DataSetError(Exception):
    """Base class for other exceptions"""
    pass


class FileTypeError(Exception):
    """Base class for other exceptions"""
    pass


class ML():
    # =========================================================================
    # core
    def __init__(self, data_path, plt_export_path, print_result=False):
        '''
        1. File validation check.
        2. Dataset validation check.
        3. Import datasets.
        '''
        self.label_ratio = 0.5
        self.data_path = data_path
        self.plt_export_path = plt_export_path
        if data_path.endswith('.data'):
            if self.data_path.endswith('wdbc.data'):
                self.dataset = 1.1
            else:
                raise DataSetError('Invalid Dataset')
            self.data = pd.read_csv(data_path, header=None)
        elif data_path.endswith('.arff'):
            if self.data_path.endswith('Pumpkin_Seeds_Dataset.arff'):
                self.dataset = 2.1
            else:
                raise DataSetError('Invalid Dataset')
            self.data = pd.DataFrame(arff.loadarff(data_path)[0])
        else:
            raise FileTypeError('Invalid data file')
        print(self.data) if print_result else None

    def preprocess_data_1(self, print_result=False):
        '''
        1. Process column names.
        2. Process missing values.(if any)
        3. Process data values(to number).(if any)
        4. Split data into X and y(result).
        '''
        if self.dataset == 1.1:
            print('Imported Dataset 1') if print_result else None
            self.wdbc_column_names = ['id', 'malignant',
                                      'nucleus_mean', 'nucleus_se', 'nucleus_worst',
                                      'texture_mean', 'texture_se', 'texture_worst',
                                      'perimeter_mean', 'perimeter_se', 'perimeter_worst',
                                      'area_mean', 'area_se', 'area_worst',
                                      'smoothness_mean', 'smoothness_se', 'smoothness_worst',
                                      'compactness_mean', 'compactness_se', 'compactness_worst',
                                      'concavity_mean', 'concavity_se', 'concavity_worst',
                                      'concave_pts_mean', 'concave_pts_se', 'concave_pts_worst',
                                      'symmetry_mean', 'symmetry_se', 'symmetry_worst',
                                      'fractal_dim_mean', 'fractal_dim_se', 'fractal_dim_worst']
            self.data.columns = self.wdbc_column_names
            self.data['malignant'] = self.data['malignant'].map(
                lambda x: 0 if x == "B" else 1)
            self.X = self.data.drop(columns=['id', 'malignant']).values
            s = StandardScaler()
            self.X = s.fit_transform(self.X)
            self.y = self.data['malignant'].values
        elif self.dataset == 2.1:
            print('Imported Dataset 2') if print_result else None
            self.data['Class'] = self.data['Class'].str.decode("utf-8")
            self.data['Class'] = self.data['Class'].map(
                lambda x: 0 if x == "CERCEVELIK" else 1)
            s = StandardScaler()
            self.X = s.fit_transform(self.data)
            self.y = self.data['Class'].values
        print(self.data) if print_result else None
        print(self.X) if print_result else None
        print(self.y) if print_result else None

    def preprocess_data_2(self, print_result=False):
        '''
        1. Umbalance data.
        2. Split data into train and test.
        '''
        if self.dataset == 1.1:
            X_0 = self.data[self.data.malignant == 0]
            X_1 = self.data[self.data.malignant == 1]
        elif self.dataset == 2.1:
            X_0 = self.data[self.data.Class == 0]
            X_1 = self.data[self.data.Class == 1]
        self.label_ratio = round(self.label_ratio, 3)
        Test_Size = 0.25
        max_train0_size = min(X_0.shape[0], X_1.shape[0])
        # the line below can sometimes cause error
        TestSizeFrom0 = 1 - min(max_train0_size,
                                X_0.shape[0] * (1 - Test_Size)) / X_0.shape[0]
        # print('Label Ratio: ', self.label_ratio)
        # print('Test Size: ', Test_Size)
        # print('Max Train Size: ', max_train0_size)
        # print('X_0 Size: ', X_0.shape[0])
        # print('X_1 Size: ', X_1.shape[0])
        # print('Test Size From 0: ', TestSizeFrom0)
        X_0_train, X_0_test = train_test_split(
            X_0, test_size=TestSizeFrom0, random_state=2018)
        X_1_train, X_1_test = train_test_split(
            X_1, test_size=(1 - self.label_ratio), random_state=2018)
        self.X_train = pd.concat([X_0_train, X_1_train])
        self.X_test = pd.concat([X_0_test, X_1_test])
        if self.dataset == 1.1:
            self.y_train = self.X_train.malignant.values
            self.X_train = self.X_train.drop(columns=['malignant'])
            self.y_test = self.X_test.malignant.values
            self.X_test = self.X_test.drop(columns=['malignant'])
        elif self.dataset == 2.1:
            self.y_train = self.X_train.Class.values
            self.X_train = self.X_train.drop(columns=['Class'])
            self.y_test = self.X_test.Class.values
            self.X_test = self.X_test.drop(columns=['Class'])
        print(self.X_train.columns) if print_result else None
        print(self.X_test.columns) if print_result else None

    def preprocess_data_3(self, print_result=False):
        '''
        1. Drop columns to change ratio of features. (malignant)
        See function "split_data_into_train_test" in "ML_w4_hw_q2.jpynb".
        '''
        pass

    def result_evaluation(self, y_test, y_pred, print_result=False):
        '''
        1. Accuracy
        2. Precision
        3. Recall
        4. F1 score
        5. Confusion matrix
        '''
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # confusion_matrix = pd.crosstab(
        #     y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        print('Accuracy: ', accuracy) if print_result else None
        print('Precision: ', precision) if print_result else None
        print('Recall: ', recall) if print_result else None
        print('F1 score: ', f1) if print_result else None
        # print(confusion_matrix) if print_result else None
        # , confusion_matrix
        return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)

    def deploy_model(self, default=False, print_result=False):
        '''
        1. Deploy Logistic Regression model.
        2. Deploy Decision Tree model.
        3. Deploy Random Forest model.
        4. Deploy SVM model.
        5. Deploy KNN model.
        ---------------------------------
        Result = [model, parameter_state, label_ratio,
            accuracy, precision, recall, f1]
        '''
        result = []

        # Logistic Regression
        if default:
            model = LogisticRegression()
        else:
            model = LogisticRegression(class_weight='balanced')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        result.append(['LR', default, round(self.label_ratio, 2)] +
                      list(self.result_evaluation(self.y_test, y_pred)))

        # Decision Tree
        if default:
            model = DecisionTreeClassifier()
        else:
            model = DecisionTreeClassifier(class_weight='balanced')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        result.append(['DT', default, round(self.label_ratio, 2)] +
                      list(self.result_evaluation(self.y_test, y_pred)))

        # Random Forest
        if default:
            model = RandomForestClassifier()
        else:
            model = RandomForestClassifier(class_weight='balanced')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        result.append(['RF', default, round(self.label_ratio, 2)] +
                      list(self.result_evaluation(self.y_test, y_pred)))

        # SVM
        if default:
            model = SVC()
        else:
            model = SVC(class_weight='balanced')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        result.append(['SVM', default, round(self.label_ratio, 2)] +
                      list(self.result_evaluation(self.y_test, y_pred)))

        # KNN
        if default:
            model = KNeighborsClassifier()
        else:
            model = KNeighborsClassifier(weights='distance')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        result.append(['KNN', default, round(self.label_ratio, 2)] +
                      list(self.result_evaluation(self.y_test, y_pred)))

        print(result) if print_result else None
        return result

    # =========================================================================
    # core routine

    def single_run(self, ratio=0.5, print_result=False):
        self.label_ratio = ratio
        self.preprocess_data_1()
        self.preprocess_data_2()
        r_default = self.deploy_model(default=True)
        r_balanced = self.deploy_model()
        print(r_default) if print_result else None
        print(r_balanced) if print_result else None
        return r_default, r_balanced

    def multi_run(self, min=10, max=100, inc=1, print_result=False):
        '''
        If serf.label_ratio < 0.3, error:
        _classification.py:1334: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
        '''
        self.r_default = []
        self.r_balanced = []
        self.min = min
        self.max = max
        self.inc = inc

        self.preprocess_data_1()
        runs = np.arange(self.min, self.max, self.inc)
        for i in range(len(runs)):
            print('Process: {0}/{1} {2:.2f}%'.format(i +
                  1, len(runs), ((i+1)/len(runs)*100)), end='\r')
            self.label_ratio = runs[i] * 0.01
            self.preprocess_data_2()
            r_default = self.deploy_model(default=True)
            r_balanced = self.deploy_model()
            self.r_default += r_default
            self.r_balanced += r_balanced
        self.r_default = pd.DataFrame(self.r_default, columns=[
            'model', 'parameter_state', 'label_ratio', 'accuracy', 'precision', 'recall', 'f1'])
        self.r_balanced = pd.DataFrame(self.r_balanced, columns=[
            'model', 'parameter_state', 'label_ratio', 'accuracy', 'precision', 'recall', 'f1'])
        print('\nDefault: \n') if print_result else None
        print(self.r_default) if print_result else None
        print('\nBalanced: \n') if print_result else None
        print(self.r_balanced) if print_result else None

    # =========================================================================
    # statistics

    def st_describe(self, print_result=False):
        '''
        1. Describe full dataset
        '''
        self.r_default_describe = self.r_default.describe()
        self.r_balanced_describe = self.r_balanced.describe()
        print()
        print('\nr_default_describe\n') if print_result else None
        print(self.r_default_describe) if print_result else None
        print('\nr_balanced_describe\n') if print_result else None
        print(self.r_balanced_describe) if print_result else None

    # =========================================================================
    # plot

    def plt_1(self, show_plot=False):
        '''
        Plot default model
        x = recall
        y = precision
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x='recall', y='precision',
                     hue='model', data=self.r_default)
        plt.savefig(self.plt_export_path + 'default_model_rp.png')
        plt.show() if show_plot else None

    def plt_2(self, show_plot=False):
        '''
        Plot default model
        x = recall
        y = precision
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x='recall', y='precision',
                     hue='model', data=self.r_balanced)
        plt.savefig(self.plt_export_path + 'balanced_model_rp.png')
        plt.show() if show_plot else None

    def plt_3_1(self, show_plot=False):
        '''
        Plot default model
        x = idex
        y = label_ratio
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_default.index,
                     y=self.r_default.label_ratio, data=self.r_default)
        plt.title('Default Model - Label Ratio')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'default_model_lr.png')
        plt.show() if show_plot else None

    def plt_3_2(self, show_plot=False):
        '''
        Plot default model
        x = idex
        y = accuracy
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_default.index,
                     y=self.r_default.accuracy, data=self.r_default)
        plt.title('Default Model - Accuracy')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'default_model_acc.png')
        plt.show() if show_plot else None

    def plt_3_3(self, show_plot=False):
        '''
        Plot default model
        x = idex
        y = precision
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_default.index,
                     y=self.r_default.precision, data=self.r_default)
        plt.title('Default Model - Precision')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'default_model_pre.png')
        plt.show() if show_plot else None

    def plt_3_4(self, show_plot=False):
        '''
        Plot default model
        x = idex
        y = recall
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_default.index,
                     y=self.r_default.recall, data=self.r_default)
        plt.title('Default Model - Recall')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'default_model_rec.png')
        plt.show() if show_plot else None

    def plt_3_5(self, show_plot=False):
        '''
        Plot default model
        x = idex
        y = f1
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_default.index,
                     y=self.r_default.f1, data=self.r_default)
        plt.title('Default Model - F1')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'default_model_f1.png')
        plt.show() if show_plot else None

    def plt_4_1(self, show_plot=False):
        '''
        Plot balanced model
        x = idex
        y = label_ratio
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_balanced.index,
                     y=self.r_balanced.label_ratio, data=self.r_balanced)
        plt.title('Balanced Model - Label Ratio')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'balanced_model_lr.png')
        plt.show() if show_plot else None

    def plt_4_2(self, show_plot=False):
        '''
        Plot balanced model
        x = idex
        y = accuracy
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_balanced.index,
                     y=self.r_balanced.accuracy, data=self.r_balanced)
        plt.title('Balanced Model - Accuracy')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'balanced_model_acc.png')
        plt.show() if show_plot else None

    def plt_4_3(self, show_plot=False):
        '''
        Plot balanced model
        x = idex
        y = precision
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_balanced.index,
                     y=self.r_balanced.precision, data=self.r_balanced)
        plt.title('Balanced Model - Precision')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'balanced_model_pre.png')
        plt.show() if show_plot else None

    def plt_4_4(self, show_plot=False):
        '''
        Plot balanced model
        x = idex
        y = recall
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_balanced.index,
                     y=self.r_balanced.recall, data=self.r_balanced)
        plt.title('Balanced Model - Recall')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'balanced_model_rec.png')
        plt.show() if show_plot else None

    def plt_4_5(self, show_plot=False):
        '''
        Plot balanced model
        x = idex
        y = f1
        '''
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=self.r_balanced.index,
                     y=self.r_balanced.f1, data=self.r_balanced)
        plt.title('Balanced Model - F1')
        plt.xlabel('Index (label_ratio: {0}-{1})'.format(self.min, self.max))
        plt.savefig(self.plt_export_path + 'balanced_model_f1.png')
        plt.show() if show_plot else None


if __name__ == "__main__":
    system('cls')
    print('[LOG] Start executing script...\n')

    path1 = join(getcwd().rstrip('src'), 'data/wdbc.data').replace('\\', '/')
    path2 = join(getcwd().rstrip('src'),
                 'data/Pumpkin_Seeds_Dataset.arff').replace('\\', '/')
    path3 = join(getcwd().rstrip('src'),
                 'pic/ML_w6_module/plt_').replace('\\', '/')

    ml = ML(path1, path3)
    # ml.single_run(print_result=True)
    ml.multi_run()
    ml.st_describe()
    ml.plt_1()
    ml.plt_2()
    ml.plt_3_1()
    ml.plt_3_2()
    ml.plt_3_3()
    ml.plt_3_4()
    ml.plt_3_5()
    ml.plt_4_1()
    ml.plt_4_2()
    ml.plt_4_3()
    ml.plt_4_4()
    ml.plt_4_5()

    print('\n[LOG] Done executing script...')
