import math
import pandas as pd
import numpy as np
import copy as copy
import statistics as stt
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


class DataSetError(Exception):
    """Base class for other exceptions"""
    pass


class FileTypeError(Exception):
    """Base class for other exceptions"""
    pass


class ML():
    # =========================================================================
    # core function
    def __init__(self, data_path, print_result=False):
        '''
        1. File validation check.
        2. Dataset validation check.
        3. Import datasets.
        '''
        self.label_ratio = 0.5
        self.data_path = data_path
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
        print('Label Ratio: ', self.label_ratio)
        print('Test Size: ', Test_Size)
        print('Max Train Size: ', max_train0_size)
        print('X_0 Size: ', X_0.shape[0])
        print('X_1 Size: ', X_1.shape[0])
        print('Test Size From 0: ', TestSizeFrom0)
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
        Result = [model, parameter_state, label_ratio, accuracy, precision, recall, f1]
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
    # core extension function

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
        result = []
        self.preprocess_data_1()
        for i in np.arange(min, max, inc):
            self.label_ratio = i * 0.01
            self.preprocess_data_2()
            r_default = self.deploy_model(default=True)
            r_balanced = self.deploy_model()
            result.append(r_default)
            result.append(r_balanced)
        print(result) if print_result else None
        return result


if __name__ == "__main__":
    system('cls')
    print('[LOG] Start executing script...\n')

    path2 = join(getcwd().rstrip('src'),
                 'data/Pumpkin_Seeds_Dataset.arff').replace('\\', '/')
    path1 = join(getcwd().rstrip('src'), 'data/wdbc.data').replace('\\', '/')

    ml = ML(path1)
    # ml.single_run(print_result=True)
    ml.multi_run(print_result=True)

    print('\n[LOG] Done executing script...')
