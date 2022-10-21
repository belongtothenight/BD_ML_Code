import pandas as pd
import numpy as np
import copy as copy
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class ML():
    def __init__(self, data_path, print_result=False):
        '''
        1. File validation check.
        2. Dataset validation check.
        3. Import datasets.
        '''
        self.data_path = data_path
        if data_path.endswith('.data'):
            if self.data_path.endswith('wdbc.data'):
                self.dataset = 1.1
            else:
                print('Invalid Dataset')
                return
            self.data = pd.read_csv(data_path, header=None)
        elif data_path.endswith('.arff'):
            if self.data_path.endswith('Pumpkin_Seeds_Dataset.arff'):
                self.dataset = 2.1
            else:
                print('Invalid Dataset')
                return
            self.data = pd.DataFrame(arff.loadarff(data_path)[0])
        else:
            print('Invalid data file')
            return
        print(self.data) if print_result else None

    def preprocess_data(self, print_result=False):
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

    def split_data_into_train_test(self, print_result=False):
        pass

    def build_mldl_model(self, print_result=False):
        pass

    def training_mldl_model(self, print_result=False):
        pass

    def test_mldl_model(self, print_result=False):
        pass

    def evaluate_the_result(self, print_result=False):
        pass

    def prepare_data(self, print_result=False):
        self.preprocess_data()
        self.split_data_into_train_test()

    def training_and_testing(self, print_result=False):
        self.build_mldl_model()
        self.training_mldl_model()
        self.test_mldl_model()
        self.evaluate_the_result()

    def single_run(self, print_result=False):
        self.prepare_data()
        self.training_and_testing()

    def multi_run(self, print_result=False):
        print('multi_run')


if __name__ == "__main__":
    system('cls')
    print('[LOG] Start executing script...\n')

    path1 = join(getcwd().rstrip('src'),
                 'data/Pumpkin_Seeds_Dataset.arff').replace('\\', '/')
    path2 = join(getcwd().rstrip('src'), 'data/wdbc.data').replace('\\', '/')

    ml = ML(path2)
    ml.single_run()

    print('\n[LOG] Done executing script...')
