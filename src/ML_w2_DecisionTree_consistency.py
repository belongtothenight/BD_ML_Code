import pandas as pd
import numpy as np
from os import system, getcwd
from os.path import join
from time import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


print_flag = False
plot_confusion_matrix_flag = False
progress_print_flag = True
statistic_print_flag = True
plot_histogram_flag = True


class decissionTreeOperation():
    def __init__(self):
        pass

    def get_data(self):
        path = join(getcwd(), 'data/wdbc.data').replace('\\', '/')
        self.data = pd.read_csv(path, header=None)
        if print_flag:
            print(self.data)
            print(self.data.shape)
            print(self.data.columns)
            print(self.data.head())

    def set_column_names(self):
        column_names = ['id', 'malignant',
                        'nucleus_mean', 'nucleus_se', 'nucleus_worst',
                        'texture_mean', 'texture_se', 'texture_worst',
                        'perimeter_mean', 'perimeter_se', 'perimeter_worst',
                        'area_mean', 'area_se', 'area_worst',
                        'smoothness_mean', 'smoothness_se', 'smoothness_worst',
                        'compactness_mean', 'compactness_se', 'compactness_worst',
                        'concavity_mean', 'concavity_se', 'concavity_worst',
                        'concave_pts_mean', 'concave_pts_se', 'concave_pts_worst',
                        'symmetry_mean', 'symmetry_se', 'symmetry_worst',
                        'fractal_dim_mean', 'fractal_dim_se', 'fractal_dim_worst'
                        ]

        self.data.columns = column_names
        if print_flag:
            print(self.data.shape)
            print(self.data.columns)
            print(self.data.head())
            self.data.tail(10)

    def make_data_all_numerical(self):
        self.data['malignant'] = self.data['malignant'].map(
            lambda x: 0 if x == 'B' else 1)
        if print_flag:
            self.data.tail(10)

    def split_data_into_train_test(self):
        self.X = self.data.drop(columns=['malignant']).values
        self.y = self.data['malignant'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=2018)

    def build_mldl_model(self):
        self.model = DecisionTreeClassifier()

    def training_mldl_model(self):
        self.model.fit(self.X_train, self.y_train)

    def test_mldl_model(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_the_result(self):
        if print_flag:
            print('y_pred: ' + str(self.y_pred))
            print('y_test: ' + str(self.y_test))
        if self.y_pred.all == self.y_test.all:
            print('Prediction successful, all values are same') if print_flag else None
        else:
            # printing
            self.y_diff = abs(self.y_pred - self.y_test)
            print('Prediction rate: {0}/{1} = {2}%'.format(len(self.y_test)-sum(
                self.diff), len(self.y_test), (len(self.y_test)-sum(self.y_diff))/len(self.y_test)*100)) if print_flag else None
            self.cm = confusion_matrix(self.y_test, self.y_pred).ravel()
            print(
                '[TruePositive, TrueNegative, FalsePositive, FalseNegative] = ' + str(self.cm)) if print_flag else None
            # plotting
            if plot_confusion_matrix_flag:
                ConfusionMatrixDisplay.from_predictions(
                    self.y_test, self.y_pred, display_labels=['True', 'False'])
                plt.show()

    def prepare_data(self):
        self.get_data()
        self.set_column_names()
        self.make_data_all_numerical()
        self.split_data_into_train_test()

    def training_and_testing(self):
        self.build_mldl_model()
        self.training_mldl_model()
        self.test_mldl_model()
        self.evaluate_the_result()

    def single_run(self):
        self.prepare_data()
        self.training_and_testing()

    def multi_run(self, n=10):
        start = time()
        self.prepare_data()
        self.confusion_matrix = []
        n_len = len(str(n))
        for i in range(n):
            print('Progress: {2:{0}d}/{3:{1}d} >> {4:.2f}%'.format(n_len, n_len, i +
                  1, n, (i+1)/n*100), end="\r") if progress_print_flag else None
            self.training_and_testing()
            self.confusion_matrix.append(self.cm)
        self.confusion_matrix = np.array(self.confusion_matrix)

        # statistics data
        self.avg = np.average(self.confusion_matrix, axis=0)
        self.mid = np.median(self.confusion_matrix, axis=0)
        self.std = np.std(self.confusion_matrix, axis=0)
        self.var = np.var(self.confusion_matrix, axis=0)
        self.hist = np.histogram(self.confusion_matrix)
        print('Data: \n' + str(self.confusion_matrix)) if print_flag else None
        if statistic_print_flag:
            print('Average: ' + str(self.avg))
            print('Median: ' + str(self.mid))
            print('Standard Deviation: ' + str(self.std))
            print('Variance: ' + str(self.var))
            print('Histogram: ' + str(self.hist))
        end = time()

        # plot histogram
        if plot_histogram_flag:
            plt_title = ['TruePositive', 'TrueNegative',
                         'FalsePositive', 'FalseNegative']
            self.tp, self.tn, self.fp, self.fn = np.split(
                self.confusion_matrix, 4, axis=1)
            self.re_confusion_matrix = np.array(
                [self.tp, self.tn, self.fp, self.fn])

            self.fig, self.axs = plt.subplots(2, 2)
            self.fig.suptitle(
                'Histogram for {0} times of execution taking {1:.2f} seconds. \n(avg/min/max/mid/std/var)'.format(n, end-start))
            for i in range(0, 4, 1):
                max = np.amax(self.re_confusion_matrix[i])
                min = np.amin(self.re_confusion_matrix[i])
                bins = np.arange(min-0.5, max-0.5+2, 1)
                counts, bins, patches = self.axs[(i)//2, (i+2) %
                                                 2].hist(self.re_confusion_matrix[i], bins=bins, rwidth=0.8, edgecolor='k')
                self.axs[(i)//2, (i+2) %
                         2].set_title(plt_title[i] + ' ({0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}/{5:.2f})'.format(
                             self.avg[i], min, max, self.mid[i], self.std[i], self.var[i]))
                self.axs[(i)//2, (i+2) %
                         2].axvline(self.avg[i], color='r', linestyle='dashed', linewidth=1)
                self.axs[(i)//2, (i+2) % 2].set_xticks(bins)
                self.axs[(i)//2, (i+2) %
                         2].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                bin_centers = 0.5 * np.diff(bins) + bins[:-1]
                for count, x in zip(counts, bin_centers):
                    self.axs[(i)//2, (i+2) % 2].annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
                                                         xytext=(0, -18), textcoords='offset points', va='top', ha='center')
                    percent = '%0.0f%%' % (
                        100 * float(count) / counts.sum())
                    self.axs[(i)//2, (i+2) % 2].annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                                                         xytext=(0, -32), textcoords='offset points', va='top', ha='center')
            # plt.subplot_tool() # adjust subplots sizes
            plt.subplots_adjust(left=0.05, bottom=0.08,
                                right=0.95, top=0.9, wspace=0.1, hspace=0.3)
            plt.show()


if __name__ == "__main__":
    system('cls')
    print('[LOG] Start executing script...\n')
    i = 0
    while True:
        runs = input('How many times do you want to execute? ')
        if runs == 'q':
            break
        try:
            runs = int(runs)
        except:
            print('Please enter a number')
            continue
        i += 1
        print(
            '\n[LOG] Start executing {0} times...===============\n'.format(runs))
        dto = decissionTreeOperation()
        dto.multi_run(runs)
        print(
            '\n[LOG] Execution {0} times finished.==============\n'.format(runs))
    print('\n[LOG] Done executing script...')
