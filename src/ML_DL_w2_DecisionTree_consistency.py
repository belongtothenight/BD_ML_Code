from os import system, getcwd
from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print_flag = False
plot_flag = True


class decissionTreeOperation():
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
            ConfusionMatrixDisplay.from_predictions(
                self.y_test, self.y_pred, display_labels=['True', 'False'])
            plt.show() if plot_flag else None

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


if __name__ == "__main__":
    system('cls')
    print('[LOG] Start executing script...')
    dto = decissionTreeOperation()
    dto.single_run()
    print('[LOG] Done executing script...')
