# BD_ML_Code

<!-- ![Repo Size](https://img.shields.io/github/repo-size/belongtothenight/BD_ML_Code) ![Code Size](https://img.shields.io/github/languages/code-size/belongtothenight/BD_ML_Code) ![File Count](https://img.shields.io/github/directory-file-count/belongtothenight/BD_ML_Code/src) ![Commit Per Month](https://img.shields.io/github/commit-activity/m/belongtothenight/BD_ML_Code) -->

This repo contains all the codes from BD_ML course.

## Notice

Scripts in this repo need to be executed with IDE with the required libraries installed in the local environment.<br><br>

## Development Environment

- Windows 11
  - Keras
  - TensorFlow (with GPU acceleration, check how to install below)
  - pandas
  - numpy
  - sklearn
  - matplotlib
  - wandb

- Anaconda Jupyter Notebook
  - Keras
  - TensorFlow
  - pandas
  - numpy
  - sklearn
  - matplotlib

## Utilized Datasets

| No. | Name                                            | Source                                                                                        |
| --- | ----------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1   | Banking Dataset (EDA and binary classification) | <https://www.kaggle.com/code/rashmiranu/banking-dataset-eda-and-binary-classification/data>   |
| 2   | CIFAR-10/100                                    | <https://www.cs.toronto.edu/~kriz/cifar.html>                                                 |
| 3   | COVID-19 Dataset by Our World in Data           | <https://github.com/owid/covid-19-data>                                                       |
| 4   | Pumpkin_Seeds_Dataset                           | <https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset>                     |
| 5   | Breast cancer wisconsin                         | <https://archive/ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data> |
| 6   | World Mortality Dataset                         | <https://github.com/akarlinsky/world_mortality>                                               |
| 7   | MINST in Tensorflow                             | <https://www.tensorflow.org/datasets/catalog/mnist>                                           |
| 8   | Taiwan Death Detail                             | Unknown                                                                                       |
| 9   | House Price                                     | <https://kaggle.com/datasets/rsizem2/house-prices-ames-cleaned-dataset>                       |

## File Description

| No. | Filename                                                                                                                                                    | Description                                                                                                                                        |
| :-: | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1  | [ML_w2_BasicProgStructure.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w2_BasicProgStructure.ipynb)                               | Original Decision Tree algorithm demonstration in Jupyter Notebook.                                                                                |
|  2  | [ML_w2_DecisionTree_basic.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w2_DecisionTree_basic.py)                                     | Perform single time of Decision Tree training and testing from tumor cell data.                                                                    |
|  3  | [ML_w2_DecisionTree_consistency.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w2_DecisionTree_consistency.py)                         | Perform multiple times of Decision Tree training and testing from tumor cell data in order to test Decision Tree algorithm consistency.            |
|  4  | [BD_w3_OWID_COVID19_class.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w3_OWID_COVID19_class.ipynb)                               | Code follows teacher's video instruction.                                                                                                          |
|  5  | [BD_w3_OWID_COVID19_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w3_OWID_COVID19_hw.ipynb)                                     | BD w3 Homework.                                                                                                                                    |
|  6  | [ML_w3_BasicProgStructure_MoreModel.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_BasicProgStructure_MoreModel.ipynb)           | Add different models.                                                                                                                              |
|  7  | [ML_w3_BasicProgStructure_AddStdScaler.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_BasicProgStructure_AddStdScaler.ipynb)     | Add scaler to preprocess data before using models.                                                                                                 |
|  8  | [ML_w3_hw_q1.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_hw_q1.ipynb)                                                         | Question inside, output file in data/w3q1data/                                                                                                     |
|  9  | [ML_w3_hw_q2.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_hw_q2.ipynb)                                                         | Question inside, output file in data/w3q2data/                                                                                                     |
| 10  | [BD_w3_OWID_COVID19_hw_adjusted.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w3_OWID_COVID19_hw_adjusted.ipynb)                   | Adjusted code based on hw last week and teacher's answer.                                                                                          |
| 11  | [BD_w4_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w4_hw.ipynb)                                                               | Different approach to display specified columns from the original dataset.                                                                         |
| 12  | [ML_w4_PrepareTrainingSetWithFewerCancer.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w4_PrepareTrainingSetWithFewerCancer.ipynb) | Showcase dropping some rows of cancer data can improve accuracy.                                                                                   |
| 13  | [ML_w4_hw_q1.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w4_hw_q1.ipynb)                                                         | Try tuning the ratio between cancer positive data and cancer negative data to acheive higher accuracy.                                             |
| 14  | [ML_w4_hw_q2.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w4_hw_q2.ipynb)                                                         | Try removing some features of the original dataset to see whether this can improve accuracy.                                                       |
| 15  | [ML_w5_TuningOnTrainData_BASIC.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w5_TuningOnTrainData_BASIC.ipynb)                     | W5 class material + hw.                                                                                                                            |
| 16  | [BD_w6_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w6_hw.ipynb)                                                               | W6 class homework. Q1: scatter plot data. Q2: plot two data comparison.                                                                            |
| 17  | [ML_w6_Homework_wdbc.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w6_Homework_wdbc.ipynb)                                         | ML result analysis with wdbc cancer cell dataset.                                                                                                  |
| 18  | [ML_w6_Homework_Pumpkin.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w6_Homework_Pumpkin.ipynb)                                   | ML result analysis with Pumpkin seed dataset.                                                                                                      |
| 19  | [ML_w6_module.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w6_module.py)                                                             | Modulize w6 homeworks.                                                                                                                             |
| 20  | [BD_w7_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w7_hw.ipynb)                                                               | Analysis the cuases leading to the trend of "%death by cases"                                                                                      |
| 21  | [ML_w7_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w7_hw.ipynb)                                                               | Machine learning with mostly text data.                                                                                                            |
| 22  | [BD_midterm.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_midterm.ipynb)                                                           | BD midterm answer.                                                                                                                                 |
| 23  | [ML_midterm.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_midterm.ipynb)                                                           | ML midterm answer.                                                                                                                                 |
| 24  | [BD_w10_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w10_hw.ipynb)                                                             | COVID-19 caused excess death across the glob.                                                                                                      |
| 25  | [ML_w10_cn.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w10_cn.ipynb)                                                             | Overfitting, details about DT and RF.                                                                                                              |
| 26  | [ML_w10_DT_RF_DiveIn_Homework.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w10_DT_RF_DiveIn_Homework.ipynb)                       | Plot tree structure of trained result of given dataset.                                                                                            |
| 27  | [BD_w11_ExcessDeathAndCovidDeathPhase1.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w11_ExcessDeathAndCovidDeathPhase1.ipynb)     | A clean and fast way to calculate excess death.                                                                                                    |
| 28  | [BD_w11_ReadMeFirst.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w11_ReadMeFirst.ipynb)                                           | How the excess death calculation approach is not logical.                                                                                          |
| 29  | [ML_w11_SVM_Homework.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w11_SVM_Homework.ipynb)                                         | Details of SVM.                                                                                                                                    |
| 30  | [BD_w12_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w12_hw.ipynb)                                                             | Demonstrate excess death changes in the range of 2020 to 2022.                                                                                     |
| 31  | [ML_w12_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w12_hw.ipynb)                                                             | Start using tf.keras                                                                                                                               |
| 32  | [BD_w13_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w13_hw.ipynb)                                                             | Adjust result of BD_w12_hw.jpynb.                                                                                                                  |
| 33  | [ML_w13_redoprehw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w13_redoprehw.ipynb)                                               | Redo "ML_w12_hw" with some adjustments.                                                                                                            |
| 34  | [ML_w13_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w13_hw.ipynb)                                                             | CNN on CIFAR 10/100 dataset.                                                                                                                       |
| 35  | [ML_w14_redoprehw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w14_redoprehw.ipynb)                                               | CNN model fit successfully and accelerated by GPU                                                                                                  |
| 36  | [ML_w14_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w14_hw.ipynb)                                                             | Build CNN module for multiple models and extra features like export models, learning curve, prediction image.                                      |
| 37  | [ML_w14_hw.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w14_hw.py)                                                                   | Loop executing built CNN module in both building and testing models.                                                                               |
| 38  | [BD_w14_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w14_hw.ipynb)                                                             | Try to use really messed up Taiwan COVID-19 Death Detail dataset.                                                                                  |
| 39  | [BD_w15_redoprehw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w15_redoprehw.ipynb)                                               | Redo COVID-19 mortality scatter plot with correct result.                                                                                          |
| 40  | [BD_w15_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w15_hw.ipynb)                                                             | Try to use really messed up Taiwan COVID-19 Death Detail dataset.                                                                                  |
| 41  | [BD_w16_class.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w15_class.ipynb)                                                       | Processing dataset column with complex content.                                                                                                    |
| 42  | [ML_w16_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w16_hw.ipynb)                                                             | Build CNN module for multiple models and extra features like export models, learning curve, prediction image, and module for preprocessing images. |
| 43  | [ML_w16_hw.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w16_hw.py)                                                                   | execute modules in ML_w17_hw.ipynb.                                                                                                                |

## Optional Code ideas

1. ML_w2_DecisionTree_DevelopmentTrend.py: use the data gathered while looping the algorithm and display it as a bar chart or line graph.
2. ML_w3_MultiModelComparison.py: compare different models mentioned in class.
3. ML_w3_StdScalerPerformance.py: compare the difference with and without using the scaler.

## Developing Process

### BD Process

1. Prepare/Preprocess Data
   1. Remove unused columns.
   2. Convert datatype. (str2timestamp, str2int, str2float, obj2str...)
2. Plot Data (scatter not plot)
   1. Column to column.
   2. Interaction between two columns.

Don't use 'concate' in most cases since some data in column might be missing, and it can cause a lot of errors.

### ML Process

1. Prepare/Preprocess Data (the dataset needs to be appropriate for the question)
   1. Read the dataset from the file. (pandas/scipy.io.arff/python)
   2. Check features (X) correlation with the result (y), and drop features with low correlation. Not necessary, but might help reduce time and not mislead the algorithm. (pandas)
   3. Deal with missing values (NAN/NA). (pandas)
   4. Convert non-numerical data to numbers representing them. (pandas)
   5. (optional) Balance out the imbalance dataset. (balance: one category of result like y=0 has a lot more data(rows) than the other/others.) (python)
   6. Split features (X) and result (y). (python)
   7. (optional) Scale dataset. (sklearn)
   8. Split the dataset into either train+test or train+cross-validation+test subsets. (random state is optional) (sklearn)
2. Deploy Model
   1. Select model. (sklearn->supervised/unsupervised)
      1. Logistic Regression, LR.
      2. Decision Tree, DT.
      3. Random Forest, RF.
      4. Support Vector Machine, SVM. (SVC)
      5. K-Nearest Neighbor, KNN.
   2. Train model. (sklearn->fit)
   3. Test model. (sklearn->pred)
3. Result Analysis
   1. accuracy (sklearn)
   2. precision (sklearn)
   3. recall (sklearn)
   4. f1 (sklearn)
   5. confusion matrix (sklearn)
   6. separate results from different models (pandas->groupby)
   7. scale ratio (dataset/python)
   8. label ratio (dataset/python)
   9. model (sklearn)
   10. model parameters (ex: classweight) (sklearn)

**Need to code a module for the final test:**<br>

- ML accessible features: process time (train/test), dataset size, weight, scale info, acc, pre, rec, f1, fitting based on iteration<br>
   1. supports multiple different datasets
   2. preprocess dataset and comment info about accessible features
   3. support for multi-run tests to average out accessible features
   4. supports multiple algorithms
   5. with commends to test different stuff about accessible features
   6. provide a table for questions like Q9 in ML_midterm.jpynb.
- BD accessible features:
   1. supports multiple different datasets
   2. preprocess dataset
   3. support all questions asked in hw

### NN Models

- It might perform better if the output node is strictly 1 result per node. (ML_w12_hw/w13 videos).
- It might have better performance if the input data is properly scaled.
- Sometimes it's not necessary to specify batch_size.

## Enable GPU acceleration when using Keras or Tensorflow (20221216)

### Steps

1. Install Visual Studio Code
2. Install Python 3.10.8
3. Install VC++ <https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170>
4. Install NVIDIA GPU drivers <https://www.nvidia.com/download/index.aspx?lang=en-us>
5. Install CUDA Toolkit 11.2 <https://developer.nvidia.com/cuda-toolkit-archive>
6. Install cuDNN SDK 8.1.0 <https://developer.nvidia.com/rdp/cudnn-archive>

### Verification

1. Verify Installation <https://www.tensorflow.org/install/pip> - 7. Verify install
2. Enable Tensorflow GPU acceleration: <https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu>

### Sources

1. <https://www.tensorflow.org/install/source#gpu>
2. <https://www.tensorflow.org/install/pip>
3. <https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu>
