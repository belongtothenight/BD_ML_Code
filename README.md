# BD_ML_Code

![Repo Size](https://img.shields.io/github/repo-size/belongtothenight/BD_ML_Code) ![Code Size](https://img.shields.io/github/languages/code-size/belongtothenight/BD_ML_Code) ![File Count](https://img.shields.io/github/directory-file-count/belongtothenight/BD_ML_Code/src) ![Commit Per Month](https://img.shields.io/github/commit-activity/m/belongtothenight/BD_ML_Code)

This repo contains all the codes from BD_ML course.

## Notice

Scripts in this repo needs to be executed with IDE with the required libraries installed in local environment.

## Development Environment

- Windows 11

- Anaconda Jupyter Notebook
  - Keras
  - TensorFlow

- Python
  - pandas
  - numpy
  - sklearn
  - matplotlib

## File Description

| No. | Filename                                                                                                                                                    | Description                                                                                                                             |
| :-: | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
|  1  | [ML_w2_BasicProgStructure.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w2_BasicProgStructure.ipynb)                               | Original Decision Tree algorithm demonstration in Jupyter Notebook.                                                                     |
|  2  | [ML_w2_DecisionTree_basic.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w2_DecisionTree_basic.py)                                     | Perform single time of Decision Tree training and testing from tumor cell data.                                                         |
|  3  | [ML_w2_DecisionTree_consistency.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w2_DecisionTree_consistency.py)                         | Perform multiple times of Decision Tree training and testing from tumor cell data in order to test Decision Tree algorithm consistency. |
|  4  | [BD_w3_OWID_COVID19_class.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w3_OWID_COVID19_class.ipynb)                               | Code follows teacher's video instruction.                                                                                               |
|  5  | [BD_w3_OWID_COVID19_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w3_OWID_COVID19_hw.ipynb)                                     | BD w3 Homework.                                                                                                                         |
|  6  | [ML_w3_BasicProgStructure_MoreModel.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_BasicProgStructure_MoreModel.ipynb)           | Add different models.                                                                                                                   |
|  7  | [ML_w3_BasicProgStructure_AddStdScaler.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_BasicProgStructure_AddStdScaler.ipynb)     | Add scaler to preprocess data before using models.                                                                                      |
|  8  | [ML_w3_hw_q1.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_hw_q1.ipynb)                                                         | Question inside, output file in data/w3q1data/                                                                                          |
|  9  | [ML_w3_hw_q2.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w3_hw_q2.ipynb)                                                         | Question inside, output file in data/w3q2data/                                                                                          |
| 10  | [BD_w3_OWID_COVID19_hw_adjusted.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w3_OWID_COVID19_hw_adjusted.ipynb)                   | Adjusted code based on hw last week and teacher's answer.                                                                               |
| 11  | [BD_w4_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w4_hw.ipynb)                                                               | Different approach to display specified columns from the original dataset.                                                              |
| 12  | [ML_w4_PrepareTrainingSetWithFewerCancer.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w4_PrepareTrainingSetWithFewerCancer.ipynb) | Showcase dropping some rows of cancer data can improve accuracy.                                                                        |
| 13  | [ML_w4_hw_q1.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w4_hw_q1.ipynb)                                                         | Try tuning the ratio between cancer positive data and cancer negative data to acheive higher accuracy.                                  |
| 14  | [ML_w4_hw_q2.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w4_hw_q2.ipynb)                                                         | Try removing some features of the original dataset to see whether this can improve accuracy.                                            |
| 15  | [ML_w5_TuningOnTrainData_BASIC.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w5_TuningOnTrainData_BASIC.ipynb)                     | W5 class material + hw.                                                                                                                 |
| 16  | [BD_w6_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w6_hw.ipynb)                                                               | W6 class homework. Q1: scatter plot data. Q2: plot two data comparison.                                                                 |
| 17  | [ML_w6_Homework_wdbc.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w6_Homework_wdbc.ipynb)                                         | ML result analysis with wdbc cancer cell dataset.                                                                                       |
| 18  | [ML_w6_Homework_Pumpkin.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w6_Homework_Pumpkin.ipynb)                                   | ML result analysis with Pumpkin seed dataset.                                                                                           |
| 19  | [ML_w6_module.py](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w6_module.py)                                                             | Modulize w6 homeworks.                                                                                                                  |
| 20  | [BD_w7_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/BD_w7_hw.ipynb)                                                               | Analysis the cuases leading to the trend of "%death by cases"                                                                           |
| 21  | [ML_w7_hw.ipynb](https://github.com/belongtothenight/BD_ML_Code/blob/main/src/ML_w7_hw.ipynb)                                                               | Machine learning with mostly text data.                                                                                                 |

## Optional Code ideas

1. ML_w2_DecisionTree_DevelopmentTrend.py : use the data gathered while looping the algorithm and display as bar chart or line graph.
2. ML_w3_MultiModelComparison.py : compare different models mentione in class.
3. ML_w3_StdScalerPerformance.py : compare the difference with and without using scaler.

## BD Process

## ML Process

1. Prepare/Preprocess Data
   1. Read dataset from file. (pandas/scipy.io.arff/python)
   2. Check features (X) correlation with result (y). Not necessary, but might help reduce time and not misleading algorithm. (pandas)
   3. Deal with missing values (NAN/NA). (pandas)
   4. Convert non-numirical data to numbers representing them. (pandas)
   5. (optional) Balance out imbalance dataset. (balance: one category of result like y=0 has a lot more data(rows) than the other/others.) (python)
   6. Split features (X) and result (y). (python)
   7. (optional) Scale dataset. (sklearn)
   8. Split dataset into either train+test or train+cross-validation+test subsets. (random state is optional) (sklearn)
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
   6. seperate result from different models (pandas->groupby)
   7. scale ratio (dataset/python)
   8. label ratio (dataset/python)
   9. model (sklearn)
   10. model parameters (ex: classweight) (sklearn)
