# BONE MARROW CLASSIFICATION
This project uses machine learning to predict the survival status of patients with bone marrow disease. Pipelines are written in Python to achieve this task.

## Data Loading and Initial Exploration
The data is initially loaded and explored to understand the context, content, and data types of all columns.
### Dataset
The dataset is bone-marrow.arff, taken from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children). It is a multivariate dataset with integer values. 

## Data Preparation
The target variable, "survival_status" is set as y, while all columns excluding "survival_status" and "survival_time" are predictive features and set as X. Based on the number of unique values in each of the feature columns, features are set as either numerical and categorical. A train-test split is then conducted. 

## Pipelines
Pipelines are created for both numerical and categorical data, to fill in the missing values, and perform one-hot encoding (categorical) or standardize values (numerical). A column transformer is then applied to the features.
The overall pipeline applies the above-mentioned preprocessing, performs PCA, and implements a logistic regression model. The pipeline is then fitted to the training data, and applied to the test data for model evaluation.

## Hyperparameter Tuning
Optimization is done through hyperparameter tuning of the number of components in the PCA step and regularization value C in logistic regression step. A grid search is used for tuning. The best model is then outputted, and its score is compared to the score of the initial pipeline. 

## Results
The initial pipeline's test score is 0.658, while the tuned pipeline's test score is 0.71. This demonstrates the importance of hyperparameter tuning, but suggests that more tuning (including different dimensionality reduction or a different machine learning algorithm) may be more appropriate for the data. 

NOTE: This project is based on Codecademy's [Building ML Pipelines project](https://www.codecademy.com/paths/machine-learning-engineer/tracks/mle-pipelines-track/modules/build-ml-pipelines-module/projects/mle-pipelines-project).