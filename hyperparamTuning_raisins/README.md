# Hyperparameter Tuning: Raisins

This project explores hyperparameter tuning (GridSearchCV, RandomSearchCV) through classifying raisins into 1 of 2 types. The models used are Logistic Regression and Decision Trees.

NOTE: This project is based on Codecademy's [Hyperparameter Tuning project](https://www.codecademy.com/projects/practice/mle-hyperparameter-tuning-project).

## Dataset
The dataset comes from Kaggle: [Raisin](https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset), by Murat Koklu. Briefly this dataset contains 7 features related to raisin properties and has two classes: 'Kecimen' and 'Besni'.

## Methods
The following steps were taken:
* Data analysis 
* Data preprocessing: encode the class column, split into features and labels, split into training and testing set
* Decision Tree classifier, tuned with GridSearchCV (swept parameters are min_samples_split, and max_depth)
* Logistic Regression classifier, tuned with RandomizedSearchCV (swept parameters are penalty and C value)

## Results and Discussion
* GridSearchCV with the decision tree achieved a best training score of 0.869 (min: 0.846) with max_depth = 3 and no impact of min_samples_split. Test score was 0.827. 
* RandomizedSearchCV with the logistic regression model achieved a best training score of 0.877 (min: 0.871) with C = 7.94 and penalty = L2. Test score was 0.827. 

Overall, it is important to tune hyperparameters to achieve the best model.