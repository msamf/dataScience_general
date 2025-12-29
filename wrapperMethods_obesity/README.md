# Wrapper Methods: Obesity

## Introduction
This project explores wrapper methods to choose a smaller feature subset to fit a logistic regression (LR) model which predicts obesity based on lifestyle. Wrapper methods explored are: sequential forward selection, sequential backward floating selection, and recursive feature elimination. Model accuracy will be used to evaluate and compare the methods.
Note: This project is based on Codecademy's project on [Wrapper Methods](https://www.codecademy.com/paths/fe-path-feature-engineering/tracks/fe-feature-selection-methods/modules/fe-wrapper-methods/projects/fe-wrapper-methods-project). 

## Dataset
The dataset estimates obesity from eating and physical conditions from survey results, and was obtained from the UCI Machine Learning Repository: [Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). 

## Method
The following steps are performed:
1. Data exploration
1. Data preparation
1. Baseline LR model and evaluation
1. LR Model and evaluation based on feature selection using the following wrapper methods:
    * Sequential forward selection
    * Sequential backward floating selection
    * Recursive feature elimination
1. Comparing results of wrapper methods

## Conclusion

The results of all investigations are summarized below:
Method | Number of features | Training Score | Chosen Features
-- | -- | -- | -- 
(base) | 17 | 0.7542315504400813 | (all)
SFS | 7 | 0.7813134732566012 | 'Gender', 'Age', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SCC', 'FAF'
SBS | 7 | 0.7596479350033852 | 'Age', 'family_history_with_overweight', 'FAVC', 'FCVC', 'CAEC', 'SCC', 'Public_Transportation'
RFE | 8 | 0.7610020311442113 | 'Age', 'family_history_with_overweight', 'FAVC', 'FCVC', 'CAEC', 'SCC', 'Automobile', 'Walking'

Overall, all feature reduction methods had a higher accuracy than the base model with all features, even if the improvement was marginal for some methods. SFS had the highest improvement. 
All three methods had common features chosen:  'Age', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SCC'- showing the importance of these features.