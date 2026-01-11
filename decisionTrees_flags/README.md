# Decision Trees: Flags
This project uses decision trees to predict which continent a particular flag (based on its characteristics) comes from. 
NOTE: This project is based on Codecademy's [Flags project](https://www.codecademy.com/journeys/data-scientist-ml/paths/dsmlcj-22-machine-learning-i/tracks/dsmlcj-22-supervised-learning-i/modules/mle-decision-trees-8b27e5b3-0352-4887-a8e5-e4a507597ad5-fc6f7fa3-d862-4a3c-9a50-e939437fdd97/projects/mlfun-decision-trees-project).

## Dataset
The dataset used in from the UCI Machine Learning Repository: [Flags](https://archive.ics.uci.edu/dataset/40/flags). "Additional Variable Information" provides further information about the columns and what the values mean (e.g. landmass:	1=N.America, 2=S.America, 3=Europe, 4=Africa, 4=Asia, 6=Oceania). 

"Flags," UCI Machine Learning Repository, 1990. [Online]. Available: https://doi.org/10.24432/C52C7Z.

## Methods
1. Data import and exploration - understand the data
1. Data preparation - create feature set (with variable encoding) and label set
1. Decision Tree modelling 
    * Tune hyperparameters of depth and pruning 
    * visualize hyperparameter values vs accuracy 
    * visualize final "best" decision tree 

## Results and Discussion
The best model is achieved with depth 5 and pruning 0.0379, giving an overall score of 0.7727; this accuracy is fine, but not great. Based on visualizing the tree, the first splitting parameter is "mainhue = blue". 

Some next steps can include:
* multiclass classification (there are 6 landmasses, but we only used data from 2)
* feature selection to have more important predictor variables 
* tune more parameters of the the Decision Tree
