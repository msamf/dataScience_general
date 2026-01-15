# KNN: Cancer Classification 
This project uses a KNN classifier to predict if a patient has breast cancer. 

NOTE: This project is based on Codecademy's [KNN project](https://www.codecademy.com/journeys/data-scientist-ml/paths/dsmlcj-22-machine-learning-i/tracks/dsmlcj-22-supervised-learning-i/modules/mle-k-nearest-neighbors-e187011d-38bf-4df9-9ef2-6675fa0da752-046f6493-ceeb-4a43-b2de-dff3dd50b8d0/projects/knn-project). 

## Dataset
The dataset is from `sklearn`: breast_cancer. 

## Methods
The following steps are performed:
1. **Analyze data**: Data is briefly analyzed through looking at one observation with its target, looking at the possible target values and their encoding, descriptive statistics, and medians of all features grouped by target values. 
1. **KNN Classifier**: A KNN classifier is fit on a training set, and evaluated on a testing set. 
1. **Hyperparameter tuning**: The k value (driving how many neighbouring points are considered when making a prediction for a data point) is parameter-swept, to see its impact on the test score. 

## Results and Discussion
The best test score occurs at a k-value of 23 (test score of 0.964). 

Some next steps can include:
* feature reduction through PCA or other feature selection methods 
* exploring different algorithms beyond KNN