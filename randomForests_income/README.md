# Random Forests: Income Prediction
This project uses random forests to predict if the income of an individual based on various metrics is greater than 50k. 

NOTE: The project is based on Codecademy's [Random Forests Project](https://www.codecademy.com/journeys/data-scientist-ml/paths/dsmlcj-22-machine-learning-ii/tracks/dsmlcj-22-ensembling/modules/mle-random-forests-fdc2b9a7-f92c-41f9-95f5-72888c03f0af/projects/ml-int-random-forests-project). 

## Dataset
The dataset is from UCI's Machine Learning Repository: [Census Income](https://archive.ics.uci.edu/dataset/20/census+income).

## Methods
The following methods are used:
1. Data exploration and cleaning 
1. Prepare features and labels
1. Build and tune RF classifier (hyperparameter tuning: depth)
1. Create additional features and retune

## Results and Discussion
With the original feature set, the highest test accuracy of 0.8346 was achieved with a max depth of 12. Based on the depth vs. accuracy plot, the testing score had an inverse parabolic shape (i.e. accuracy increased up to a point, and then decreased), while the training score consistently increased with depth until a plateau of about 20. 

With the addition of the education feature, the highest test accuracy of 0.8443 was achieved with a max depth of 9. Overall, this model was more accurate than without the education feature, showing the importance of carefully selecting features. Based on the depth and accuracy plot, the train and test accuracies had similar shapes, plateauing around a depth of 5-7; choosing one of these values as opposed to 9 may help to reduce the complexity and computation time of the model. 

Next steps could include:
* exploring more features; for example, in all likelihood, the "workclass" and "occupation" columns also likely are related to the income and could be strong predictors 
* tune more parameters of the model, such as the number of estimators, etc. 
* use different evaluation metrics; as the income classes are quite unbalanced, it may make sense to use for example f1 score