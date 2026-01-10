# Email Similarity with Naive Bayes Classifer
This project uses a Naive Bayes classifier to classify emails from several datasets; by comparing the accuracy, we can find which emails are harder to distinguish from others. 

NOTE: This project is based on Codecademy's [Email Similarity project](https://www.codecademy.com/journeys/data-scientist-ml/paths/dsmlcj-22-machine-learning-ii/tracks/dsmlcj-22-supervised-learning-ii-sv-ms-rm-nb/modules/naive-bayes-classifier-skill-path-3ca45f60-810b-4e2a-b234-149f1c5d88f5-fab544fa-f232-4f73-bee4-aa3017772436/projects/naive-bayes-project). 

## Datasets
Email data used is from `sklearn.datasets`. 

## Methods
1. Import and explore data
1. Prepare training and testing set
1. Count words in the emails to input to model 
1. Create and train Naive Bayes model 
1. Evaluate model 
1. Compare Model on different pairs of email categories

## Results and Discussion
Category 1 | Category 2 | Train Score | Test Score 
-- | -- | -- | --
Hockey | Baseball | 0.9975 | 0.9724
Hockey | Computer Graphics | 0.9958 | 0.9962
Medicine | Space | 0.9992 | 0.9823

Overall, the NB model does very well in classifying emails to the right category. The best performance was for hockey vs. computer graphics with a test score of 0.9962. Since these are quite different concepts, it's expected that it would be easy to distinguish. Even hockey and baseball, which are both sports and thus may have some similarities in terms of common words like "game", "player", "score", "sport" etc, the model performed well. 