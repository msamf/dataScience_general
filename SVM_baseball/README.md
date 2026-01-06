# Support Vector Machines: Strike Zones in Baseball
This project explores the support vector machine (SVM), by using it to predict whether a pitch is a ball or a strike based on its location over the plate. 

NOTE: This project is based on Codecadmey's [SVM Project](https://www.codecademy.com/journeys/data-scientist-ml/paths/dsmlcj-22-machine-learning-ii/tracks/dsmlcj-22-supervised-learning-ii-sv-ms-rm-nb/modules/support-vector-machines-skill-path-fe1996b3-8d55-4eb8-b35b-42b0110f2011-af506b18-77c9-4c9e-85eb-305bc37c29fd/projects/baseball). 

## Data
The data used is baseball statistics from `pybaseball`. 

Player data for each of Aaron Judge, Jose Altuve, and David Ortiz will be extracted for 2 years within the years they were active. This is done with `pybaseball.statcast_batter()` (documentation found [here](https://github.com/jldbc/pybaseball/blob/master/docs/statcast_batter.md)). 

Judge is a very tall player, while Altuve is very short - it is expected that their strike zones would be quite different. 

## Methods
The following is performed
* Data Analysis to look at the information provided
* Data Preparation to extract and clean the features and labels, and split the data into a training and testing set
* Building the SVM, including with hyperparameter tuning 
* Comparing strike zone and model accuracy of the 3 players

## Results and Discussion
Player | Model Accuracy | Gamma | C
-- | -- | -- | -- 
Judge | 0.854 | 10 | 0.1
Altuve | 0.861 | 1 | 5 
Ortiz | 0.859 | 0.05 | 50

By looking at the graphs from the notebook, one can conclude the following about the strike zones of the players: 
* SVM accuracy across all players is comparable 
* Gamma and C values have wide ranges across all players
* the width of the strike zones between all players is about the same
* the height of the strike zone of Judge is taller than that of Altuve - considering Judge is a taller player and Altuve a shorter one, this makes sense
* strike zones are approximately ovals

Some next steps could include exploring other features to see if the model is more accurate in predicting strikes (e.g. adding in features like how many previous strikes exist, the pitcher ID, etc). 
