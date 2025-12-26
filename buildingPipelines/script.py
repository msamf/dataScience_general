import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

from scipy.io import arff

## __DATA LOADING AND INITIAL EXPLORATION
data = arff.loadarff('bone-marrow.arff')
df = pd.DataFrame(data[0])
df.drop(columns=['Disease'], inplace=True)

# convert all columns to numeric, coerce errors to null values
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    
# make sure binary columns are encoded as 0 and 1
for c in df.columns[df.nunique()==2]:
    df[c] = (df[c]==1)*1.0

# calculate the number of unique values for each column
# (this is used to determine categorical vs numerical columns)
print('Count of unique values in each column:')
print(df.nunique())

## __DATA PREPARATION
# set target survival_status as y
# set features (dropping survival status and time) as X
y = df["survival_status"]
X = df.drop(columns=["survival_time", "survival_status"])

# define lists of numeric and categorical columns based on number of unique values
threshold = 7 # 7 will be the threshold ( <= 7 = categorical)
num_cols = X.columns[X.nunique() > threshold]
cat_cols = X.columns[X.nunique() <= threshold]

# determine columns with missing values
print("Columns with missing values: ")
print(X.columns[X.isnull().sum() > 0])

# split data into train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## __PIPELINES
# create categorical preprocessing pipeline
#   use mode to fill in missing values and OHE
cat_vals = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), 
                     ("ohe", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"))])

# create numerical preprocessing pipeline
#   use mean to fill in missing values and standard scaling of features
num_vals = Pipeline([("imputer", SimpleImputer(strategy="mean")), 
                     ("scaler", StandardScaler())])

# create column transformer that will preprocess the numerical and categorical features separately
preprocess = ColumnTransformer(transformers=[("cat_process", cat_vals, cat_cols), 
                                             ("num_process", num_vals, num_cols)])

# create overall pipeline with preprocess, PCA, and a logistic regresssion model
pipeline = Pipeline([("preprocess", preprocess), 
                     ("PCA", PCA()), 
                     ("classifier", LogisticRegression())])

# fit the pipeline on the training data
pipeline.fit(X_train, y_train)
# evaluate pipeline on test data
test_score = pipeline.score(X_test, y_test)
print(f"The pipeline's test score is {test_score}.")

## __HYPERPARAMETER TUNING
# define search space of hyperparameters
search_space = [{'classifier': [LogisticRegression()], 
                 'classifier__C': np.logspace(-4, 2, 10), 
                 'PCA__n_components': np.linspace(30,37,3).astype(int)}]
# search over hyperparameters above to optimize pipeline and fit
gs = GridSearchCV(estimator=pipeline, param_grid=search_space, cv=5)
gs.fit(X_train, y_train)

# save the best estimator from the gridsearch
best_model = gs.best_estimator_
# print attributes and final accuracy on test set
print('The best classification model is:')
print(best_model.named_steps['classifier'])
print('The hyperparameters of the best classification model are:')
print(best_model.named_steps['classifier'].get_params())
print('The number of components selected in the PCA step are:')
print(best_model.named_steps['PCA'].n_components)

# evaluate best pipeline on test data
final_test_score = best_model.score(X_test, y_test)
print(f"With the best model, the test score is {final_test_score}.")